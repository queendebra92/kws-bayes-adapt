"""Train and evaluate the TC-ResNet8 baseline (B2) under the same protocol as run_baselines.

    python -m kwsba.run_tcresnet --musan-root <musan> --cache-dir cache --out results_tcresnet

Protocol (identical to the Gaussian-Bayes baselines wherever it applies):
  * same cached splits, same MFCC front end (40 coefficients here), same test-noise
    realisations: condition (kind, snr) and seed s use cond_rng(s, kind, snr), so a given
    seed's noisy test clips are bit-identical to those scored by run_baselines.
  * seed s controls weight initialisation, data order, time-shift augmentation AND the test
    noise realisation. Its std therefore includes training variability.
  * model selection: the epoch with the best macro-F1 on the validation split is kept
    (clean validation for the clean-trained model; clean + noisy validation for -aug).
    The test split is never used for any choice.
  * -aug trains on clean + one noisy copy of every training clip (the same noisy audio as
    B1-aug: aug_transform(pool, 777, "train")).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import LABELS, load_split
from .features import mfcc
from .noise import NOISE_KINDS, NoisePool, add_noise, to_float
from .run_baselines import BATCH, SNRS, accuracy, aug_transform, cond_rng, macro_f1
from .tcresnet import TCResNet8, n_params, time_shift

N_MFCC = 40
MAX_SHIFT = 10  # frames (100 ms)


def mfcc40_of(wav, transform=None) -> np.ndarray:
    """[N, 40, 98] float32 MFCCs for a (memmap) int16 array, in batches of BATCH."""
    out = []
    for s in range(0, len(wav), BATCH):
        b = wav[s:s + BATCH]
        x = transform(b) if transform is not None else to_float(b)
        out.append(mfcc(x, N_MFCC))
    return np.concatenate(out)


def cached(path: Path, fn) -> np.ndarray:
    if path.exists():
        return np.load(path)
    x = fn().astype(np.float16)
    np.save(path, x)
    return x


@torch.no_grad()
def predict(model, X: torch.Tensor, device, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = [model(X[s:s + bs].to(device).float()).argmax(1).cpu() for s in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def train_one(seed, Xtr, ytr, Xva, yva, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TCResNet8(N_MFCC, len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(Xtr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss()
    best, best_state, best_ep, hist = -1.0, None, -1, []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            xb = time_shift(Xtr[idx].float(), MAX_SHIFT).to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        f1 = macro_f1(yva, predict(model, Xva, device))
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_macro_f1": f1,
                     "seconds": time.time() - t0})
        print(f"  seed {seed} epoch {ep + 1}/{args.epochs}  loss {tot / len(Xtr):.4f}  "
              f"val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, best_ep = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep, hist


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--musan-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results_tcresnet")
    ap.add_argument("--train-set", choices=["clean", "aug"], default="clean")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cache = Path(a.cache_dir)
    t_all = time.time()
    pool = NoisePool(a.musan_root)
    wtr, ytr = load_split(cache, "train")
    wva, yva = load_split(cache, "val")
    wte, yte = load_split(cache, "test")

    print("computing / loading MFCC-40 features ...", flush=True)
    Xtr = cached(cache / "mfcc40_train.npy", lambda: mfcc40_of(wtr)).astype(np.float32)
    Xva = cached(cache / "mfcc40_val.npy", lambda: mfcc40_of(wva)).astype(np.float32)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)           # per coefficient, clean train only
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float16)  # noqa: E731
    ytr_t, yva_np = torch.from_numpy(ytr), yva
    if a.train_set == "aug":
        Xtr_n = cached(cache / "mfcc40_train_aug.npy",
                       lambda: mfcc40_of(wtr, aug_transform(pool, 777, "train"))).astype(np.float32)
        Xva_n = cached(cache / "mfcc40_val_aug.npy",
                       lambda: mfcc40_of(wva, aug_transform(pool, 778, "val"))).astype(np.float32)
        Xtr_all = torch.from_numpy(norm(np.concatenate([Xtr, Xtr_n])))
        ytr_all = torch.cat([ytr_t, ytr_t])
        Xva_all = torch.from_numpy(norm(np.concatenate([Xva, Xva_n])))
        yva_all = np.concatenate([yva, yva])
        name = "B2-aug TC-ResNet8 (noise-augmented)"
    else:
        Xtr_all, ytr_all = torch.from_numpy(norm(Xtr)), ytr_t
        Xva_all, yva_all = torch.from_numpy(norm(Xva)), yva_np
        name = "B2 TC-ResNet8 (clean-trained)"
    del Xtr, Xva

    rows, seed_info = [], []
    for seed in range(a.n_seeds):
        t0 = time.time()
        print(f"[{name}] training seed {seed}", flush=True)
        model, best_f1, best_ep, hist = train_one(seed, Xtr_all, ytr_all, Xva_all, yva_all, a, device)
        train_s = time.time() - t0
        torch.save(model.state_dict(), out / "models" / f"tcresnet8_{a.train_set}_seed{seed}.pt")
        seed_info.append({"seed": seed, "best_epoch": best_ep, "val_macro_f1": best_f1,
                          "train_seconds": train_s, "history": hist})

        def record(cond, kind, snr, p):
            rows.append({"method": name, "condition": cond, "noise": kind, "snr_db": snr, "seed": seed,
                         "accuracy": accuracy(yte, p), "macro_f1": macro_f1(yte, p)})

        Xc = torch.from_numpy(norm(mfcc40_of(wte)))
        record("clean", "clean", "", predict(model, Xc, device))
        for kind in NOISE_KINDS:
            for snr in SNRS:
                rng = cond_rng(seed, kind, snr)
                X = torch.from_numpy(norm(mfcc40_of(
                    wte, lambda b: add_noise(b, kind, snr, pool, rng, "test"))))
                record(f"{kind}@{snr}dB", kind, snr, predict(model, X, device))
        print(f"[{name}] seed {seed} evaluated ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / f"per_run_{a.train_set}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"per_run_{a.train_set}.csv", index=False)
    info = {"method": name, "n_params": n_params(model), "device": str(device),
            "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "weight_decay": a.wd,
            "max_shift_frames": MAX_SHIFT, "n_mfcc": N_MFCC, "seeds": seed_info,
            "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "total_seconds": time.time() - t_all}
    (out / f"run_info_{a.train_set}.json").write_text(json.dumps(info, indent=2, default=str))
    noisy = df[df.condition != "clean"].groupby(["snr_db", "seed"])["macro_f1"].mean().groupby("snr_db").agg(["mean", "std"])
    print(f"\n{name}\nclean macro-F1: {df[df.condition == 'clean'].macro_f1.mean():.4f} "
          f"+- {df[df.condition == 'clean'].macro_f1.std():.4f}\n{noisy}\nparams: {n_params(model)}")


if __name__ == "__main__":
    main()
