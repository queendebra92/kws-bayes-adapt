"""Reproduce the Stage 3 baseline table (Gaussian Bayes + majority-class floor).

    python -m kwsba.run_baselines --sc-root <speech_commands_v0.02> \
        --musan-root <musan> --out results

Protocol (all fixed before looking at any test number):
  * train on the official training split; tune the one hyper-parameter (eps)
    on the validation split only; the test split is used once for reporting.
  * test conditions: clean, and {babble, environmental, white} x {20, 10, 0} dB
    mixed with MUSAN files from the held-out *test* noise pool.
  * seed s controls the noise realisation (files, offsets, babble mixture,
    white-noise samples). The Bayes models are deterministic given the data, so
    their reported std reflects test-noise variability only, not training
    variability. Clean has no randomness (std = 0).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import f1_score

from .bayes import GaussianBayes, Preproc
from .data import LABELS, build_cache, load_split
from .features import pooled_features
from .noise import NOISE_KINDS, NoisePool, add_noise, add_random_noise, to_float

SNRS = (20, 10, 0)
EPS_GRID = (1e-3, 1e-2, 1e-1, 1.0)
BATCH = 2048


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, labels=list(range(len(LABELS))), average="macro", zero_division=0))


def accuracy(y, p) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


N_SEG = 4  # set from --n-seg in main()


def features_of(wav, transform=None) -> np.ndarray:
    """Pooled features for a (memmap) int16 array, in batches."""
    out = []
    for s in range(0, len(wav), BATCH):
        b = wav[s:s + BATCH]
        x = transform(b) if transform is not None else to_float(b)
        out.append(pooled_features(x, N_SEG))
    return np.concatenate(out)


def tune_eps(Xtr, ytr, Xva, yva):
    best = None
    for eps in EPS_GRID:
        m = GaussianBayes(eps).fit(Xtr, ytr, len(LABELS))
        f = macro_f1(yva, m.predict(Xva))
        if best is None or f > best[1]:
            best = (eps, f)
    return best


def aug_transform(pool: NoisePool, seed: int, part: str):
    """Noisy-copy transform with ONE generator shared by all batches.

    (An earlier version re-created the generator inside the per-batch call, so every
    batch of 2048 clips received the identical noise draws. Fixed; see tests.)
    """
    rng = np.random.default_rng(seed)
    return lambda b: add_random_noise(b, pool, rng, part)


def cond_rng(seed: int, kind: str, snr: int) -> np.random.Generator:
    return np.random.default_rng([seed, NOISE_KINDS.index(kind), snr])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sc-root", required=True, help="speech_commands_v0.02 folder")
    ap.add_argument("--musan-root", required=True, help="musan folder (with noise/ and speech/)")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-seg", type=int, default=4, help="time segments for pooling (validation-selected)")
    ap.add_argument("--pca-dim", type=int, default=80, help="PCA dimension, 0 = none (validation-selected)")
    ap.add_argument("--eps", type=float, default=None,
                    help="freeze the clean-trained B1 regulariser (default: pick from the grid on validation)")
    a = ap.parse_args(argv)
    global N_SEG
    N_SEG = a.n_seg

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = build_cache(a.sc_root, a.cache_dir)
    pool = NoisePool(a.musan_root)
    wtr, ytr = load_split(a.cache_dir, "train")
    wva, yva = load_split(a.cache_dir, "val")
    wte, yte = load_split(a.cache_dir, "test")
    info = {"data": meta, "noise_pool_files": pool.counts(), "snrs": list(SNRS),
            "n_seeds": a.n_seeds, "eps_grid": list(EPS_GRID),
            "n_seg": a.n_seg, "pca_dim": a.pca_dim}

    # ---- training features -------------------------------------------------
    Xtr = features_of(wtr)
    Xva = features_of(wva)
    Xtr_n = features_of(wtr, aug_transform(pool, 777, "train"))
    Xva_n = features_of(wva, aug_transform(pool, 778, "val"))
    info["feature_dim_raw"] = int(Xtr.shape[1])
    prep = Preproc(a.pca_dim or None).fit(Xtr)  # fitted on clean training data only
    Xtr, Xva, Xtr_n, Xva_n = (prep.transform(v) for v in (Xtr, Xva, Xtr_n, Xva_n))
    info["feature_dim"] = int(Xtr.shape[1])

    # ---- models ------------------------------------------------------------
    models, fit_time = {}, {}
    if a.eps is not None:
        eps = a.eps
        f = macro_f1(yva, GaussianBayes(eps).fit(Xtr, ytr, len(LABELS)).predict(Xva))
    else:
        eps, f = tune_eps(Xtr, ytr, Xva, yva)
    t = time.time()
    models["B1 Gaussian Bayes (clean-trained)"] = GaussianBayes(eps).fit(Xtr, ytr, len(LABELS))
    fit_time["B1 Gaussian Bayes (clean-trained)"] = time.time() - t
    info["eps_clean"] = {"eps": eps, "val_macro_f1": f}

    Xa, ya = np.concatenate([Xtr, Xtr_n]), np.concatenate([ytr, ytr])
    Xv, yv = np.concatenate([Xva, Xva_n]), np.concatenate([yva, yva])
    eps2, f2 = tune_eps(Xa, ya, Xv, yv)
    t = time.time()
    models["B1-aug Gaussian Bayes (noise-augmented)"] = GaussianBayes(eps2).fit(Xa, ya, len(LABELS))
    fit_time["B1-aug Gaussian Bayes (noise-augmented)"] = time.time() - t
    info["eps_aug"] = {"eps": eps2, "val_macro_f1": f2}

    majority = int(np.bincount(ytr, minlength=len(LABELS)).argmax())
    info["majority_class"] = LABELS[majority]

    def predict_all(X):
        preds = {"B3 majority class": np.full(len(X), majority)}
        for name, m in models.items():
            preds[name] = m.predict(X)
        return preds

    # ---- evaluation --------------------------------------------------------
    rows = []

    def record(cond, kind, snr, seed, preds):
        for name, p in preds.items():
            rows.append({"method": name, "condition": cond, "noise": kind, "snr_db": snr,
                         "seed": seed, "accuracy": accuracy(yte, p), "macro_f1": macro_f1(yte, p)})

    record("clean", "clean", "", 0, predict_all(prep.transform(features_of(wte))))
    for seed in range(a.n_seeds):
        for kind in NOISE_KINDS:
            for snr in SNRS:
                rng = cond_rng(seed, kind, snr)
                X = prep.transform(features_of(
                    wte, lambda b: add_noise(b, kind, snr, pool, rng, "test")))
                record(f"{kind}@{snr}dB", kind, snr, seed, predict_all(X))
        print(f"seed {seed} done  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run.csv", index=False)

    # ---- aggregate: per SNR, averaged over the three noise types -----------
    clean = df[df.condition == "clean"].groupby("method")[["accuracy", "macro_f1"]].first()
    noisy = df[df.condition != "clean"]
    per_seed = noisy.groupby(["method", "snr_db", "seed"])[["accuracy", "macro_f1"]].mean().reset_index()
    agg = per_seed.groupby(["method", "snr_db"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "table2_by_snr.csv", index=False)
    df.groupby(["method", "condition"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).to_csv(
        out / "table2_by_condition.csv")

    order = ["B3 majority class", "B1 Gaussian Bayes (clean-trained)",
             "B1-aug Gaussian Bayes (noise-augmented)"]
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & Clean & 20\,dB & 10\,dB & 0\,dB \\", r"\midrule"]
    for name in order:
        cells = [f"{clean.loc[name, 'macro_f1']:.3f} $\\pm$ 0.000"]
        for snr in SNRS:
            r = agg[(agg.method == name) & (agg.snr_db == snr)].iloc[0]
            cells.append(f"{r.macro_f1_mean:.3f} $\\pm$ {r.macro_f1_std:.3f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table2_macro_f1.tex").write_text("\n".join(lines) + "\n")

    info.update({
        "fit_seconds": fit_time,
        "total_seconds": time.time() - t0,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "platform": platform.platform(),
        "processor": platform.processor(),
    })
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(agg.to_string(index=False))
    print(f"\nclean macro-F1:\n{clean['macro_f1'].to_string()}")
    print(f"\nwrote {out}/ (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
