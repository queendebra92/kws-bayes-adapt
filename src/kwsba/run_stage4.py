"""Stage 4 experiments: per-clip predictions for every method, test-time-adaptation baselines
for B2, ablations and sensitivity analyses of the proposed shared-affine adaptation (v2).

    python run_stage4.py --musan-root <musan> --cache-dir cache \
        --b2-clean-models <dir with tcresnet8_clean_seed*.pt> \
        --b2-aug-models   <dir with tcresnet8_aug_seed*.pt>   --out results_stage4

Everything reuses the Stage 3 protocol. B1, B1-aug, B2, B2-aug and B3 reproduce the Stage 3 numbers
exactly. The adapted rows (v2, v2+EM) do NOT: Stage 3 run_adapt.py sorted the random draw of adaptation clips
before taking the first n, and the validation clips are stored class by class, so its n < 3000 results used
class-biased subsets. This script draws the m clips at random (see README, 'Known issue'):
  * seed s and condition (kind, snr) use cond_rng(s, kind, snr) for the noisy TEST audio, so the
    test clips are bit-identical to those scored by run_baselines / run_tcresnet / run_adapt.
  * the main adaptation set (m = 1000 unlabeled validation clips mixed with the target noise)
    uses adapt_rng(s, kind, snr), exactly as run_adapt.
  * every hyper-parameter is chosen on the validation split only; the test split is never used.

What is new (all unsupervised unless stated):
  * per-clip predictions + confidences saved for every method (seed{s}.npz) so that significance
    tests, confusion matrices and risk-coverage curves can be computed over test CLIPS;
  * a shared calibration set (noisy validation clips, all conditions mixed) used to fit one
    temperature per model, so the Chow reject rule can be applied to B1, B1-aug, B2, B2-aug and
    the adapted models on equal terms;
  * TTA baselines on the neural nets, adapted on the SAME m clips and then frozen (same protocol
    as v2): AdaBN (BN statistics only), Tent (entropy minimisation of BN affine parameters) and an
    AdaKWS-style variant re-implemented from the paper's description (several details are not
    specified in the paper; see adakws_loss);
  * ablations of v2 (shift only, scale only, kappa) and sensitivity to the number of adaptation
    clips m and to the class balance of the adaptation set (the labels are used ONLY to build the
    composition of that set in this one experiment).
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

try:                       # PyTorch is only needed for the neural baselines (B2, B2-aug and their TTA variants)
    import torch
    from torch import nn
except ImportError:        # pragma: no cover - Gaussian-only runs (--no-neural) work without it
    torch, nn = None, None

from . import run_baselines
from .bayes import AffineAdapted, GaussianBayes, Preproc, adapt_affine_then_em, fit_affine
from .data import LABELS, UNKNOWN, load_split
from .features import mfcc, pooled_features
from .noise import NOISE_KINDS, NoisePool, add_noise, to_float
from .run_adapt import adapt_rng
from .run_baselines import BATCH, SNRS, accuracy, aug_transform, cond_rng, features_of, macro_f1, tune_eps

N_MFCC = 40
CONDS = [("clean", 0)] + [(k, s) for k in NOISE_KINDS for s in SNRS]
KAPPA, TAU2 = 20.0, 4096.0
M_GRID = (10, 30, 100, 300, 1000, 3000)
UNK_GRID = (0.1, 0.3, 0.607, 0.8, 0.95)
KAPPA_GRID = (1.0, 5.0, 20.0, 100.0, 1000.0)
TTA_GRID = [(lr, ep) for lr in (1e-4, 1e-3, 1e-2) for ep in (1, 3)]


def cond_name(kind, snr):
    return "clean" if kind == "clean" else f"{kind}@{snr}dB"


# ------------------------------------------------------------------ features
def both_features(wav, transform=None):
    """Pooled (Gaussian) features and MFCC-40 (neural nets) for the SAME, optionally noisy, audio."""
    P, M = [], []
    for s in range(0, len(wav), BATCH):
        b = wav[s:s + BATCH]
        x = transform(b) if transform is not None else to_float(b)
        P.append(pooled_features(x, run_baselines.N_SEG))
        M.append(mfcc(x, N_MFCC))
    return np.concatenate(P), np.concatenate(M)


def cached(path: Path, fn) -> np.ndarray:
    """Same on-disk cache as run_tcresnet (float16 MFCC-40 features)."""
    if path.exists():
        return np.load(path)
    x = fn().astype(np.float16)
    np.save(path, x)
    return x


def mfcc40_of(wav, transform=None) -> np.ndarray:
    return both_features(wav, transform)[1]


# ------------------------------------------------------------------ calibration
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature T > 0 minimising the NLL of softmax(logits / T). Fitted on validation only."""
    def nll(logT):
        z = logits / np.exp(logT)
        return float(-(z[np.arange(len(y)), y] - logsumexp(z, axis=1)).mean())
    r = minimize_scalar(nll, bounds=(-4.0, 8.0), method="bounded")
    return float(np.exp(r.x))


def conf_pred(logits: np.ndarray, T: float = 1.0):
    z = logits / T
    z = z - logsumexp(z, axis=1, keepdims=True)
    return z.argmax(axis=1).astype(np.int8), np.exp(z.max(axis=1)).astype(np.float16)


# ------------------------------------------------------------------ neural nets
def logits_of(model, X, bs: int = 2048) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[s:s + bs].float()).cpu() for s in range(0, len(X), bs)]).numpy()


def load_b2(path: Path):
    from .tcresnet import TCResNet8
    m = TCResNet8(N_MFCC, len(LABELS))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def recompute_bn_stats(model, X, bs: int = 128) -> None:
    """AdaBN / TBN: replace the BN running statistics by those of the unlabeled adaptation clips."""
    for m in bn_layers(model):
        m.reset_running_stats()
        m.momentum = None            # cumulative average over the adaptation batches
    model.train()
    with torch.no_grad():
        for s in range(0, len(X), bs):
            model(X[s:s + bs].float())
    model.eval()


def spec_mask(x, gen, n_t=2, max_t=20, n_f=2, max_f=5) -> torch.Tensor:
    """Two time masks (<= max_t frames) and two frequency masks (<= max_f coefficients); zero = train mean."""
    x = x.clone()
    B, C, T = x.shape
    for i in range(B):
        for _ in range(n_t):
            w = int(torch.randint(0, max_t + 1, (1,), generator=gen))
            s = int(torch.randint(0, max(T - w, 1), (1,), generator=gen))
            x[i, :, s:s + w] = 0
        for _ in range(n_f):
            w = int(torch.randint(0, max_f + 1, (1,), generator=gen))
            s = int(torch.randint(0, max(C - w, 1), (1,), generator=gen))
            x[i, s:s + w, :] = 0
    return x


def entropy(logits):
    lp = torch.log_softmax(logits, dim=1)
    return -(lp.exp() * lp).sum(dim=1)


def adakws_loss(model, xb, gen, tau_ent=0.4, sigma=0.5, tau_pkc=0.05):
    """AdaKWS-style loss (Xiao et al., Interspeech 2025), re-implemented from the paper's text.

    The paper gives: entropy L_ent; a prediction-keeping-consistency score L_pkc = p(x)_c - p(x')_c
    with x' a SpecAugment-masked copy and c the predicted class; thresholds tau_ent = 0.4,
    tau_pkc = 0.05; weight alpha = 1/exp(L_ent - sigma) + 1/exp(-L_pkc), sigma = 0.5; loss =
    alpha * L_ent on the selected samples. Details the paper does not fix, and our choices:
      * which parameters: BN affine parameters (as Tent);
      * direction of the PKC selection: the text says "stable" samples are kept although its
        Eq. 4 reads L_pkc > tau_pkc; we follow the text (keep L_pkc <= tau_pkc);
      * a sample must pass BOTH the entropy and the PKC filter; the loss is the mean over kept samples;
      * alpha is treated as a constant weight (no gradient).
    """
    logits = model(xb)
    ent = entropy(logits)
    with torch.no_grad():
        p = torch.softmax(logits, dim=1)
        p2 = torch.softmax(model(spec_mask(xb, gen)), dim=1)
        c = p.argmax(dim=1)
        pkc = p.gather(1, c[:, None])[:, 0] - p2.gather(1, c[:, None])[:, 0]
        alpha = torch.exp(sigma - ent) + torch.exp(pkc)
        keep = (ent < tau_ent) & (pkc <= tau_pkc)
    if not bool(keep.any()):
        return None
    return (alpha[keep] * ent[keep]).mean()


def tta_adapt(model, Xa, method: str, lr: float, epochs: int, seed: int, bs: int = 128):
    """Offline test-time adaptation on the m unlabeled adaptation clips, then frozen.

    adabn : BN running statistics only.
    tent  : AdaBN + entropy minimisation of the BN affine parameters (Wang et al., ICLR 2021).
    adakws: AdaBN + AdaKWS-style filtered/weighted entropy minimisation of the BN affine parameters.
    """
    m = copy.deepcopy(model).eval()
    recompute_bn_stats(m, Xa, bs)
    if method == "adabn":
        return m
    params = []
    for layer in bn_layers(m):
        params += [layer.weight, layer.bias]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xa), generator=gen)
        for s in range(0, len(perm), bs):
            xb = Xa[perm[s:s + bs]].float()
            loss = entropy(m(xb)).mean() if method == "tent" else adakws_loss(m, xb, gen)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in m.parameters():
        p.requires_grad_(False)
    recompute_bn_stats(m, Xa, bs)       # statistics consistent with the updated affine parameters
    return m


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--musan-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results_stage4")
    ap.add_argument("--b2-clean-models", default=None, help="folder with tcresnet8_clean_seed*.pt")
    ap.add_argument("--b2-aug-models", default=None, help="folder with tcresnet8_aug_seed*.pt")
    ap.add_argument("--no-neural", action="store_true", help="Gaussian models only (no PyTorch, no B2)")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=80)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--tau2", type=float, default=TAU2)
    ap.add_argument("--n-adapt", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-tta", action="store_true", help="skip the neural TTA baselines")
    a = ap.parse_args(argv)
    neural = not a.no_neural
    if neural and (torch is None or not a.b2_clean_models or not a.b2_aug_models):
        raise SystemExit("need PyTorch and --b2-clean-models / --b2-aug-models (or pass --no-neural)")
    if a.no_neural:
        a.skip_tta = True
    run_baselines.N_SEG = a.n_seg

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)
    t0 = time.time()
    pool = NoisePool(a.musan_root)
    wtr, ytr = load_split(cache, "train")
    wva, yva = load_split(cache, "val")
    wte, yte = load_split(cache, "test")
    np.save(out / "y_test.npy", yte)
    (out / "conditions.json").write_text(json.dumps([cond_name(k, s) for k, s in CONDS]))

    # ---- Gaussian models (identical to run_baselines / run_adapt) ----------------------------
    print("fitting B1 / B1-aug ...", flush=True)
    ftr = cache / f"pooled{a.n_seg}_train.npy"
    Xtr = np.load(ftr) if ftr.exists() else features_of(wtr)
    if not ftr.exists():
        np.save(ftr, Xtr)
    prep = Preproc(a.pca_dim or None).fit(Xtr)
    Ztr = prep.transform(Xtr)
    del Xtr
    Zva = prep.transform(features_of(wva))
    base = GaussianBayes(a.eps).fit(Ztr, ytr, len(LABELS))
    Zn = prep.transform(features_of(wtr, aug_transform(pool, 777, "train")))
    Zvn = prep.transform(features_of(wva, aug_transform(pool, 778, "val")))
    eps2, _ = tune_eps(np.concatenate([Ztr, Zn]), np.concatenate([ytr, ytr]),
                       np.concatenate([Zva, Zvn]), np.concatenate([yva, yva]))
    base_aug = GaussianBayes(eps2).fit(np.concatenate([Ztr, Zn]), np.concatenate([ytr, ytr]), len(LABELS))
    del Ztr, Zn, Zva, Zvn
    majority = int(np.bincount(ytr, minlength=len(LABELS)).argmax())

    # ---- neural inputs: normalisation exactly as run_tcresnet ----------------------------------
    norm = None
    if neural:
        Xtr40 = cached(cache / "mfcc40_train.npy", lambda: mfcc40_of(wtr)).astype(np.float32)
        mu, sd = Xtr40.mean(axis=(0, 2), keepdims=True), Xtr40.std(axis=(0, 2), keepdims=True) + 1e-6
        del Xtr40
        norm = lambda X: torch.from_numpy(((X - mu) / sd).astype(np.float16))  # noqa: E731

    # ---- shared calibration set: noisy validation clips, all conditions mixed -------------------
    print("building the shared calibration set ...", flush=True)
    crng = np.random.default_rng(4242)
    which = crng.integers(len(CONDS), size=len(wva))
    Pc_list, Mc_list, order = [], [], []
    for ci, (kind, snr) in enumerate(CONDS):
        idx = np.where(which == ci)[0]
        if len(idx) == 0:
            continue
        tr = None if kind == "clean" else (lambda b, k=kind, s=snr: add_noise(b, k, s, pool, crng, "val"))
        P, M = both_features(wva[idx], tr)
        Pc_list.append(P); Mc_list.append(M); order.append(idx)
    order = np.concatenate(order)
    inv = np.argsort(order)
    Pc = np.concatenate(Pc_list)[inv]
    Mc = norm(np.concatenate(Mc_list)[inv]) if neural else None
    Zc = prep.transform(Pc)
    yc = yva
    T = {"B1": fit_temperature(base.log_joint(Zc), yc), "B1-aug": fit_temperature(base_aug.log_joint(Zc), yc)}
    models_b2 = {}
    for s in range(a.n_seeds if neural else 0):
        models_b2[("B2", s)] = load_b2(Path(a.b2_clean_models) / f"tcresnet8_clean_seed{s}.pt")
        models_b2[("B2-aug", s)] = load_b2(Path(a.b2_aug_models) / f"tcresnet8_aug_seed{s}.pt")
        for k in ("B2", "B2-aug"):
            T[f"{k}|{s}"] = fit_temperature(logits_of(models_b2[(k, s)], Mc), yc)
    print("temperatures:", {k: round(v, 3) for k, v in T.items()}, flush=True)

    # ---- TTA hyper-parameters: validation only (seed-0 networks) ---------------------------------
    tta_cfg = {}
    if not a.skip_tta:
        print("tuning the TTA baselines on validation ...", flush=True)
        perm = np.random.default_rng(2024).permutation(len(wva))
        A, B = np.sort(perm[: len(perm) // 2][: a.n_adapt]), np.sort(perm[len(perm) // 2:][:2000])
        vcache = []
        for kind, snr in CONDS[1:]:
            r = np.random.default_rng([7, NOISE_KINDS.index(kind), snr])
            _, MA = both_features(wva[A], lambda b: add_noise(b, kind, snr, pool, r, "val"))
            _, MB = both_features(wva[B], lambda b: add_noise(b, kind, snr, pool, r, "val"))
            vcache.append((norm(MA), norm(MB)))
        yB = yva[B]
        for kname in ("B2", "B2-aug"):
            m0 = models_b2[(kname, 0)]
            ref = np.mean([macro_f1(yB, logits_of(m0, MB).argmax(1)) for _, MB in vcache])
            print(f"  [{kname}] no adaptation: val macro-F1 {ref:.4f}", flush=True)
            for meth in ("tent", "adakws"):
                best = None
                for lr, ep in TTA_GRID:
                    f = np.mean([macro_f1(yB, logits_of(tta_adapt(m0, MA, meth, lr, ep, 0), MB).argmax(1))
                                 for MA, MB in vcache])
                    print(f"  [{kname}] {meth} lr={lr:g} epochs={ep}: val macro-F1 {f:.4f}", flush=True)
                    if best is None or f > best[0]:
                        best = (f, lr, ep)
                tta_cfg[f"{kname}|{meth}"] = {"lr": best[1], "epochs": best[2], "val_macro_f1": best[0],
                                              "val_macro_f1_no_adaptation": ref}
        (out / "tta_validation.json").write_text(json.dumps(tta_cfg, indent=2))

    # ---- evaluation -----------------------------------------------------------------------------
    abl_rows, sens_rows = [], []
    for seed in range(a.n_seeds):
        f_out = out / f"seed{seed}.npz"
        if f_out.exists():
            print(f"seed {seed}: already done, skipping", flush=True)
            continue
        store: dict[str, list] = {}

        def put(method, cond_i, logits, Tkey):
            pred, conf_raw = conf_pred(logits, 1.0)
            _, conf_cal = conf_pred(logits, T[Tkey])
            for suffix, arr in (("pred", pred), ("conf", conf_raw), ("confcal", conf_cal)):
                store.setdefault(f"{method}__{suffix}", [None] * len(CONDS))[cond_i] = arr

        for ci, (kind, snr) in enumerate(CONDS):
            cond = cond_name(kind, snr)
            if kind == "clean":
                Pt, Mt = both_features(wte)
            else:
                rng = cond_rng(seed, kind, snr)
                Pt, Mt = both_features(wte, lambda b: add_noise(b, kind, snr, pool, rng, "test"))
            Zt, Xt = prep.transform(Pt), (norm(Mt) if neural else None)

            # main adaptation set: m = 1000 clips drawn at random (all of them are used, so order is irrelevant)
            ar = adapt_rng(seed, kind, snr if kind != "clean" else 0)
            idx = ar.choice(len(wva), size=a.n_adapt, replace=False)
            sub = wva[np.sort(idx)]
            Pu, Mu = both_features(sub) if kind == "clean" else both_features(
                sub, lambda b: add_noise(b, kind, snr, pool, ar, "test"))
            Zu, Xu = prep.transform(Pu), (norm(Mu) if neural else None)

            put("B3 majority class", ci, np.eye(len(LABELS))[np.full(len(yte), majority)] * 10.0, "B1")
            put("B1", ci, base.log_joint(Zt), "B1")
            put("B1-aug", ci, base_aug.log_joint(Zt), "B1-aug")
            v2 = fit_affine(base, Zu, a.kappa)
            put("Proposed v2", ci, v2.log_joint(Zt), "B1")
            v2em = adapt_affine_then_em(base, Zu, a.kappa, a.tau2, a.iters)
            put("v2+EM", ci, v2em.log_joint(Zt), "B1")
            put("B1-aug + v2", ci, fit_affine(base_aug, Zu, a.kappa).log_joint(Zt), "B1-aug")
            for kname in (("B2", "B2-aug") if neural else ()):
                m = models_b2[(kname, seed)]
                put(kname, ci, logits_of(m, Xt), f"{kname}|{seed}")
                if a.skip_tta:
                    continue
                put(f"{kname} + AdaBN", ci, logits_of(tta_adapt(m, Xu, "adabn", 0, 0, seed), Xt), f"{kname}|{seed}")
                for meth, label in (("tent", "Tent"), ("adakws", "AdaKWS-style")):
                    cfg = tta_cfg[f"{kname}|{meth}"]
                    put(f"{kname} + {label}", ci,
                        logits_of(tta_adapt(m, Xu, meth, cfg["lr"], cfg["epochs"], seed), Xt), f"{kname}|{seed}")

            # ---- ablations of v2 (m = 1000) -------------------------------------------------
            aff = fit_affine(base, Zu, a.kappa)
            variants = {"full (shift + scale)": aff,
                        "shift only": AffineAdapted(base, aff.m, np.ones_like(aff.s)),
                        "scale only": AffineAdapted(base, np.zeros_like(aff.m), aff.s),
                        "no adaptation": AffineAdapted(base, np.zeros_like(aff.m), np.ones_like(aff.s))}
            for kap in KAPPA_GRID:
                variants[f"kappa={kap:g}"] = fit_affine(base, Zu, kap)
            for name, mdl in variants.items():
                p = mdl.predict(Zt)
                abl_rows.append({"variant": name, "condition": cond, "noise": kind, "snr_db": snr, "seed": seed,
                                 "accuracy": accuracy(yte, p), "macro_f1": macro_f1(yte, p)})

            # ---- sensitivity to the number of adaptation clips m (nested sets) ---------------
            if kind != "clean":
                r2 = np.random.default_rng([seed, NOISE_KINDS.index(kind), snr, 98])
                idx3 = r2.choice(len(wva), size=min(max(M_GRID), len(wva)), replace=False)
                P3, _ = both_features(wva[idx3], lambda b: add_noise(b, kind, snr, pool, r2, "test"))
                Z3 = prep.transform(P3)
                for mm in M_GRID:
                    p = fit_affine(base, Z3[:mm], a.kappa).predict(Zt)
                    sens_rows.append({"sens": "m", "value": mm, "condition": cond, "noise": kind, "snr_db": snr,
                                      "seed": seed, "accuracy": accuracy(yte, p), "macro_f1": macro_f1(yte, p)})
                # ---- sensitivity to the class balance of the adaptation set (0 dB only) -------
                if snr == 0:
                    r3 = np.random.default_rng([seed, NOISE_KINDS.index(kind), snr, 97])
                    for u in UNK_GRID:
                        n_unk = int(round(u * a.n_adapt))
                        unk_pool, others = np.where(yva == UNKNOWN)[0], np.where(yva != UNKNOWN)[0]
                        idxu = r3.choice(unk_pool, size=n_unk, replace=n_unk > len(unk_pool))   # replace only on tiny test data
                        rest = r3.choice(others, size=a.n_adapt - n_unk, replace=a.n_adapt - n_unk > len(others))
                        ids = np.sort(np.concatenate([idxu, rest]))
                        Pq, _ = both_features(wva[ids], lambda b: add_noise(b, kind, snr, pool, r3, "test"))
                        p = fit_affine(base, prep.transform(Pq), a.kappa).predict(Zt)
                        sens_rows.append({"sens": "unknown_fraction", "value": u, "condition": cond, "noise": kind,
                                          "snr_db": snr, "seed": seed, "accuracy": accuracy(yte, p),
                                          "macro_f1": macro_f1(yte, p)})
            print(f"  seed {seed} {cond} done ({time.time() - t0:.0f}s)", flush=True)

        np.savez_compressed(f_out, **{k: np.stack(v) for k, v in store.items()})
        pd.DataFrame(abl_rows).to_csv(out / "ablation_per_run.csv", index=False)
        pd.DataFrame(sens_rows).to_csv(out / "sensitivity_per_run.csv", index=False)
        print(f"seed {seed} saved ({time.time() - t0:.0f}s)", flush=True)

    info = {"n_seeds": a.n_seeds, "kappa": a.kappa, "tau2": a.tau2, "n_adapt": a.n_adapt, "eps": a.eps,
            "eps_aug": eps2, "temperatures": T, "tta": tta_cfg, "m_grid": list(M_GRID),
            "unknown_fraction_grid": list(UNK_GRID), "kappa_grid": list(KAPPA_GRID),
            "python": platform.python_version(), "numpy": np.__version__, "torch": getattr(torch, "__version__", None),
            "platform": platform.platform(), "total_seconds": time.time() - t0}
    (out / "run_info_stage4.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"wrote {out}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
