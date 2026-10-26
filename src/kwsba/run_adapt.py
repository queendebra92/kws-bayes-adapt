"""Preliminary evaluation of the proposed method: unlabeled MAP-EM adaptation + Chow reject rule.

    python run_adapt.py --musan-root <musan> --cache-dir cache --out results_adapt \
        --n-seg 4 --pca-dim 80 --eps 0.01

Protocol
  * Base model = the frozen B1 (clean-trained Gaussian Bayes, same features/PCA/eps as run_baselines),
    so the "fixed" rows reproduce B1 exactly (same noisy test audio: cond_rng(seed, kind, snr)).
  * Adaptation set: n UNLABELED clips drawn from the *validation* split (speaker-disjoint from test),
    mixed with the target noise condition (same noise pool as evaluation, different random draws).
    Labels are never read. A different adaptation sample is drawn for every seed.
  * Three adaptation variants, all unsupervised, all starting from the frozen B1:
      v1  per-class MAP-EM of means/covariances (priors fixed, relevance tau1)
      v2  one SHARED per-dimension shift+scale for all classes (2d parameters, shrinkage kappa)
      v2+EM  v2 followed by a strongly anchored MAP-EM (relevance tau2)
    v1 was run first and degraded accuracy (documented); v2 was designed in response and, like every
    hyper-parameter, is selected on the validation split only.
  * tau: given with --tau, or chosen on the VALIDATION split with --tune-tau (adapt on one half of
    the validation clips with validation noise, score on the other half; the test split is untouched).
  * Reject rule: Chow, reject iff max posterior < 1 - lambda_r. Reported for the fixed and adapted models.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .bayes import REJECT, GaussianBayes, Preproc, adapt_affine_then_em, adapt_em, fit_affine
from .data import LABELS, load_split
from .noise import NOISE_KINDS, NoisePool, add_noise
from .run_baselines import SNRS, accuracy, cond_rng, features_of, macro_f1
from . import run_baselines

LAMBDAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def reject_stats(model, Z, y, lam, proba=None):
    p = model.predict_proba(Z) if proba is None else proba
    pred = p.argmax(axis=1)
    pred[p.max(axis=1) < 1.0 - lam] = REJECT
    acc_mask = pred != REJECT
    cov = float(acc_mask.mean())
    wrong = int(((pred != y) & acc_mask).sum())
    rejected = int((~acc_mask).sum())
    risk = (wrong + lam * rejected) / len(y)            # expected 0-1 risk with rejection cost lam
    a = float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else float("nan")
    f = macro_f1(y[acc_mask], pred[acc_mask]) if acc_mask.any() else float("nan")
    return {"lambda_r": lam, "coverage": cov, "acc_accepted": a, "macro_f1_accepted": f, "risk": risk}


def adapt_rng(seed: int, kind: str, snr: int) -> np.random.Generator:
    k = NOISE_KINDS.index(kind) if kind in NOISE_KINDS else len(NOISE_KINDS)   # "clean" gets its own stream
    return np.random.default_rng([seed, k, snr, 99])


def noisy_val_features(wva, idx, kind, snr, pool, rng, part):
    sub = wva[np.sort(idx)]
    return features_of(sub, lambda b: add_noise(b, kind, snr, pool, rng, part))


def tune_adaptation(base, prep, wva, yva, pool, taus1, taus2, kappa, n_adapt, iters):
    """Validation-only selection. Adapt on half A of the validation clips, score on half B,
    validation noise pool, all 9 noise conditions. Returns {config: mean macro-F1}; includes the
    no-adaptation reference and v2 so that 'adaptation does not help' is a possible outcome."""
    rng = np.random.default_rng(2024)
    perm = rng.permutation(len(wva))
    A, B = perm[: len(perm) // 2][:n_adapt], np.sort(perm[len(perm) // 2:][:3000])
    yB = yva[B]
    scores: dict[str, list] = {}
    def add(k, v):
        scores.setdefault(k, []).append(v)
    for kind in NOISE_KINDS:
        for snr in SNRS:
            r = np.random.default_rng([7, NOISE_KINDS.index(kind), snr])
            Zu = prep.transform(noisy_val_features(wva, A, kind, snr, pool, r, "val"))
            ZB = prep.transform(features_of(wva[B], lambda b: add_noise(b, kind, snr, pool, r, "val")))
            add("fixed (no adaptation)", macro_f1(yB, base.predict(ZB)))
            add(f"v2 affine kappa={kappa:g}", macro_f1(yB, fit_affine(base, Zu, kappa).predict(ZB)))
            for t in taus1:
                add(f"v1 per-class EM tau={t:g}", macro_f1(yB, adapt_em(base, Zu, t, iters).predict(ZB)))
            for t in taus2:
                add(f"v2+EM tau={t:g}", macro_f1(
                    yB, adapt_affine_then_em(base, Zu, kappa, t, iters).predict(ZB)))
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--musan-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results_adapt")
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=80)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-adapt", type=int, nargs="+", default=[1000], help="adaptation-set sizes")
    ap.add_argument("--tau", type=float, default=64.0, help="tau1 for v1 if --tune-tau is not given")
    ap.add_argument("--tau2", type=float, default=1024.0, help="tau2 for v2+EM if --tune-tau2 is not given")
    ap.add_argument("--kappa", type=float, default=20.0, help="shrinkage of the shared affine adaptation (fixed)")
    ap.add_argument("--tune-tau", type=float, nargs="+", default=None,
                    help="grid for tau1 (v1); best on validation is used")
    ap.add_argument("--tune-tau2", type=float, nargs="+", default=None,
                    help="grid for tau2 (v2+EM); best on validation is used")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tune-n", type=int, default=1000, help="adaptation-set size used when tuning tau")
    a = ap.parse_args(argv)
    run_baselines.N_SEG = a.n_seg

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)
    t0 = time.time()
    pool = NoisePool(a.musan_root)
    wtr, ytr = load_split(cache, "train")
    wva, yva = load_split(cache, "val")
    wte, yte = load_split(cache, "test")

    ftr = cache / f"pooled{a.n_seg}_train.npy"
    if ftr.exists():
        Xtr = np.load(ftr)
    else:
        Xtr = features_of(wtr)
        np.save(ftr, Xtr)
    prep = Preproc(a.pca_dim or None).fit(Xtr)
    base = GaussianBayes(a.eps).fit(prep.transform(Xtr), ytr, len(LABELS))
    del Xtr
    info = {"n_seg": a.n_seg, "pca_dim": a.pca_dim, "eps": a.eps, "iters": a.iters,
            "n_adapt": a.n_adapt, "n_seeds": a.n_seeds}

    tau1, tau2 = a.tau, a.tau2
    if a.tune_tau or a.tune_tau2:
        sc = tune_adaptation(base, prep, wva, yva, pool, a.tune_tau or [a.tau], a.tune_tau2 or [a.tau2],
                             a.kappa, a.tune_n, a.iters)
        info["validation_selection_macro_f1"] = sc
        for k, v in sorted(sc.items(), key=lambda kv: -kv[1]):
            print(f"  validation macro-F1 {v:.4f}  {k}", flush=True)
        if a.tune_tau:
            tau1 = max((float(k.split("=")[1]) for k in sc if k.startswith("v1")),
                       key=lambda t: sc[f"v1 per-class EM tau={t:g}"])
        if a.tune_tau2:
            tau2 = max((float(k.split("=")[1]) for k in sc if k.startswith("v2+EM")),
                       key=lambda t: sc[f"v2+EM tau={t:g}"])
    info.update({"tau1": tau1, "tau2": tau2, "kappa": a.kappa})
    print(f"using tau1={tau1:g} (v1), kappa={a.kappa:g} (v2), tau2={tau2:g} (v2+EM)", flush=True)

    rows, rej = [], []
    max_n = max(a.n_adapt)
    fixed_name = "B1 Gaussian Bayes (clean-trained)"
    for seed in range(a.n_seeds):
        for kind, snr in [("clean", "")] + [(k, s) for k in NOISE_KINDS for s in SNRS]:
            cond = "clean" if kind == "clean" else f"{kind}@{snr}dB"
            if kind == "clean":
                Xt = prep.transform(features_of(wte))
            else:
                rng = cond_rng(seed, kind, snr)
                Xt = prep.transform(features_of(
                    wte, lambda b: add_noise(b, kind, snr, pool, rng, "test")))
            ar = adapt_rng(seed, kind, snr if kind != "clean" else 0)
            idx = ar.choice(len(wva), size=max_n, replace=False)
            if kind == "clean":
                Zu = prep.transform(features_of(wva[np.sort(idx)]))
            else:
                Zu = prep.transform(noisy_val_features(wva, idx, kind, snr, pool, ar, "test"))
            models = {fixed_name: base}
            for n in a.n_adapt:
                models[f"Proposed v1 per-class MAP-EM (n={n})"] = adapt_em(base, Zu[:n], tau1, a.iters)
                models[f"Proposed v2 shared affine (n={n})"] = fit_affine(base, Zu[:n], a.kappa)
                models[f"Proposed v2+EM (n={n})"] = adapt_affine_then_em(base, Zu[:n], a.kappa, tau2, a.iters)
            for name, m in models.items():
                proba = m.predict_proba(Xt)           # posteriors once; reused for every lambda_r
                p = proba.argmax(axis=1)
                rows.append({"method": name, "condition": cond, "noise": kind, "snr_db": snr, "seed": seed,
                             "accuracy": accuracy(yte, p), "macro_f1": macro_f1(yte, p)})
                for lam in LAMBDAS:
                    rej.append({"method": name, "condition": cond, "noise": kind, "snr_db": snr,
                                "seed": seed, **reject_stats(m, Xt, yte, lam, proba)})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "per_run_adapt.csv", index=False)

    df, rj = pd.DataFrame(rows), pd.DataFrame(rej)
    df.to_csv(out / "per_run_adapt.csv", index=False)
    rj.to_csv(out / "reject_curve.csv", index=False)
    info.update({"total_seconds": time.time() - t0, "python": platform.python_version(),
                 "numpy": np.__version__, "platform": platform.platform()})
    (out / "run_info_adapt.json").write_text(json.dumps(info, indent=2, default=str))

    noisy = df[df.condition != "clean"].groupby(["method", "snr_db", "seed"])["macro_f1"].mean().reset_index()
    print(noisy.groupby(["method", "snr_db"])["macro_f1"].agg(["mean", "std"]).round(4).to_string())
    print("\nclean:\n", df[df.condition == "clean"].groupby("method")["macro_f1"].agg(["mean", "std"]).round(4).to_string())
    print(f"\nwrote {out}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
