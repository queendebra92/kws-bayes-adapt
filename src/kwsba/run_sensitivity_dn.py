"""Stage 4 extra sensitivity analysis (Gaussian models only, no PyTorch needed):
how do B1 and the proposed v2 depend on the PCA dimension d and on the TRAINING-set size n?

    python run_sensitivity_dn.py --musan-root <musan> --cache-dir cache --out results_stage4_dn --n-seeds 3

For every seed and noisy condition the test audio is bit-identical to run_stage4 (cond_rng) and the m = 1000
adaptation clips are the same draw (adapt_rng). The settings d = 80 and n = 100 % therefore reproduce the
Stage 4 rows of B1 and v2 exactly (a built-in check, printed at the end). Everything else about the models is
unchanged (4 segments, epsilon = 0.01, kappa = 20); nothing is tuned on test data.

  d sweep : PCA dimension in {20, 40, 80, 120} with the full training set;
  n sweep : random fraction of the training clips in {1 %, 3 %, 10 %, 30 %, 100 %} with d = 80
            (the standardiser, PCA and Gaussians are all refitted on the subset; subset drawn per seed).
Output: sens_dn_per_run.csv (columns: sens, value, method, condition, noise, snr_db, seed, accuracy, macro_f1).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import run_baselines
from .bayes import GaussianBayes, Preproc, fit_affine
from .data import LABELS, load_split
from .noise import NOISE_KINDS, NoisePool, add_noise
from .run_adapt import adapt_rng
from .run_baselines import SNRS, accuracy, cond_rng, features_of, macro_f1

D_GRID = (20, 40, 80, 120)
N_FRAC = (0.01, 0.03, 0.1, 0.3, 1.0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--musan-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results_stage4_dn")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--n-adapt", type=int, default=1000)
    ap.add_argument("--dims", type=int, nargs="+", default=list(D_GRID))
    ap.add_argument("--fractions", type=float, nargs="+", default=list(N_FRAC))
    ap.add_argument("--main-dim", type=int, default=80)
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
    Xtr = np.load(ftr) if ftr.exists() else features_of(wtr)
    if not ftr.exists():
        np.save(ftr, Xtr)
    nl = len(LABELS)

    # ---- models of every setting (fitted once; they do not depend on the condition) -------------------
    settings = []          # (sens, value, seed or None, preproc, model)
    for d in a.dims:
        prep = Preproc(d).fit(Xtr)
        settings.append(("d", d, None, prep, GaussianBayes(a.eps).fit(prep.transform(Xtr), ytr, nl)))
    for seed in range(a.n_seeds):
        for fr in a.fractions:
            idx = np.arange(len(ytr)) if fr >= 1.0 else np.sort(
                np.random.default_rng([seed, 55, int(fr * 1000)]).choice(len(ytr), size=max(int(fr * len(ytr)), nl * 5),
                                                                          replace=False))
            prep = Preproc(a.main_dim).fit(Xtr[idx])
            settings.append(("n_train", fr, seed, prep, GaussianBayes(a.eps).fit(prep.transform(Xtr[idx]), ytr[idx], nl)))
    print(f"fitted {len(settings)} model settings ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for seed in range(a.n_seeds):
        for kind in NOISE_KINDS:
            for snr in SNRS:
                cond = f"{kind}@{snr}dB"
                rng = cond_rng(seed, kind, snr)
                Pt = features_of(wte, lambda b: add_noise(b, kind, snr, pool, rng, "test"))
                ar = adapt_rng(seed, kind, snr)
                idx = ar.choice(len(wva), size=a.n_adapt, replace=False)
                sub = wva[np.sort(idx)]
                Pu = features_of(sub, lambda b: add_noise(b, kind, snr, pool, ar, "test"))
                for sens, val, s_seed, prep, base in settings:
                    if s_seed is not None and s_seed != seed:
                        continue
                    Zt, Zu = prep.transform(Pt), prep.transform(Pu)
                    for meth, mdl in (("B1", base), ("Proposed v2", fit_affine(base, Zu, a.kappa))):
                        p = mdl.predict(Zt)
                        rows.append({"sens": sens, "value": val, "method": meth, "condition": cond, "noise": kind,
                                     "snr_db": snr, "seed": seed, "accuracy": accuracy(yte, p),
                                     "macro_f1": macro_f1(yte, p)})
                print(f"  seed {seed} {cond} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "sens_dn_per_run.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sens_dn_per_run.csv", index=False)
    g = df.groupby(["sens", "value", "method", "snr_db", "seed"])["macro_f1"].mean().reset_index()
    print(g.groupby(["sens", "value", "method", "snr_db"])["macro_f1"].mean().unstack().round(3).to_string())
    print(f"wrote {out}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
