"""Stage 4 extra analysis for RQ2 (Gaussian models only, no PyTorch needed):
how do coverage and expected risk of Chow's reject rule depend on the number m of adaptation clips?

    python run_reject_m.py --musan-root <musan> --cache-dir cache --run-info results_stage4/run_info_stage4.json \
        --out results_stage4_rejm --n-seeds 5 [--check results_stage4/sensitivity_per_run.csv]

For every seed and noisy condition the test audio and the nested adaptation sets are bit-identical to the m sweep of
run_stage4 (cond_rng and the [seed, noise, snr, 98] stream), so the macro-F1 column reproduces sensitivity_per_run.csv
exactly (--check verifies this). The reject rule uses the temperature of B1 stored in run_info_stage4.json (one T per
model, fitted on the shared calibration set); adapted models reuse it, as in the paper. lambda_r = 0.2.
m = 0 is the unadapted B1.
Output: reject_m_per_run.csv with, per row, coverage, accuracy on the answered clips, expected risk with and without the
rule, and the accuracy of the 30% most confident clips (ranking quality, independent of the threshold).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from . import run_baselines
from .bayes import GaussianBayes, Preproc, fit_affine
from .data import LABELS, load_split
from .noise import NOISE_KINDS, NoisePool, add_noise
from .run_baselines import SNRS, cond_rng, features_of, macro_f1

M_GRID = (10, 30, 100, 300, 1000, 3000)


def reject_stats(logits: np.ndarray, y: np.ndarray, T: float, lam: float) -> dict:
    z = logits / T
    z = z - logsumexp(z, axis=1, keepdims=True)
    conf, pred = np.exp(z.max(axis=1)), z.argmax(axis=1)
    ok, acc = pred == y, conf >= 1.0 - lam
    top = np.argsort(-conf)[: int(round(0.3 * len(y)))]
    return {"coverage": float(acc.mean()),
            "acc_answered": float(ok[acc].mean()) if acc.any() else float("nan"),
            "risk_chow": float(((~ok & acc).sum() + lam * (~acc).sum()) / len(y)),
            "risk_no_reject": float(1.0 - ok.mean()),
            "top30_acc": float(ok[top].mean())}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--musan-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--run-info", required=True, help="run_info_stage4.json (holds the temperature of B1)")
    ap.add_argument("--out", default="results_stage4_rejm")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=80)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--lambda-r", type=float, default=0.2)
    ap.add_argument("--check", default=None, help="sensitivity_per_run.csv of run_stage4 for a built-in consistency check")
    a = ap.parse_args(argv)
    run_baselines.N_SEG = a.n_seg
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)
    T = float(json.load(open(a.run_info))["temperatures"]["B1"])
    t0 = time.time()
    pool = NoisePool(a.musan_root)
    wtr, ytr = load_split(cache, "train")
    wva, _ = load_split(cache, "val")
    wte, yte = load_split(cache, "test")
    ftr = cache / f"pooled{a.n_seg}_train.npy"
    Xtr = np.load(ftr) if ftr.exists() else features_of(wtr)
    if not ftr.exists():
        np.save(ftr, Xtr)
    prep = Preproc(a.pca_dim).fit(Xtr)
    base = GaussianBayes(a.eps).fit(prep.transform(Xtr), ytr, len(LABELS))
    del Xtr
    print(f"B1 fitted, T={T:.3f} ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for seed in range(a.n_seeds):
        for kind in NOISE_KINDS:
            for snr in SNRS:
                cond = f"{kind}@{snr}dB"
                rng = cond_rng(seed, kind, snr)
                Zt = prep.transform(features_of(wte, lambda b: add_noise(b, kind, snr, pool, rng, "test")))
                r2 = np.random.default_rng([seed, NOISE_KINDS.index(kind), snr, 98])
                idx = r2.choice(len(wva), size=min(max(M_GRID), len(wva)), replace=False)
                Z3 = prep.transform(features_of(wva[idx], lambda b: add_noise(b, kind, snr, pool, r2, "test")))
                for m in (0,) + M_GRID:
                    mdl = base if m == 0 else fit_affine(base, Z3[:m], a.kappa)
                    lg = mdl.log_joint(Zt)
                    st = reject_stats(lg, yte, T, a.lambda_r)
                    rows.append({"m": m, "method": "B1" if m == 0 else "Proposed v2", "condition": cond, "noise": kind,
                                 "snr_db": snr, "seed": seed, "macro_f1": macro_f1(yte, lg.argmax(axis=1)), **st})
                print(f"  seed {seed} {cond} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "reject_m_per_run.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "reject_m_per_run.csv", index=False)
    g = df.groupby(["snr_db", "m"])[["coverage", "risk_chow", "risk_no_reject", "top30_acc", "macro_f1"]].mean()
    print(g.round(3).to_string())
    if a.check:
        s = pd.read_csv(a.check)
        s = s[s["sens"] == "m"].rename(columns={"value": "m", "macro_f1": "mf1_ref"})
        mm = df[df.m > 0].merge(s[["m", "condition", "seed", "mf1_ref"]], on=["m", "condition", "seed"])
        err = float((mm["macro_f1"] - mm["mf1_ref"]).abs().max())
        print(f"consistency with {a.check}: {len(mm)} rows compared, max |diff| macro-F1 = {err:.2e}")
        if err > 1e-6:
            raise SystemExit("MISMATCH with the Stage 4 m sweep: do not use these results")
    print(f"wrote {out}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
