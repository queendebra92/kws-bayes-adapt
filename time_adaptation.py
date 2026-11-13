"""Time the test-time adaptation step (Table 2, "Train time" column for the adaptation rows).

Uses the SAME cache, features, frozen B1, adaptation sets and final hyper-parameters as run_adapt.py,
and measures only the work done at deployment time on m unlabeled clips:

  features   MFCC pooling + PCA of the m noisy clips
  v2         fit_affine           (the proposed method)
  v1         adapt_em             (per-class MAP-EM)
  v2+EM      adapt_affine_then_em

Each (seed, noise condition) is one sample (45 samples with the defaults); every call is repeated
--repeats times and the median is kept, so a background process cannot inflate the number.

    python time_adaptation.py --musan-root <musan> --cache-dir cache --out results/timing
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kwsba import run_baselines  # noqa: E402
from kwsba.bayes import GaussianBayes, Preproc, adapt_affine_then_em, adapt_em, fit_affine  # noqa: E402
from kwsba.data import LABELS, load_split  # noqa: E402
from kwsba.noise import NOISE_KINDS, NoisePool  # noqa: E402
from kwsba.run_adapt import adapt_rng, noisy_val_features  # noqa: E402
from kwsba.run_baselines import SNRS, features_of  # noqa: E402


def timed(fn, repeats):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--musan-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results/timing")
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--pca-dim", type=int, default=80)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--m", type=int, default=1000, help="number of unlabeled adaptation clips")
    ap.add_argument("--tau1", type=float, default=1024.0)
    ap.add_argument("--tau2", type=float, default=4096.0)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)
    run_baselines.N_SEG = a.n_seg

    cache = Path(a.cache_dir)
    pool = NoisePool(a.musan_root)
    wtr, ytr = load_split(cache, "train")
    wva, _ = load_split(cache, "val")
    ftr = cache / f"pooled{a.n_seg}_train.npy"
    Xtr = np.load(ftr) if ftr.exists() else features_of(wtr)
    prep = Preproc(a.pca_dim or None).fit(Xtr)
    base = GaussianBayes(a.eps).fit(prep.transform(Xtr), ytr, len(LABELS))
    del Xtr
    m = min(a.m, len(wva))

    names = ("features", "v2", "v1", "v2+EM")
    rows = {k: [] for k in names}
    base.predict_proba(prep.transform(features_of(wva[:8])))          # warm-up (BLAS, caches)
    for seed in range(a.n_seeds):
        for kind in NOISE_KINDS:
            for snr in SNRS:
                ar = adapt_rng(seed, kind, snr)
                idx = ar.choice(len(wva), size=m, replace=False)
                t = time.perf_counter()
                Zu = prep.transform(noisy_val_features(wva, idx, kind, snr, pool, ar, "test"))
                rows["features"].append(time.perf_counter() - t)
                rows["v2"].append(timed(lambda: fit_affine(base, Zu, a.kappa), a.repeats))
                rows["v1"].append(timed(lambda: adapt_em(base, Zu, a.tau1, a.iters), a.repeats))
                rows["v2+EM"].append(timed(lambda: adapt_affine_then_em(base, Zu, a.kappa, a.tau2, a.iters), a.repeats))
        print(f"seed {seed} done", flush=True)

    summary = {k: {"mean_s": float(np.mean(v)), "std_s": float(np.std(v, ddof=1)),
                   "median_s": float(np.median(v)), "n": len(v)} for k, v in rows.items()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {"m": m, "tau1": a.tau1, "tau2": a.tau2, "kappa": a.kappa, "iters": a.iters, "repeats": a.repeats,
            "n_seeds": a.n_seeds, "summary": summary, "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor()}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"\nadaptation timing, m = {m} unlabeled clips (seconds, mean ± std over {len(rows['v2'])} runs)")
    for k, s in summary.items():
        print(f"  {k:9s} {s['mean_s']:.4f} ± {s['std_s']:.4f}   (median {s['median_s']:.4f})")
    print(f"\nwrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
