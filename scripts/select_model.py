"""Validation-only model selection for the classical baseline (the test split is NOT read).

    python scripts/select_model.py --cache-dir cache --out results_select2 --segs 4 8 --pcas 80 120 160

Single-Gaussian Bayes only (mixtures were evaluated in round 1 and did not help). MFCCs are
computed once and reused for every segment count. Selection criterion: macro-F1 on the clean
validation split. Takes the cache built by run_baselines.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kwsba.bayes import GaussianBayes, Preproc  # noqa: E402
from kwsba.data import LABELS, load_split  # noqa: E402
from kwsba.features import pooled_multi  # noqa: E402
from kwsba.noise import to_float  # noqa: E402
from kwsba.run_baselines import accuracy, macro_f1  # noqa: E402


def feats(wav, segs, bs=2048):
    parts = {n: [] for n in segs}
    for s in range(0, len(wav), bs):
        d = pooled_multi(to_float(wav[s:s + bs]), segs)
        for n in segs:
            parts[n].append(d[n])
    return {n: np.concatenate(v) for n, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results_select2")
    ap.add_argument("--segs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--pcas", type=int, nargs="+", default=[80, 120, 160])
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    wtr, ytr = load_split(a.cache_dir, "train")
    wva, yva = load_split(a.cache_dir, "val")
    t = time.time()
    Ftr, Fva = feats(wtr, a.segs), feats(wva, a.segs)
    print(f"features done in {time.time() - t:.0f}s", flush=True)
    rows = []
    for n in a.segs:
        for pca in a.pcas:
            if pca > Ftr[n].shape[1]:
                continue
            prep = Preproc(pca).fit(Ftr[n])
            Ztr, Zva = prep.transform(Ftr[n]), prep.transform(Fva[n])
            for eps in a.eps:
                p = GaussianBayes(eps).fit(Ztr, ytr, len(LABELS)).predict(Zva)
                r = dict(n_seg=n, pca=pca, eps=eps, raw_dim=Ftr[n].shape[1],
                         val_macro_f1=macro_f1(yva, p), val_acc=accuracy(yva, p))
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                pd.DataFrame(rows).to_csv(out / "model_selection.csv", index=False)
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(out / "model_selection.csv", index=False)
    print("\nbest by validation macro-F1:\n", df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
