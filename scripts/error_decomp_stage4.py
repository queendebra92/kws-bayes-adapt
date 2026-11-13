"""Stage 4 error decomposition at 0 dB (paper Table 4).

    python scripts/error_decomp_stage4.py --results results_stage4 --out paper/figures

For each method it splits the 0 dB errors (mean over 5 seeds and the three noise types, no rejection) into
  keyword -> unknown, keyword -> silence, keyword -> other keyword, unknown -> keyword, unknown -> silence,
  silence -> anything,
as a share of all errors, and reports the silence recall per noise type (a "sink" check: a model that labels
every noisy clip as silence has recall 1.0 and F1 close to 0). Writes tab_errors.tex and table_error_decomp.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIL, UNK = 0, 1
METHODS = [("B1", "B1"), ("Proposed v2", r"\textbf{Proposed v2}"), ("B2", "B2"), ("B2 + AdaBN", "B2 + AdaBN"),
           ("B2-aug", "B2-aug")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y = np.load(res / "y_test.npy")
    conds = json.load(open(res / "conditions.json"))
    ci = [i for i, c in enumerate(conds) if c.endswith("@0dB")]
    seeds = sorted(res.glob("seed*.npz"))
    rows = []
    for m, _ in METHODS:
        acc: dict[str, list] = {}
        for f in seeds:
            P = np.load(f)[m + "__pred"]
            for i in ci:
                p = P[i]
                err = p != y
                e = max(err.sum(), 1)
                d = {"error rate": err.mean(),
                     "kw->unk": ((y >= 2) & (p == UNK)).sum() / e,
                     "kw->sil": ((y >= 2) & (p == SIL)).sum() / e,
                     "kw->kw": ((y >= 2) & (p >= 2) & err).sum() / e,
                     "unk->kw": ((y == UNK) & (p >= 2)).sum() / e,
                     "unk->sil": ((y == UNK) & (p == SIL)).sum() / e,
                     "sil->any": ((y == SIL) & err).sum() / e}
                nm = conds[i].split("@")[0]
                d[f"silrec_{nm}"] = (p[y == SIL] == SIL).mean()
                for k, v in d.items():
                    acc.setdefault(k, []).append(v)
        rows.append({"method": m, **{k: float(np.mean(v)) for k, v in acc.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_error_decomp.csv", index=False)
    lines = []
    for (m, lab), r in zip(METHODS, rows):
        lines.append(f"{lab} & {r['error rate']:.3f} & {100 * r['kw->unk']:.0f} & {100 * r['kw->sil']:.0f} & "
                     f"{100 * r['kw->kw']:.0f} & {100 * r['unk->kw']:.0f} & {100 * r['unk->sil']:.0f} & "
                     f"{100 * r['sil->any']:.0f} & {r['silrec_babble']:.2f} / {r['silrec_environmental']:.2f} / "
                     f"{r['silrec_white']:.2f} \\\\")
    txt = "\n".join(lines).rstrip()
    (out / "tab_errors.tex").write_text((txt[:-2].rstrip() if txt.endswith("\\\\") else txt) + "\n")  # no final row terminator; the paper adds it after \\input
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
