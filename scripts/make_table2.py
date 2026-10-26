"""Combine per-run CSVs into the final baseline table (macro-F1 and accuracy).

    python scripts/make_table2.py --out results_final results_5seeds_final2/per_run.csv \
        results_tcresnet/per_run_clean.csv results_tcresnet/per_run_aug.csv

Per SNR: for each seed average over the three noise types, then report mean +- std over seeds.
Clean: mean +- std over seeds (std is 0 for methods with a single deterministic clean score).
"""
import argparse
from pathlib import Path

import pandas as pd

ORDER = ["B3 majority class", "B1 Gaussian Bayes (clean-trained)", "B1-aug Gaussian Bayes (noise-augmented)",
         "B2 TC-ResNet8 (clean-trained)", "B2-aug TC-ResNet8 (noise-augmented)"]


def ms(x):
    x = pd.Series(x)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)


def table(df, metric):
    rows = []
    for m in [m for m in ORDER if m in set(df.method)]:
        d = df[df.method == m]
        cells = {"method": m}
        cells["clean"] = ms(d[d.condition == "clean"][metric])
        for snr in (20, 10, 0):
            per_seed = d[(d.condition != "clean") & (d.snr_db == snr)].groupby("seed")[metric].mean()
            cells[f"{snr} dB"] = ms(per_seed)
        rows.append(cells)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="results_final")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in a.csvs], ignore_index=True)
    df.to_csv(out / "per_run_all.csv", index=False)
    for metric, tag in (("macro_f1", "macro_f1"), ("accuracy", "accuracy")):
        rows = table(df, metric)
        flat = [{"method": r["method"], **{k: f"{v[0]:.3f} +- {v[1]:.3f}" for k, v in r.items() if k != "method"}}
                for r in rows]
        pd.DataFrame(flat).to_csv(out / f"table2_{tag}.csv", index=False)
        tex = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
               r"Method & Clean & 20\,dB & 10\,dB & 0\,dB \\", r"\midrule"]
        for r in rows:
            tex.append(r["method"] + " & " + " & ".join(
                f"{r[k][0]:.3f} $\\pm$ {r[k][1]:.3f}" for k in ("clean", "20 dB", "10 dB", "0 dB")) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}"]
        (out / f"table2_{tag}.tex").write_text("\n".join(tex) + "\n")
        print(f"\n== {tag} ==")
        print(pd.DataFrame(flat).to_string(index=False))
    byc = df.groupby(["method", "noise", "snr_db"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    byc.to_csv(out / "table2_by_noise_type.csv")


if __name__ == "__main__":
    main()
