"""Table and figure for RQ2 vs m (input: reject_m_per_run.csv from run_reject_m.py).

    python scripts/rejm_stage4.py --csv results_stage4_rejm/reject_m_per_run.csv --out paper/figures

Writes tab_reject_m.tex (0 dB, mean +- std over seeds; noise types averaged; no final row terminator, the paper adds it),
fig_reject_m.pdf (risk and coverage versus m at 20/10/0 dB) and rejm_summary.csv (all numbers).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLS = ["macro_f1", "coverage", "risk_chow", "risk_no_reject", "top30_acc"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    per = df.groupby(["snr_db", "m", "seed"])[COLS].mean().reset_index()      # average the noise types first
    mean = per.groupby(["snr_db", "m"])[COLS].mean()
    std = per.groupby(["snr_db", "m"])[COLS].std()
    summ = mean.join(std, rsuffix="_std").reset_index()
    summ.to_csv(out / "rejm_summary.csv", index=False)

    lines = []
    for m in sorted(df.m.unique()):
        r, s = mean.loc[(0, m)], std.loc[(0, m)]
        f = lambda c, d=3: f"{r[c]:.{d}f}\\,$\\pm$\\,{s[c]:.{d}f}"  # noqa: E731
        lab = "0 (B1)" if m == 0 else str(int(m))
        lines.append(f"{lab} & {f('macro_f1')} & {f('coverage')} & {f('risk_chow')} & {r['risk_no_reject']:.3f} \\\\")
    txt = "\n".join(lines).rstrip()
    (out / "tab_reject_m.tex").write_text((txt[:-2].rstrip() if txt.endswith("\\\\") else txt) + "\n")

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
    ms = [m for m in sorted(df.m.unique()) if m > 0]
    for snr, c in ((20, "#1f77b4"), (10, "#ff7f0e"), (0, "#2ca02c")):
        for k, (col, ttl) in enumerate((("risk_chow", "Expected risk with Chow's rule"), ("coverage", "Coverage"))):
            mu = [mean.loc[(snr, m), col] for m in ms]
            sd = [std.loc[(snr, m), col] for m in ms]
            ax[k].errorbar(ms, mu, yerr=sd, marker="o", ms=3, lw=1, color=c, capsize=2, label=f"{snr} dB")
            ax[k].axhline(mean.loc[(snr, 0), col], color=c, ls=":", lw=0.8)
            ax[k].set_title(ttl)
    ax[0].axhline(0.2, color="k", ls="--", lw=0.8); ax[0].text(ms[0], 0.205, "reject all = 0.2", fontsize=7)
    for x in ax:
        x.set_xscale("log"); x.set_xlabel("adaptation clips $m$ (dotted: B1, no adaptation)")
    ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig_reject_m.pdf"); plt.close(fig)
    print(summ[summ.snr_db == 0].round(3).to_string())


if __name__ == "__main__":
    main()
