"""Build the figures and LaTeX table fragments of the Stage 4 paper from the analysis outputs.

    python scripts/make_paper_stage4.py --results results_stage4 --analysis results_stage4/analysis --out paper/figures

Reads  : results_stage4/*.npz, ablation_per_run.csv, sensitivity_per_run.csv, and the CSVs written by
         analyze_stage4.py (table_significance_clipbootstrap.csv, table_reject_all_models.csv, ...).
Writes : fig_main.pdf, fig_sensitivity.pdf, fig_errors.pdf, fig_riskcov.pdf, fig_perclass.pdf (vector) and
         tab_main.tex, tab_ablation.tex, tab_sig.tex, tab_reject.tex (included with \\input in the paper).
Nothing is typed by hand: every number in these tables and plots comes from the result files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = ["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
C = len(LABELS)
SNRS = (20, 10, 0)
KINDS = ("babble", "environmental", "white")

# fixed method -> style map (blue/orange/aqua are slots 1-3 of the validated categorical palette; grey = B1 family)
INK, GREY = "#0b0b0b", "#6b6a66"
STY = {
    "B1": dict(color=GREY, ls="-", marker="s"),
    "B1-aug": dict(color=GREY, ls="--", marker="s"),
    "Proposed v2": dict(color="#2a78d6", ls="-", marker="o", lw=2.2),
    "B2": dict(color="#eb6834", ls="-", marker="^"),
    "B2 + AdaBN": dict(color="#eb6834", ls="--", marker="^"),
    "B2-aug": dict(color="#1baf7a", ls="-", marker="D"),
    "B2-aug + AdaKWS-style": dict(color="#1baf7a", ls="--", marker="D"),
}
NAME = {"Proposed v2": "Proposed v2", "B2 + AdaBN": "B2 + AdaBN", "B2-aug + AdaKWS-style": "B2-aug + AdaKWS-style"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 150, "pdf.fonttype": 42,
                     "axes.edgecolor": "#8a8985", "axes.labelcolor": "#2b2b2a", "xtick.color": "#2b2b2a",
                     "ytick.color": "#2b2b2a", "legend.frameon": False})


def f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())


def cm_of(y, p):
    return np.bincount(y * C + p.astype(np.int64), minlength=C * C).reshape(C, C)


def load(results: Path):
    y = np.load(results / "y_test.npy").astype(np.int64)
    conds = json.loads((results / "conditions.json").read_text())
    seeds = sorted(int(p.stem[4:]) for p in results.glob("seed*.npz"))
    data = {}
    for s in seeds:
        z = np.load(results / f"seed{s}.npz")
        for k in z.files:
            m, kind = k.rsplit("__", 1)
            data.setdefault(m, {}).setdefault(kind, {})[s] = z[k]
    return y, conds, seeds, data


def legend_below(fig, axs, ncol):
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=7.5, handlelength=2.6, columnspacing=1.4)


def pm(mean, std, nd=3, bold=False):
    s = f"{mean:.{nd}f}"
    if std is not None and not np.isnan(std) and std >= 0.0005:
        s += f"{{\\tiny\\,$\\pm$\\,{std:.{nd}f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def _write_rows(path, lines):
    """Write table rows WITHOUT the final row terminator: the paper does \\input{...}\\\\ so that the rule that follows
    is not read straight after the end of the input file (which makes LaTeX report a misplaced \\noalign)."""
    txt = "\n".join(lines).rstrip()
    if txt.endswith("\\\\"):
        txt = txt[:-2].rstrip()
    path.write_text(txt + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--dn", default=None, help="sens_dn_per_run.csv from run_sensitivity_dn.py (adds panels c, d to Figure 3)")
    a = ap.parse_args()
    res, ana, out = Path(a.results), Path(a.analysis), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y, conds, seeds, data = load(res)
    pr = pd.read_csv(ana / "per_run_stage4.csv")
    pr["noise"] = pr.condition.str.split("@").str[0]
    pr["snr"] = pr.condition.str.extract(r"@(\d+)dB")[0].astype(float)
    noisy = pr[pr.condition != "clean"]
    by_snr = noisy.groupby(["method", "snr", "seed"])[["macro_f1", "accuracy"]].mean().reset_index()
    ms = by_snr.groupby(["method", "snr"])[["macro_f1", "accuracy"]].agg(["mean", "std"])
    clean = pr[pr.condition == "clean"].groupby("method")["macro_f1"].agg(["mean", "std"])
    sig = pd.read_csv(ana / "table_significance_clipbootstrap.csv")
    if "p_holm" not in sig:
        sig["p_holm"] = sig.p_boot

    # ------------------------------------------------------------------ Figure: main results
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw=dict(width_ratios=[1.05, 1]))
    xs = [0, 1, 2, 3]
    for m in ("B1", "B1-aug", "Proposed v2", "B2", "B2 + AdaBN", "B2-aug"):
        yv = [clean.loc[m, "mean"]] + [ms.loc[(m, s), ("macro_f1", "mean")] for s in SNRS]
        ev = [0.0] + [ms.loc[(m, s), ("macro_f1", "std")] for s in SNRS]
        st = dict(STY[m]); lw = st.pop("lw", 1.5)
        ax[0].errorbar(xs, yv, ev, label=m, lw=lw, ms=4.5, capsize=2, **st)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(["clean", "20 dB", "10 dB", "0 dB"])
    ax[0].set_ylabel("macro-F1 (mean $\\pm$ std, 5 seeds)"); ax[0].set_ylim(0, 1.0); ax[0].set_title("(a) by SNR, noise types averaged", fontsize=8)
    z0 = noisy[noisy.snr == 0].groupby(["method", "noise", "seed"])["macro_f1"].mean().reset_index()
    z0 = z0.groupby(["method", "noise"])["macro_f1"].agg(["mean", "std"])
    meths = ["B1", "Proposed v2", "B2", "B2 + AdaBN", "B2-aug"]
    w = 0.16
    for i, m in enumerate(meths):
        mu = [z0.loc[(m, k), "mean"] for k in KINDS]
        sd = [z0.loc[(m, k), "std"] for k in KINDS]
        st = STY[m]
        ax[1].bar(np.arange(3) + (i - 2) * w, mu, w * 0.88, yerr=sd, capsize=1.5, color=st["color"],
                  hatch="////" if st["ls"] == "--" else None, edgecolor="white" if st["ls"] != "--" else st["color"],
                  linewidth=0.4, error_kw=dict(lw=0.7), label=m)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["babble", "environmental", "white"])
    ax[1].set_ylim(0, 1.0); ax[1].set_title("(b) at 0 dB by noise type", fontsize=8)
    ax[1].set_ylabel("macro-F1")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    h0, l0 = ax[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=6, fontsize=7, handlelength=2.4, columnspacing=1.1)
    fig.savefig(out / "fig_main.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Figure: sensitivity
    se = pd.read_csv(res / "sensitivity_per_run.csv")
    ab = pd.read_csv(res / "ablation_per_run.csv")
    dn = pd.read_csv(a.dn) if a.dn and Path(a.dn).exists() else None
    nrow = 2 if dn is not None else 1
    fig, axs = plt.subplots(nrow, 2, figsize=(6.5, 2.45 * nrow), squeeze=False)
    ax = axs[0]
    b1 = noisy[noisy.method == "B1"].groupby(["snr", "seed"])["macro_f1"].mean().groupby("snr").mean()
    sty = {20: ":", 10: "--", 0: "-"}
    for snr in SNRS:
        sub = se[(se.sens == "m") & (se.snr_db == snr)]
        g = sub.groupby(["value", "seed"])["macro_f1"].mean().groupby("value").agg(["mean", "std"])
        ax[0].errorbar(g.index, g["mean"], g["std"], ls=sty[snr], marker="o", ms=4.5, capsize=2, color="#2a78d6", lw=1.8,
                       label=f"v2, {snr} dB")
        ax[0].axhline(b1[snr], color=GREY, ls=sty[snr], lw=1.0)
    ax[0].set_xscale("log"); ax[0].set_xlabel("unlabeled adaptation clips $m$ (log scale)")
    ax[0].set_ylabel("macro-F1"); ax[0].set_title("(a) amount of adaptation data", fontsize=8)
    ax[0].text(11, b1[0] - 0.035, "grey: B1 without adaptation", fontsize=6.5, color="#2b2b2a")
    noad = ab[(ab.variant == "no adaptation") & (ab.snr_db == 0)].groupby(["noise", "seed"])["macro_f1"].mean().groupby("noise").mean()
    sub = se[(se.sens == "unknown_fraction")]
    for kind, col, mk in zip(KINDS, ("#2a78d6", "#eb6834", "#1baf7a"), ("o", "^", "D")):
        g = sub[sub.noise == kind].groupby(["value", "seed"])["macro_f1"].mean().groupby("value").agg(["mean", "std"])
        ax[1].errorbar(g.index, g["mean"], g["std"], marker=mk, ms=4.5, capsize=2, color=col, lw=1.6, label=f"v2, {kind}")
        ax[1].axhline(noad[kind], color=col, ls=":", lw=1.0)
    ax[1].axvline(0.607, color="#8a8985", lw=0.8, ls="-.")
    ax[1].text(0.62, 0.015, "natural", fontsize=6.5, color="#2b2b2a")
    ax[1].set_xlabel("fraction of 'unknown' clips in the adaptation set")
    ax[1].set_title("(b) class balance of the adaptation set, 0 dB", fontsize=8)
    ax[1].set_ylim(0, 0.4); ax[0].set_ylim(0.1, 0.55)
    if dn is not None:
        for axx, sens, xl, ttl in ((axs[1][0], "d", "PCA dimension $d$", "(c) feature dimension"),
                                   (axs[1][1], "n_train", "fraction of the training set (log scale)", "(d) training-set size")):
            sub = dn[dn.sens == sens]
            for snr in SNRS:
                for meth, col in (("Proposed v2", "#2a78d6"), ("B1", GREY)):
                    g = sub[(sub.method == meth) & (sub.snr_db == snr)].groupby(["value", "seed"])["macro_f1"].mean().groupby("value").agg(["mean", "std"])
                    axx.errorbar(g.index, g["mean"], g["std"], ls=sty[snr], marker="o" if meth != "B1" else "s", ms=4,
                                 capsize=1.5, color=col, lw=1.6 if meth != "B1" else 1.1)
            axx.set_xlabel(xl); axx.set_title(ttl, fontsize=8); axx.set_ylabel("macro-F1")
            if sens == "n_train":
                axx.set_xscale("log")
        axs[1][0].text(0.03, 0.04, "blue: v2, grey: B1; dotted/dashed/solid: 20/10/0 dB", transform=axs[1][0].transAxes, fontsize=6.3)
    fig.tight_layout(rect=(0, 0.075 if nrow == 2 else 0.13, 1, 1))
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=6, fontsize=7, handlelength=2.2, columnspacing=1.0)
    fig.savefig(out / "fig_sensitivity.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Figure: confusion matrices at 0 dB
    def agg_cm(meth, cids):
        cm = np.zeros((C, C))
        for s in seeds:
            for k in cids:
                cm += cm_of(y, data[meth]["pred"][s][k])
        return cm / (len(seeds) * len(cids))
    c0 = [i for i, c in enumerate(conds) if c.endswith("@0dB")]
    rn = lambda m: m / m.sum(1, keepdims=True)  # noqa: E731
    cmB, cmA = rn(agg_cm("B1", c0)), rn(agg_cm("Proposed v2", c0))
    d = cmA - cmB
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.55))
    for k, (cm, t) in enumerate(((cmB, "B1"), (cmA, "proposed v2"), (d, "difference (v2 $-$ B1)"))):
        vmax = 1.0 if k < 2 else float(np.abs(d).max())
        im = axs[k].imshow(cm, cmap="Blues" if k < 2 else "RdBu_r", vmin=0 if k < 2 else -vmax, vmax=vmax)
        axs[k].set_title(t, fontsize=8)
        axs[k].set_xticks(range(C)); axs[k].set_xticklabels(LABELS, rotation=90, fontsize=5.5)
        axs[k].set_yticks(range(C)); axs[k].set_yticklabels(LABELS if k == 0 else [], fontsize=5.5)
        axs[k].grid(False)
        cb = fig.colorbar(im, ax=axs[k], fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5.5)
    axs[0].set_ylabel("true class", fontsize=7)
    fig.supxlabel("predicted class (rows normalised; mean over the three noise types and 5 seeds, 0 dB)", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig_errors.pdf"); plt.close(fig)

    # per-class F1 change by noise type
    rows = []
    for kind in KINDS:
        ck = [i for i, c in enumerate(conds) if c == f"{kind}@0dB"]
        for meth in ("B1", "Proposed v2"):
            cm = agg_cm(meth, ck)
            f = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
            rows += [{"noise": kind, "method": meth, "class": LABELS[c], "f1": f[c]} for c in range(C)]
    pc = pd.DataFrame(rows).pivot_table(index=["noise", "class"], columns="method", values="f1")
    pc["delta"] = pc["Proposed v2"] - pc["B1"]
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    for i, (kind, col) in enumerate(zip(KINDS, ("#2a78d6", "#eb6834", "#1baf7a"))):
        ax.bar(np.arange(C) + (i - 1) * 0.27, pc.loc[kind].reindex(LABELS)["delta"].values, 0.25, color=col, label=kind)
    ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(range(C)); ax.set_xticklabels(LABELS, rotation=40, ha="right")
    ax.set_ylabel("per-class F1 change\n(v2 $-$ B1), 0 dB"); ax.legend(ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_perclass.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Figure: risk-coverage
    rm = ("B1", "B1-aug", "Proposed v2", "B2", "B2-aug")
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.55), sharey=True)
    for axx, snr in zip(axs, SNRS):
        cids = [i for i, c in enumerate(conds) if c.endswith(f"@{snr}dB")]
        for m in rm:
            conf = np.concatenate([data[m]["confcal"][s][k] for s in seeds for k in cids]).astype(np.float64)
            pred = np.concatenate([data[m]["pred"][s][k] for s in seeds for k in cids]).astype(np.int64)
            ok = pred == np.tile(y, len(seeds) * len(cids))
            o = np.argsort(-conf); cs = np.cumsum(ok[o]); n = np.arange(1, len(ok) + 1)
            st = dict(STY[m]); lw = st.pop("lw", 1.5); st.pop("marker")
            axx.plot((n / len(ok))[::40], (cs / n)[::40], lw=lw, label=m, **st)
        axx.set_title(f"{snr} dB", fontsize=8); axx.set_xlabel("coverage (fraction of clips answered)")
        axx.set_xlim(0, 1)
    axs[0].set_ylabel("accuracy on answered clips"); axs[0].set_ylim(0.1, 1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    h0, l0 = axs[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=5, fontsize=7, handlelength=2.6, columnspacing=1.2)
    fig.savefig(out / "fig_riskcov.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Table: main
    def sigrow(A, B, scope):
        r = sig[(sig.A == A) & (sig.B == B) & (sig.scope == scope)]
        if len(r):
            r = r.iloc[0]; return r.diff_macro_f1, r.ci_lo, r.ci_hi, r.p_holm
        r = sig[(sig.A == B) & (sig.B == A) & (sig.scope == scope)]
        if len(r):
            r = r.iloc[0]; return -r.diff_macro_f1, -r.ci_hi, -r.ci_lo, r.p_holm
        return None
    base = {"Proposed v2": "B1", "v2+EM": "Proposed v2", "B1-aug + v2": "B1-aug", "B2 + AdaBN": "B2", "B2 + Tent": "B2",
            "B2 + AdaKWS-style": "B2", "B2-aug + AdaBN": "B2-aug", "B2-aug + Tent": "B2-aug",
            "B2-aug + AdaKWS-style": "B2-aug"}
    disp = {"B3 majority class": "B3 majority class", "B1": "B1 Gaussian Bayes", "B1-aug": "B1-aug", "B2": "B2 TC-ResNet8",
            "B2-aug": "B2-aug", "Proposed v2": "\\textbf{Proposed v2 (shared affine)}", "v2+EM": "v2+EM (v2, then anchored EM)",
            "B1-aug + v2": "B1-aug + v2", "B2 + AdaBN": "B2 + AdaBN", "B2 + Tent": "B2 + Tent",
            "B2 + AdaKWS-style": "B2 + AdaKWS-style", "B2-aug + AdaBN": "B2-aug + AdaBN", "B2-aug + Tent": "B2-aug + Tent",
            "B2-aug + AdaKWS-style": "B2-aug + AdaKWS-style"}
    top = ["B3 majority class", "B1", "B1-aug", "B2", "B2-aug"]
    adp = ["Proposed v2", "v2+EM", "B1-aug + v2", "B2 + AdaBN", "B2 + Tent", "B2 + AdaKWS-style", "B2-aug + AdaBN",
           "B2-aug + Tent", "B2-aug + AdaKWS-style"]
    L = []
    def row(m, bold=False):
        cells = [disp[m], pm(clean.loc[m, "mean"], clean.loc[m, "std"], bold=bold)]
        for s in SNRS:
            cells.append(pm(ms.loc[(m, s), ("macro_f1", "mean")], ms.loc[(m, s), ("macro_f1", "std")], bold=bold))
        cells.append(pm(ms.loc[(m, 0), ("accuracy", "mean")], ms.loc[(m, 0), ("accuracy", "std")]))
        r = sigrow(m, base[m], "all noises @0dB") if m in base else None
        if r:
            d, lo, hi, ph = r
            cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("$^\\dagger$" if m == "v2+EM" else ""))
        else:
            cells.append("---")
        L.append(" & ".join(cells) + " \\\\")
    for m in top:
        row(m)
    L.append("\\midrule")
    for m in adp:
        row(m, bold=(m == "Proposed v2"))
    _write_rows(out / "tab_main.tex", L)

    # ------------------------------------------------------------------ Table: ablation
    abn = ab[ab.condition != "clean"]
    t1 = abn.groupby(["variant", "snr_db", "seed"])["macro_f1"].mean().reset_index().groupby(["variant", "snr_db"])["macro_f1"].agg(["mean", "std"])
    t2 = abn[abn.snr_db == 0].groupby(["variant", "noise", "seed"])["macro_f1"].mean().reset_index().groupby(["variant", "noise"])["macro_f1"].agg(["mean", "std"])
    order = [("no adaptation", "no adaptation (B1)"), ("shift only", "shift only"), ("scale only", "scale only"),
             ("full (shift + scale)", "\\textbf{full (shift + scale), $\\kappa=20$}"), ("kappa=1", "$\\kappa=1$"),
             ("kappa=5", "$\\kappa=5$"), ("kappa=100", "$\\kappa=100$"), ("kappa=1000", "$\\kappa=1000$")]
    L = []
    for v, nm in order:
        cells = [nm] + [pm(t1.loc[(v, s), "mean"], t1.loc[(v, s), "std"]) for s in SNRS] + \
                [pm(t2.loc[(v, k), "mean"], t2.loc[(v, k), "std"]) for k in KINDS]
        L.append(" & ".join(cells) + " \\\\")
        if v == "scale only":
            L.append("\\midrule")
        if v == "full (shift + scale)":
            L.append("\\midrule")
    _write_rows(out / "tab_ablation.tex", L)

    # ------------------------------------------------------------------ Table: significance
    spec = [("Proposed v2", "B1", "all noises @20dB", "v2 $-$ B1, 20 dB"), ("Proposed v2", "B1", "all noises @10dB", "v2 $-$ B1, 10 dB"),
            ("Proposed v2", "B1", "all noises @0dB", "v2 $-$ B1, 0 dB"), ("Proposed v2", "B1", "babble@0dB", "\\quad babble"),
            ("Proposed v2", "B1", "environmental@0dB", "\\quad environmental"), ("Proposed v2", "B1", "white@0dB", "\\quad white"),
            ("Proposed v2", "B1-aug", "all noises @0dB", "v2 $-$ B1-aug, 0 dB"),
            ("Proposed v2", "v2+EM", "all noises @0dB", "v2 $-$ v2+EM, 0 dB"),
            ("B2 + AdaBN", "B2", "all noises @0dB", "B2+AdaBN $-$ B2, 0 dB"),
            ("B2 + Tent", "B2 + AdaBN", "all noises @0dB", "B2+Tent $-$ B2+AdaBN, 0 dB"),
            ("B2 + AdaKWS-style", "B2 + AdaBN", "all noises @0dB", "B2+AdaKWS $-$ B2+AdaBN, 0 dB"),
            ("B2-aug + AdaBN", "B2-aug", "all noises @0dB", "B2-aug+AdaBN $-$ B2-aug, 0 dB"),
            ("B2-aug + AdaKWS-style", "B2-aug", "all noises @0dB", "B2-aug+AdaKWS $-$ B2-aug, 0 dB"),
            ("B2 + AdaKWS-style", "Proposed v2", "all noises @0dB", "B2+AdaKWS $-$ v2, 0 dB"),
            ("B2-aug", "Proposed v2", "all noises @0dB", "B2-aug $-$ v2, 0 dB")]
    L = []
    for A, B, scope, nm in spec:
        r = sigrow(A, B, scope)
        if r is None:
            continue
        d, lo, hi, ph = r
        ps = "$<10^{-3}$" if ph < 1e-3 else f"{ph:.3f}"
        L.append(f"{nm} & {d:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {ps} \\\\")
    _write_rows(out / "tab_sig.tex", L)

    # ------------------------------------------------------------------ Table: reject rule on every model
    rj = pd.read_csv(ana / "table_reject_all_models.csv")
    L = []
    for m in ("B1", "B1-aug", "Proposed v2", "B2", "B2-aug", "B2-aug + AdaKWS-style"):
        sub = rj[rj.method == m].set_index("snr_db")
        if sub.empty:
            continue
        cells = [disp.get(m, m)]
        for s in SNRS:
            cells.append(f"{sub.loc[s, 'risk_no_reject']:.3f}$\\to${sub.loc[s, 'risk_chow']:.3f}")
        cells += [f"{sub.loc[0, 'coverage']:.2f}", f"{sub.loc[0, 'acc_at_cov30']:.3f}", f"{sub.loc[0, 'acc_at_cov50']:.3f}"]
        L.append(" & ".join(cells) + " \\\\")
    _write_rows(out / "tab_reject.tex", L)
    print("wrote figures and tables to", out)


if __name__ == "__main__":
    main()
