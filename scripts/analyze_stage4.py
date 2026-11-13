"""Analyse the Stage 4 outputs (results_stage4/): tables, significance tests, error analysis, figures.

    python scripts/analyze_stage4.py --results results_stage4 --out stage4_analysis [--boot 1000]

Reads seed*.npz (per-clip predictions and confidences), y_test.npy, conditions.json,
ablation_per_run.csv and sensitivity_per_run.csv. Writes CSV tables, LaTeX table bodies and vector
figures (PDF). No new model is trained or evaluated here; every number is a function of the saved
per-clip predictions, so every table and figure is regenerable from the results folder.

Significance: the unit of resampling is the test CLIP. In each bootstrap replicate the same resampled
clip set is used for every seed, condition and method, and the statistic is recomputed (macro-F1
averaged over the noisy conditions of an SNR bucket and over the seeds). This captures the finite test
set, which the paired t-test over 5 seeds does not (the seeds differ only in the noise draw).
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

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "pdf.fonttype": 42})
COL = {"B1": "#6b7280", "B1-aug": "#9ca3af", "Proposed v2": "#0f766e", "v2+EM": "#14b8a6",
       "B2": "#c2410c", "B2-aug": "#ea580c", "B2-aug + AdaKWS-style": "#7c3aed", "B2 + AdaKWS-style": "#a78bfa",
       "B1-aug + v2": "#047857"}


def f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    p = cm.sum(axis=0)
    r = cm.sum(axis=1)
    denom = p + r
    f = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f.mean())


def cm_w(y, p, w=None):
    return np.bincount(y * C + p.astype(np.int64), weights=w, minlength=C * C).reshape(C, C)


def load(results: Path):
    y = np.load(results / "y_test.npy").astype(np.int64)
    conds = json.loads((results / "conditions.json").read_text())
    seeds = sorted(int(p.stem[4:]) for p in results.glob("seed*.npz"))
    data = {}
    for s in seeds:
        z = np.load(results / f"seed{s}.npz")
        for k in z.files:
            meth, kind = k.rsplit("__", 1)
            data.setdefault(meth, {}).setdefault(kind, {})[s] = z[k]
    return y, conds, seeds, data


def cond_ids(conds, snr=None, kind=None):
    out = []
    for i, c in enumerate(conds):
        if c == "clean":
            continue
        k, s = c.split("@")
        if (snr is None or int(s[:-2]) == snr) and (kind is None or k == kind):
            out.append(i)
    return out


def metric_table(y, conds, seeds, data):
    rows = []
    for meth, d in data.items():
        for s in seeds:
            P = d["pred"][s]
            for ci, c in enumerate(conds):
                p = P[ci].astype(np.int64)
                rows.append({"method": meth, "seed": s, "condition": c,
                             "macro_f1": f1_from_cm(cm_w(y, p)), "accuracy": float((p == y).mean())})
    return pd.DataFrame(rows)


def boot_diff(y, seeds, data, a, b, cids, B, rng):
    """Bootstrap over clips of mean (macro-F1_a - macro-F1_b) over seeds and the conditions in cids."""
    n = len(y)
    key = y * C
    pa = {s: data[a]["pred"][s][cids].astype(np.int64) for s in seeds}
    pb = {s: data[b]["pred"][s][cids].astype(np.int64) for s in seeds}
    def stat(w):
        d = []
        for s in seeds:
            for k in range(len(cids)):
                d.append(f1_from_cm(np.bincount(key + pa[s][k], weights=w, minlength=C * C).reshape(C, C)) -
                         f1_from_cm(np.bincount(key + pb[s][k], weights=w, minlength=C * C).reshape(C, C)))
        return float(np.mean(d))
    obs = stat(None)
    reps = np.array([stat(rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)) for _ in range(B)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
    return obs, lo, hi, max(p, 1.0 / (B + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_stage4")
    ap.add_argument("--out", default="stage4_analysis")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--lambda-r", type=float, default=0.2)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y, conds, seeds, data = load(res)
    rng = np.random.default_rng(0)
    print(f"{len(seeds)} seeds, {len(conds)} conditions, {len(y)} test clips, methods: {sorted(data)}")

    # ---------------- 1. main table (mean +- std over seeds, per SNR over the three noise types) -----
    df = metric_table(y, conds, seeds, data)
    df.to_csv(out / "per_run_stage4.csv", index=False)
    clean = df[df.condition == "clean"].groupby("method")[["macro_f1", "accuracy"]].mean()
    noisy = df[df.condition != "clean"].copy()
    noisy["snr"] = noisy.condition.str.extract(r"@(\d+)dB").astype(int)
    per_seed = noisy.groupby(["method", "snr", "seed"])[["macro_f1", "accuracy"]].mean().reset_index()
    main = per_seed.groupby(["method", "snr"])[["macro_f1", "accuracy"]].agg(["mean", "std"])
    main.columns = ["_".join(c) for c in main.columns]
    main = main.reset_index()
    main.to_csv(out / "table_main_by_snr.csv", index=False)
    allnoisy = noisy.groupby(["method", "seed"])[["macro_f1", "accuracy"]].mean().groupby("method").agg(["mean", "std"])
    allnoisy.columns = ["_".join(c) for c in allnoisy.columns]
    allnoisy.to_csv(out / "table_main_all_noisy.csv")

    # ---------------- 2. significance over clips ------------------------------------------------
    pairs = [("Proposed v2", "B1"), ("Proposed v2", "B1-aug"), ("B1-aug + v2", "B1-aug"), ("Proposed v2", "v2+EM"),
             ("B2-aug", "Proposed v2"), ("B2", "Proposed v2"), ("B2 + AdaBN", "B2"), ("B2 + Tent", "B2"),
             ("B2 + AdaKWS-style", "B2"), ("B2 + Tent", "B2 + AdaBN"), ("B2 + AdaKWS-style", "B2 + AdaBN"),
             ("B2-aug + AdaBN", "B2-aug"), ("B2-aug + Tent", "B2-aug"), ("B2-aug + AdaKWS-style", "B2-aug"),
             ("B2 + AdaKWS-style", "Proposed v2")]
    sig = []
    for A, Bm in pairs:
        if A not in data or Bm not in data:
            continue
        for snr in SNRS:
            o, lo, hi, p = boot_diff(y, seeds, data, A, Bm, cond_ids(conds, snr), a.boot, rng)
            sig.append({"A": A, "B": Bm, "scope": f"all noises @{snr}dB", "diff_macro_f1": o, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
        for kind in KINDS:
            o, lo, hi, p = boot_diff(y, seeds, data, A, Bm, cond_ids(conds, 0, kind), a.boot, rng)
            sig.append({"A": A, "B": Bm, "scope": f"{kind}@0dB", "diff_macro_f1": o, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
        print("  bootstrap", A, "vs", Bm, flush=True)
    sg = pd.DataFrame(sig)
    # Holm step-down correction over the whole family of comparisons in this table
    order = np.argsort(sg.p_boot.values)
    m_tests, adj, run = len(sg), np.empty(len(sg)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m_tests - rank) * sg.p_boot.values[i]))
        adj[i] = run
    sg["p_holm"] = adj
    sg.to_csv(out / "table_significance_clipbootstrap.csv", index=False)

    # ---------------- 3. error analysis ------------------------------------------------------------
    def agg_cm(meth, cids):
        cm = np.zeros((C, C))
        for s in seeds:
            for k in cids:
                cm += cm_w(y, data[meth]["pred"][s][k].astype(np.int64))
        return cm / (len(seeds) * len(cids))
    err = []
    cids0 = cond_ids(conds, 0)
    for A, Bm in (("Proposed v2", "B1"),):
        if A in data and Bm in data:
            cmA, cmB = agg_cm(A, cids0), agg_cm(Bm, cids0)
            rn = lambda cm: cm / cm.sum(axis=1, keepdims=True)  # noqa: E731
            d = rn(cmA) - rn(cmB)
            fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.2))
            for k, (cm, t) in enumerate(((rn(cmB), "B1 (no adaptation)"), (rn(cmA), "Proposed v2"), (d, "Difference (v2 - B1)"))):
                vmax = 1.0 if k < 2 else float(np.abs(d).max())
                im = ax[k].imshow(cm, cmap="Blues" if k < 2 else "RdBu_r", vmin=0 if k < 2 else -vmax, vmax=vmax)
                ax[k].set_title(t, fontsize=9)
                ax[k].set_xticks(range(C)); ax[k].set_xticklabels(LABELS, rotation=90, fontsize=6)
                ax[k].set_yticks(range(C)); ax[k].set_yticklabels(LABELS if k == 0 else [], fontsize=6)
                ax[k].grid(False)
            fig.supxlabel("predicted", fontsize=8); fig.tight_layout()
            fig.savefig(out / "fig_confusion_0dB.pdf"); plt.close(fig)
            # per-class F1 at 0 dB by noise type
            rows = []
            for kind in KINDS:
                ck = cond_ids(conds, 0, kind)
                for meth in (A, Bm):
                    cm = agg_cm(meth, ck)
                    tp = np.diag(cm); f = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
                    for c in range(C):
                        rows.append({"noise": kind, "method": meth, "class": LABELS[c], "f1": f[c]})
            pc = pd.DataFrame(rows)
            pc.to_csv(out / "table_per_class_f1_0dB.csv", index=False)
            piv = pc.pivot_table(index=["noise", "class"], columns="method", values="f1")
            piv["delta"] = piv[A] - piv[Bm]
            piv.to_csv(out / "table_per_class_delta_0dB.csv")
            fig, ax = plt.subplots(figsize=(7.2, 2.8))
            w = 0.27
            for i, kind in enumerate(KINDS):
                dd = piv.loc[kind].reindex(LABELS)["delta"].values
                ax.bar(np.arange(C) + (i - 1) * w, dd, w, label=kind)
            ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(C)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
            ax.set_ylabel("per-class F1 change\n(v2 - B1) at 0 dB"); ax.legend(frameon=False, fontsize=7, ncol=3)
            fig.tight_layout(); fig.savefig(out / "fig_perclass_delta_0dB.pdf"); plt.close(fig)
            # most confused pairs (off-diagonal mass, 0 dB, all noise)
            off = rn(cmB).copy(); np.fill_diagonal(off, 0)
            offA = rn(cmA).copy(); np.fill_diagonal(offA, 0)
            for nm, m in (("B1", off), ("v2", offA)):
                idx = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:8], m.shape))[0]
                for i, j in idx:
                    err.append({"method": nm, "true": LABELS[i], "predicted": LABELS[j], "row_fraction": m[i, j]})
    pd.DataFrame(err).to_csv(out / "table_top_confusions_0dB.csv", index=False)

    # ---------------- 4. reject rule on every model (risk-coverage) -----------------------------
    lam = a.lambda_r
    meths = [m for m in ("B1", "B1-aug", "Proposed v2", "B2", "B2-aug", "B1-aug + v2", "B2-aug + AdaKWS-style") if m in data]
    rr = []
    curves = {}
    for meth in meths:
        for snr in SNRS:
            cids = cond_ids(conds, snr)
            conf = np.concatenate([data[meth]["confcal"][s][k] for s in seeds for k in cids]).astype(np.float64)
            pred = np.concatenate([data[meth]["pred"][s][k] for s in seeds for k in cids]).astype(np.int64)
            yy = np.tile(y, len(seeds) * len(cids))
            ok = pred == yy
            thr = 1.0 - lam
            acc_m = conf >= thr
            cov = acc_m.mean()
            risk = ((~ok & acc_m).sum() + lam * (~acc_m).sum()) / len(yy)
            rr.append({"method": meth, "snr_db": snr, "coverage": cov, "acc_on_accepted": ok[acc_m].mean() if acc_m.any() else np.nan,
                       "risk_chow": risk, "risk_reject_all": lam, "risk_no_reject": 1 - ok.mean()})
            order = np.argsort(-conf)
            cs = np.cumsum(ok[order]); n = np.arange(1, len(yy) + 1)
            curves[(meth, snr)] = (n / len(yy), cs / n)
            # accuracy at matched coverage
            for target in (0.3, 0.5, 0.7):
                kk = int(target * len(yy))
                rr[-1][f"acc_at_cov{int(target * 100)}"] = cs[kk - 1] / kk
    rr = pd.DataFrame(rr)
    rr.to_csv(out / "table_reject_all_models.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    for ax, snr in zip(axs, SNRS):
        for meth in meths:
            x, yv = curves[(meth, snr)]
            ax.plot(x[::50], yv[::50], label=meth, color=COL.get(meth), lw=1.4)
        ax.set_title(f"{snr} dB"); ax.set_xlabel("coverage")
    axs[0].set_ylabel("accuracy on accepted clips"); axs[2].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig_risk_coverage.pdf"); plt.close(fig)

    # ---------------- 5. ablations and sensitivity ----------------------------------------------
    ab_path = res / "ablation_per_run.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        ab = ab[ab.condition != "clean"]
        ps = ab.groupby(["variant", "snr_db", "seed"])[["macro_f1", "accuracy"]].mean().reset_index()
        t = ps.groupby(["variant", "snr_db"])[["macro_f1", "accuracy"]].agg(["mean", "std"])
        t.columns = ["_".join(c) for c in t.columns]
        t.reset_index().to_csv(out / "table_ablation.csv", index=False)
        pk = ab[ab.snr_db == 0].groupby(["variant", "noise", "seed"])["macro_f1"].mean().reset_index()
        pk.groupby(["variant", "noise"])["macro_f1"].agg(["mean", "std"]).reset_index().to_csv(out / "table_ablation_by_noise_0dB.csv", index=False)
    se_path = res / "sensitivity_per_run.csv"
    if se_path.exists():
        se = pd.read_csv(se_path)
        b1 = df[(df.method == "B1") & (df.condition != "clean")].copy()
        b1["noise"] = b1.condition.str.split("@").str[0]; b1["snr_db"] = b1.condition.str.extract(r"@(\d+)dB").astype(int)
        for sens, xl, fn in (("m", "adaptation clips m (log scale)", "fig_sensitivity_m.pdf"),
                             ("unknown_fraction", "fraction of 'unknown' in the adaptation set", "fig_sensitivity_unknown.pdf")):
            s2 = se[se.sens == sens]
            if s2.empty:
                continue
            fig, ax = plt.subplots(figsize=(3.6, 2.7))
            for snr, ls in ((20, ":"), (10, "--"), (0, "-")):
                sub = s2[s2.snr_db == snr]
                if sub.empty:
                    continue
                g = sub.groupby(["value", "seed"])["macro_f1"].mean().groupby("value").agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], g["std"], label=f"v2 {snr} dB", ls=ls, marker="o", ms=3, capsize=2, color=COL["Proposed v2"])
                ref = b1[b1.snr_db == snr].groupby("seed")["macro_f1"].mean().mean()
                ax.axhline(ref, color=COL["B1"], ls=ls, lw=0.9)
            if sens == "m":
                ax.set_xscale("log")
            ax.set_xlabel(xl); ax.set_ylabel("macro-F1"); ax.legend(frameon=False, fontsize=6.5)
            fig.tight_layout(); fig.savefig(out / fn); plt.close(fig)
        g = se.groupby(["sens", "value", "snr_db", "seed"])["macro_f1"].mean().reset_index().groupby(["sens", "value", "snr_db"])["macro_f1"].agg(["mean", "std"]).reset_index()
        g.to_csv(out / "table_sensitivity.csv", index=False)

    print(f"\nwrote tables and figures to {out}/")
    print(main.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
