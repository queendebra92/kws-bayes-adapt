"""One command for the whole baseline table (Table 2: B3, B1, B1-aug, B2, B2-aug).

    python reproduce_baselines.py --sc-root D:\\data\\speech_commands_v0.02 --musan-root D:\\data\\musan

Runs, in order: Gaussian Bayes + majority floor (B3, B1, B1-aug; eps frozen at 0.01),
TC-ResNet8 clean-trained (B2), TC-ResNet8 noise-augmented (B2-aug), then merges everything
into results_final/ (per_run.csv, table2_*.csv, table2_macro_f1.tex).
Quick check first:  add  --n-seeds 1 --epochs 2
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-root", required=True, help="speech_commands_v0.02 folder")
    ap.add_argument("--musan-root", required=True, help="musan folder (noise/ and speech/)")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results_final", help="folder for the merged table")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.01, help="B1 ridge, frozen after validation selection")
    a = ap.parse_args()
    out = Path(a.out)
    gb, tc = out / "gaussian", out / "tcresnet"
    py = sys.executable
    run([py, "run_baselines.py", "--sc-root", a.sc_root, "--musan-root", a.musan_root,
         "--cache-dir", a.cache_dir, "--out", gb, "--n-seeds", a.n_seeds, "--eps", a.eps])
    for ts in ("clean", "aug"):
        run([py, "run_tcresnet.py", "--musan-root", a.musan_root, "--cache-dir", a.cache_dir,
             "--out", tc, "--train-set", ts, "--n-seeds", a.n_seeds, "--epochs", a.epochs])
    run([py, "scripts/make_table2.py", "--out", out, gb / "per_run.csv",
         tc / "per_run_clean.csv", tc / "per_run_aug.csv"])
    print(f"\nDone. Table 2: {out / 'table2_macro_f1.tex'}  (all runs: {out / 'per_run_all.csv'})")


if __name__ == "__main__":
    main()
