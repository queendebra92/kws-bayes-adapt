# Bayesian generative adaptation with minimum-risk rejection for noise-robust KWS

COSC 750 Pattern Recognition I, Fall 2026, Bowie State University (D. Adeyemi).

Task: 12-class keyword spotting on Speech Commands v2 (ten keywords, *unknown*,
*silence*) under additive noise (MUSAN babble / environmental / white at 20, 10,
0 dB SNR). Stage 3 baselines and the proposed method (unlabeled shared-affine adaptation of
class-conditional Gaussians + Chow reject rule) are implemented here.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Data (not included, download separately):

* Speech Commands v0.02 (CC-BY 4.0): `speech_commands_v0.02.tar.gz`, extract to a folder that
  contains `yes/`, `no/`, ..., `_background_noise_/`, `validation_list.txt`, `testing_list.txt`.
* MUSAN (CC-BY 4.0): `musan.tar.gz`, extract to a folder that contains `noise/` and `speech/`.

## Reproduce the baseline table (one command)

```
python reproduce_baselines.py --sc-root D:\data\speech_commands_v0.02 --musan-root D:\data\musan
```

This runs B3 and B1/B1-aug (`run_baselines.py`, ridge frozen at `--eps 0.01`), TC-ResNet8 clean-trained (B2) and
noise-augmented (B2-aug) (`run_tcresnet.py`), and merges everything with `scripts/make_table2.py`.
Outputs in `results_final/`: `table2_macro_f1.tex` (Table 2 of the paper), `table2_macro_f1.csv`, `per_run_all.csv`
(every method x condition x seed) and the per-run folders `gaussian/` and `tcresnet/`.
Check the pipeline first with `--n-seeds 1 --epochs 2`. A full run takes roughly two hours on a laptop CPU
(B2 about 5 min per seed, B2-aug about 15 min per seed), plus a one-time audio decode into `cache/` (about 3.4 GB).
The individual steps are listed below.

## Reproduce the neural baseline

```
python run_tcresnet.py --musan-root D:\data\musan --cache-dir cache --out results_tcresnet --train-set clean
python run_tcresnet.py --musan-root D:\data\musan --cache-dir cache --out results_tcresnet --train-set aug
python scripts/make_table2.py --out results_final results/per_run.csv results_tcresnet/per_run_clean.csv results_tcresnet/per_run_aug.csv
```

Quick check before the full run: add `--n-seeds 1 --epochs 2`. Seed *s* controls weight initialisation, data order,
augmentation and the test-noise realisation (the same noisy test audio as the Gaussian-Bayes rows). The epoch with the
best validation macro-F1 is kept. If `pip install torch` fails on a very new Python, create the venv with Python 3.12 or 3.13.

## Proposed method (preliminary): unsupervised adaptation + reject rule

```
python run_adapt.py --musan-root D:\data\musan --cache-dir cache --out results_adapt_v2 --n-adapt 100 300 1000 3000 --tune-tau 16 64 256 1024 --tune-tau2 256 1024 4096
```

Three unsupervised variants start from the frozen B1 (all use only unlabeled noisy clips from the validation split):

* **v1** per-class MAP-EM of the class means and covariances (priors fixed, relevance tau1). First design; on real data it
  degraded accuracy (results kept and reported).
* **v2** one shared per-dimension shift and scale for all classes (2d parameters, shrinkage kappa = 20).
* **v2+EM** v2 followed by a strongly anchored MAP-EM (relevance tau2).

tau1 / tau2 are chosen on the validation split only (the printout also scores "no adaptation", so "adaptation does not help"
is a possible outcome). Chow rule: reject when the maximum posterior is below 1 - lambda_r. The fixed-model rows in
`per_run_adapt.csv` reproduce B1 exactly (same noisy test audio); `reject_curve.csv` gives coverage, accuracy on accepted
clips and expected risk per lambda_r.

## Tests

```
python -m pytest -q tests
```

The tests run on a small **synthetic** dataset (`scripts/make_fake_data.py`) and check the code,
not the science: MFCC cross-check, exact SNR of the mixer, noise-pool disjointness, speaker
leakage detector, Gaussian Bayes against the analytic Bayes error, Chow-rule behaviour, and an
end-to-end smoke run. Numbers from synthetic data must never be reported.

## Protocol (fixed before any test result was seen)

| Item | Decision |
|---|---|
| Splits | Official `validation_list.txt` / `testing_list.txt`; the rest is train. Speaker disjointness asserted. |
| Classes | 12: silence, unknown, yes, no, up, down, left, right, on, off, stop, go. |
| Class balance | *unknown* keeps its natural frequency (not re-balanced). |
| Silence | 1-s cuts of `_background_noise_`, time-split 80/10/10 across train/val/test, random volume in [0, 1], count = mean keyword class count per split. |
| Features | 13 MFCC + delta + delta-delta (30 ms window, 10 ms hop); means over 4 time segments + whole-clip std (195-d), standardised, PCA 80 with whitening (chosen on validation). |
| Noise pools | MUSAN files split by file 70/10/20 into train/val/test pools (fixed seed). Test noise never appears in training or validation. |
| Test conditions | Clean; babble / environmental / white at 20, 10, 0 dB. |
| Tuning | Validation split only: B1 pooling / PCA / eps (eps frozen at 0.01), B1-aug eps, tau1 / tau2 of the adaptation variants. TC-ResNet8 settings are fixed a priori (best epoch on validation). kappa = 20 is fixed. The test split is never used for tuning. |
| Seeds | 5. A seed fixes the test-noise realisation. Gaussian Bayes is deterministic given the data, so its std reflects test-noise variability only. |
| Metrics | Macro-F1 (primary: *unknown* dominates accuracy) and accuracy. Under 0-1 loss, expected risk = 1 - accuracy. |

## Baselines in this table

* **B3** majority class (floor).
* **B1** Gaussian Bayes, clean-trained (the classical course method; the "fixed clean-trained model" of RQ1).
* **B1-aug** the same classifier trained on clean + one noisy copy of every training clip (noise-augmentation baseline).
* **B2** TC-ResNet8 (Choi et al., 2019), own PyTorch reimplementation (about 65k parameters), trained on clean data;
  **B2-aug** the same network trained on clean + one noisy copy of every training clip.

## Provenance

All code in `src/kwsba` is original to this project (NumPy/SciPy/scikit-learn; no code copied from
other repositories). `librosa` is used only inside one unit test as an independent check of the
mel filterbank.
