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

`--eps 0.01` freezes the clean-trained B1 regulariser (chosen on validation before any test result was seen); without it the script re-selects eps from the grid on validation, which picks 0.001 and changes B1 by about 0.001-0.003 macro-F1. B1-aug always tunes its own eps on validation.

```
python run_baselines.py --sc-root D:\data\speech_commands_v0.02 --musan-root D:\data\musan --out results --eps 0.01
```

Outputs in `results/`: `per_run.csv` (every method x condition x seed),
`table2_by_snr.csv`, `table2_by_condition.csv`, `table2_macro_f1.tex` (the table in the
paper), `run_info.json` (counts, tuned values, versions, timings, hardware).
The first run decodes all audio into `cache/` (about 3.4 GB int16); later runs reuse it.

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

## Stage 4: final experiments, ablations, error analysis (one command each)

```
python run_stage4.py --musan-root D:\data\musan --cache-dir cache --b2-clean-models results_tcresnet\models --b2-aug-models results_tcresnet_aug\models --out results_stage4
python scripts\analyze_stage4.py --results results_stage4 --out results_stage4\analysis --boot 10000
```

`run_stage4.py` reuses the Stage 3 protocol (same noisy test audio, same adaptation draws), so the Stage 3 rows are
reproduced exactly. It adds per-clip predictions for every method, a shared calibration set (one temperature per model) so
the Chow rule can be applied to all models, the test-time-adaptation baselines AdaBN / Tent / an AdaKWS-style variant on
B2 and B2-aug (lr and epochs chosen on validation only, `tta_validation.json`), ablations of v2 (shift only, scale only,
kappa) and sensitivity to the number of adaptation clips and the unknown fraction of the adaptation set.
`--no-neural` runs the Gaussian methods only. Finished seeds are skipped on re-run.
`analyze_stage4.py` builds every table and figure of the paper from `results_stage4/` (clip-level paired bootstrap,
confusion matrices, per-class F1, risk-coverage curves, sensitivity plots); nothing is typed by hand. The paper uses `--boot 10000` (the smallest attainable Holm-adjusted p is then 0.009 over 90 tests).
`scripts/make_paper_stage4.py --results results_stage4 --analysis results_stage4\analysis --out paper\figures` writes the paper's table fragments and figures;
`scripts/error_decomp_stage4.py --results results_stage4 --out paper\figures` writes the 0 dB error-decomposition table (Table 3).

### Stage 4 extra: sensitivity to the PCA dimension d and the training-set size n (Gaussian models, a few minutes)

```
python run_sensitivity_dn.py --musan-root D:\data\musan --cache-dir cache --out results_stage4_dn --n-seeds 3
```

Same test audio and adaptation draws as `run_stage4.py`, so the settings d = 80 and n = 100 % reproduce the Stage 4 rows
of B1 and v2 exactly. `scripts/make_paper_stage4.py --dn results_stage4_dn/sens_dn_per_run.csv ...` adds them to Figure 3.

### Stage 4 extra: reject rule versus the number of adaptation clips m (RQ2, Gaussian models, a few minutes)

```
python run_reject_m.py --musan-root D:\data\musan --cache-dir cache --run-info results_stage4\run_info_stage4.json --out results_stage4_rejm --n-seeds 5 --check results_stage4\sensitivity_per_run.csv
python scripts\rejm_stage4.py --csv results_stage4_rejm\reject_m_per_run.csv --out paper\figures
```

Same test audio and nested adaptation sets as the m sweep of `run_stage4.py`; `--check` verifies that the macro-F1 column
reproduces `sensitivity_per_run.csv` exactly. Coverage, expected risk (lambda_r = 0.2, B1's temperature) and the accuracy of
the 30 % most confident clips are reported for m = 0 (B1), 10, 30, 100, 300, 1000, 3000.

## Known issue found in Stage 4 (Stage 3 adapted rows)

Stage 3 `run_adapt.py` drew `max(n_adapt)` = 3000 adaptation clips, **sorted** the indices and then used the first `n`.
The validation clips are stored class by class, so every `n < 3000` used a class-biased subset (the lowest-index
clips). B1, B1-aug, B2, B2-aug and B3 are unaffected and are reproduced exactly by `run_stage4.py`. The adapted rows
(v2, v2+EM) and the Stage 3 curve against `n` changed slightly: at m = 1000 the mean macro-F1 of v2 differs from the
Stage 3 table by 0.006 over all conditions (largest 0.038, babble at 10 dB). `run_adapt.py` now keeps the random order,
and `run_stage4.py` draws the m clips directly; the Stage 4 numbers are the ones to cite.

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
