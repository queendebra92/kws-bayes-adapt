# Raw results behind the Stage 3 paper

| Folder | Run | Paper rows |
|---|---|---|
| `b1/` | B3, B1, B1-aug, 5 seeds, `--eps 0.01` | Table 2: B3, B1, B1-aug |
| `b2_clean/` | B2 TC-ResNet8, clean-trained, 5 seeds | Table 2: B2 |
| `b2_aug/` | B2-aug TC-ResNet8, noise-augmented, 5 seeds | Table 2: B2-aug |
| `adapt_v2/` | v1, v2, v2+EM at m = 100, 300, 1000, 3000; `reject_curve.csv` for the Chow rule | Table 2 ablations and proposed v2; reject-rule results |

`per_run*.csv` has one row per method, condition and seed (accuracy, macro-F1). `run_info*.json` records hyper-parameters, validation selection and software versions. Model checkpoints (`.pt`) are not included.
