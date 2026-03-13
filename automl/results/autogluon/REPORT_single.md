# TELCO TS1 - AutoGluon TimeSeries Forecast Report

**Date**: 2026-03-13 11:53
**Mode**: single
**Training time**: 51.8 seconds
**Forecast horizon**: 288 steps (1440 min)

## Metrics
| Metric | Value |
|--------|-------|
| MAE | 0.074109 |
| RMSE | 0.101416 |
| MAPE | 75.101438 |
| R2 | 0.972546 |

## Leaderboard (Top 10)
| model            |   score_test |   score_val |   pred_time_test |   pred_time_val |   fit_time_marginal |   fit_order |
|:-----------------|-------------:|------------:|-----------------:|----------------:|--------------------:|------------:|
| WeightedEnsemble |    -0.200025 |   -0.185329 |        0.615426  |       1.31813   |           0.0801788 |           7 |
| RecursiveTabular |    -0.213874 |   -0.210662 |        0.275032  |       0.265294  |          22.7354    |           3 |
| DirectTabular    |    -0.227362 |   -0.197608 |        0.205645  |       0.204367  |          20.6303    |           4 |
| SeasonalNaive    |    -0.257159 |   -0.245075 |        0.0166342 |       0.733608  |           1.05113   |           1 |
| Theta            |    -0.921488 |   -1.0343   |        0.11746   |       0.114216  |           1.25392   |           6 |
| Naive            |    -3.08818  |   -3.06895  |        0.0173011 |       0.0172951 |           0.0421679 |           2 |
| ETS              |    -8.55635  |  -13.9271   |        1.80826   |       1.86531   |           1.36908   |           5 |

## Settings
```json
{
  "target": "TS1",
  "mode": "single",
  "prediction_length": 288,
  "freq": "5min",
  "eval_metric": "MASE",
  "time_limit": 300,
  "presets": "medium_quality",
  "num_val_windows": 2,
  "enable_ensemble": true,
  "refit_full": false,
  "random_seed": 42,
  "quantile_levels": [
    0.1,
    0.25,
    0.5,
    0.75,
    0.9
  ],
  "hyperparameters": {
    "SeasonalNaive": {},
    "Naive": {},
    "ETS": {},
    "Theta": {},
    "RecursiveTabular": {},
    "DirectTabular": {}
  },
  "verbosity": 2
}
```