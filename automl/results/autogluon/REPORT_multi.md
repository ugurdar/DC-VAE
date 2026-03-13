# TELCO ALL (multi) - AutoGluon TimeSeries Forecast Report

**Date**: 2026-03-13 11:43
**Mode**: multi
**Training time**: 75.5 seconds
**Forecast horizon**: 288 steps (1440 min)

## Metrics
| Metric | Value |
|--------|-------|
| MAE | 0.286191 |
| RMSE | 0.396098 |
| MAPE | 246.517428 |
| R2 | 0.683959 |

## Leaderboard (Top 10)
| model            |   score_test |   score_val |   pred_time_test |   pred_time_val |   fit_time_marginal |   fit_order |
|:-----------------|-------------:|------------:|-----------------:|----------------:|--------------------:|------------:|
| DirectTabular    |    -0.672822 |   -0.59929  |        0.492776  |       0.483754  |            7.76449  |           4 |
| WeightedEnsemble |    -0.706133 |   -0.571466 |        1.21171   |       1.97286   |            0.205248 |           7 |
| Theta            |    -1.08831  |   -1.28662  |        0.259139  |       0.269823  |            0.440833 |           6 |
| SeasonalNaive    |    -1.14106  |   -0.657651 |        0.018013  |       0.791918  |            1.59053  |           1 |
| RecursiveTabular |    -1.16587  |   -0.691475 |        0.440678  |       0.426307  |           56.6643   |           3 |
| Naive            |    -2.2532   |   -2.2939   |        0.0200832 |       0.0185869 |            0.129973 |           2 |
| ETS              |    -9.54719  |   -7.91831  |        1.60872   |       1.82146   |            3.41616  |           5 |

## Settings
```json
{
  "target": "TS1",
  "mode": "multi",
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