# TELCO TS1 - H2O AutoML Forecast Report

**Date**: 2026-03-13 12:51
**Training time**: 301.2 seconds
**Forecast horizon**: 288 steps (1440 minutes)

## Test Set Metrics
| Metric | Value |
|--------|-------|
| MAE | 0.010432 |
| RMSE | 0.019649 |
| MAPE | 7.640158 |
| R2 | 0.999622 |

## Leaderboard (Top 10)
| model_id                       |        mae |      rmse |         mse |   rmsle |   mean_residual_deviance |
|:-------------------------------|-----------:|----------:|------------:|--------:|-------------------------:|
| GBM_4_AutoML_1_20260313_123742 | 0.00974136 | 0.0244303 | 0.000596837 |     nan |              0.000596837 |
| GBM_1_AutoML_1_20260313_123742 | 0.0104708  | 0.0283207 | 0.00080206  |     nan |              0.00080206  |
| GBM_2_AutoML_1_20260313_123742 | 0.01205    | 0.0256045 | 0.000655588 |     nan |              0.000655588 |
| GBM_3_AutoML_1_20260313_123742 | 0.0121625  | 0.0263457 | 0.000694098 |     nan |              0.000694098 |
| DRF_1_AutoML_1_20260313_123742 | 0.0200215  | 0.0421192 | 0.00177403  |     nan |              0.00177403  |

## Control Panel Settings
```json
{
  "target": "TS1",
  "prediction_length": 288,
  "freq": "5min",
  "target_lags": [
    1,
    2,
    3,
    6,
    12,
    24,
    72,
    144,
    288
  ],
  "rolling_windows": [
    12,
    72,
    288
  ],
  "exog_lags": [
    1,
    12,
    288
  ],
  "exog_rolling_windows": [
    12,
    288
  ],
  "max_runtime_secs": 300,
  "max_models": 30,
  "nfolds": 5,
  "sort_metric": "MAE",
  "seed": 42,
  "diff_lags": [
    1,
    12,
    288
  ],
  "ewm_spans": [
    12,
    72
  ],
  "rashomon_factor": 4.0,
  "pdp_top_n": 10
}
```

## Output Files
- `plots/forecast_TS1.png` - Best model forecast
- `plots/multi_model_forecast_TS1.png` - All models comparison
- `xai/varimp_TS1_*.png` - Variable importance (per model)
- `xai/pdp/` - Partial Dependence Profiles per model type
- `xai/pdp_grid_TS1_*.png` - PDP grids per model type
- `xai/pdp_comparison_TS1.png` - Cross-model PDP overlay
- `xai/shap_TS1_*.png` - SHAP contributions (per model)
- `xai/rashomon/` - Rashomon set analysis
- `xai/residuals_TS1.png` - Residual diagnostics