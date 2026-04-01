# Implet XAI Report - Time Series Forecasting

**Date**: 2026-03-24 14:51
**Attribution method**: residual
**Lambda**: 0.1
**Threshold factor**: 1.0
**Window size**: 24 (occlusion)
**Max implets per series**: 8
**Attribution computation time**: 0.0s

## Method
Adaptation of 'Implet: A Post-hoc Subsequence Explainer for Time Series Models'
(Meng et al., 2025 IEEE ICDMW) for forecasting models.

### Pipeline:
1. **Attribution**: Per-timestep importance via occlusion/residual analysis
2. **Implet Extraction**: Contiguous high-attribution subsequences (Algorithm 1)
3. **Coh-Implet Clustering**: DTW-based clustering of implets (Algorithm 2)
4. **Faithfulness Test**: Ablation comparison (implet vs random)

## Results
Total implets extracted: 96
Optimal clusters (k): 6

## Per-Series Implet Summary
| Series | N Implets | Top Score | Avg Length |
|--------|-----------|-----------|------------|
| TS1 | 8 | 0.4913 | 3 |
| TS10 | 8 | 0.4746 | 3 |
| TS11 | 8 | 0.4744 | 3 |
| TS12 | 8 | 0.4579 | 3 |
| TS2 | 8 | 0.4672 | 3 |
| TS3 | 8 | 0.6749 | 3 |
| TS4 | 8 | 0.5011 | 3 |
| TS5 | 8 | 0.5028 | 3 |
| TS6 | 8 | 0.4573 | 3 |
| TS7 | 8 | 0.4229 | 3 |
| TS8 | 8 | 0.5624 | 3 |
| TS9 | 8 | 0.4829 | 3 |

## Faithfulness Test
Average implet/random ablation ratio: **6.09x**
(>1.0 means implets are more faithful than random subsequences)