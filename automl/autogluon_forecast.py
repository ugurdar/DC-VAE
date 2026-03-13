"""
AutoGluon TimeSeries - TELCO Multivariate Time Series Forecasting
=================================================================
Two modes:
  A) single: One TS as target, others as known covariates
  B) multi:  All 12 TS forecasted independently (item_id based)

Models: SeasonalNaive, Naive, ETS, Theta, RecursiveTabular, DirectTabular, + Ensemble

Usage:
    python autogluon_forecast.py --target TS1 --mode single
    python autogluon_forecast.py --mode multi --horizon 288
"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError as exc:
    raise SystemExit(
        "autogluon.timeseries not found. Install:\n  pip install autogluon.timeseries\n"
    ) from exc

# =====================================================================
# Directory setup
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABEL_DIR   = PROJECT_DIR / "TELCO_labels"
RESULTS_DIR = BASE_DIR / "results" / "autogluon"
PLOTS_DIR   = RESULTS_DIR / "plots"
MODELS_DIR  = BASE_DIR / "models" / "autogluon"

# =====================================================================
# Control Panel
# =====================================================================
CONTROL = {
    "target": "TS1",
    "mode": "single",            # "single" or "multi"
    "prediction_length": 288,    # 288 x 5min = 1 day
    "freq": "5min",
    "eval_metric": "MASE",
    "time_limit": 300,
    "presets": "medium_quality",
    "num_val_windows": 2,
    "enable_ensemble": True,
    "refit_full": False,
    "random_seed": 42,
    "quantile_levels": [0.1, 0.25, 0.5, 0.75, 0.9],
    # CPU-friendly models (add Chronos2/TFT if GPU available)
    "hyperparameters": {
        "SeasonalNaive": {},
        "Naive": {},
        "ETS": {},
        "Theta": {},
        "RecursiveTabular": {},
        "DirectTabular": {},
    },
    "verbosity": 2,
}


# =====================================================================
# Data loading & conversion
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    return train, val, test


def to_single_series_tsdf(df: pd.DataFrame, target: str, covariate_cols: list[str]):
    out = df[[target] + covariate_cols].copy()
    out = out.rename(columns={target: "target"})
    out["item_id"] = "TELCO"
    out = out.reset_index().rename(columns={"time": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return TimeSeriesDataFrame.from_data_frame(out, id_column="item_id", timestamp_column="timestamp")


def to_multi_series_tsdf(df: pd.DataFrame):
    ts_cols = sorted([c for c in df.columns if c.startswith("TS")])
    records = []
    for col in ts_cols:
        temp = df[[col]].copy().rename(columns={col: "target"})
        temp["item_id"] = col
        temp = temp.reset_index().rename(columns={"time": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        records.append(temp)
    long_df = pd.concat(records, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id", timestamp_column="timestamp")


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# =====================================================================
# Plot: Single series forecast
# =====================================================================
def plot_single_forecast(train_tail, test_series, predictions, target, out_dir):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=1.5, label="Train (tail)")
    ax.plot(test_series.index, test_series.values, color="forestgreen", lw=2, label="Actual (test)")

    pred_idx = predictions.index.get_level_values("timestamp") if "timestamp" in predictions.index.names else predictions.index
    ax.plot(pred_idx, predictions["mean"].values, color="crimson", lw=2, ls="--", label="Forecast (mean)")

    if "0.1" in predictions.columns and "0.9" in predictions.columns:
        ax.fill_between(pred_idx, predictions["0.1"].values, predictions["0.9"].values,
                        alpha=0.12, color="crimson", label="80% interval")
    if "0.25" in predictions.columns and "0.75" in predictions.columns:
        ax.fill_between(pred_idx, predictions["0.25"].values, predictions["0.75"].values,
                        alpha=0.22, color="crimson", label="50% interval")

    ax.set_title(f"TELCO {target} - AutoGluon Forecast", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(target)
    ax.set_xlabel("Time")
    plt.tight_layout()
    fig.savefig(out_dir / f"forecast_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot: Multi-series forecast (ALL TS in one figure)
# =====================================================================
def plot_multi_forecast(full_df, predictions, prediction_length, out_dir, history_length=500):
    if not HAS_MPL:
        return
    ts_cols = sorted(predictions.item_ids)
    n = len(ts_cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), squeeze=False)

    for i, item_id in enumerate(ts_cols):
        ax = axes[i, 0]
        actual = full_df[item_id]
        train_part = actual.iloc[-(history_length + prediction_length):-prediction_length]
        test_part  = actual.iloc[-prediction_length:]

        ax.plot(train_part.index, train_part.values, color="steelblue", lw=1.2, label="Train")
        ax.plot(test_part.index, test_part.values, color="forestgreen", lw=2, label="Actual")

        item_preds = predictions.loc[item_id]
        pred_idx = item_preds.index
        ax.plot(pred_idx, item_preds["mean"].values, color="crimson", lw=2, ls="--", label="Forecast")

        if "0.1" in item_preds.columns and "0.9" in item_preds.columns:
            ax.fill_between(pred_idx, item_preds["0.1"].values, item_preds["0.9"].values,
                            alpha=0.12, color="crimson")
        if "0.25" in item_preds.columns and "0.75" in item_preds.columns:
            ax.fill_between(pred_idx, item_preds["0.25"].values, item_preds["0.75"].values,
                            alpha=0.22, color="crimson")

        ax.axvspan(test_part.index[0], test_part.index[-1], alpha=0.05, color="forestgreen")
        ax.set_title(item_id, fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(item_id)

    plt.suptitle("TELCO - AutoGluon Multi-Series Forecast (Train / Actual / Forecast)",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(out_dir / "forecast_multi_all.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot: Per-series individual forecast (larger, with quantiles)
# =====================================================================
def plot_individual_series(full_df, predictions, prediction_length, out_dir, history_length=800):
    if not HAS_MPL:
        return
    series_dir = out_dir / "per_series"
    series_dir.mkdir(exist_ok=True)

    for item_id in sorted(predictions.item_ids):
        actual = full_df[item_id]
        train_part = actual.iloc[-(history_length + prediction_length):-prediction_length]
        test_part  = actual.iloc[-prediction_length:]
        item_preds = predictions.loc[item_id]

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(train_part.index, train_part.values, color="steelblue", lw=1.5, label="Train (tail)")
        ax.plot(test_part.index, test_part.values, color="forestgreen", lw=2, label="Actual (test)")
        ax.plot(item_preds.index, item_preds["mean"].values, color="crimson", lw=2, ls="--", label="Forecast")

        if "0.1" in item_preds.columns and "0.9" in item_preds.columns:
            ax.fill_between(item_preds.index, item_preds["0.1"].values, item_preds["0.9"].values,
                            alpha=0.12, color="crimson", label="80% interval")
        if "0.25" in item_preds.columns and "0.75" in item_preds.columns:
            ax.fill_between(item_preds.index, item_preds["0.25"].values, item_preds["0.75"].values,
                            alpha=0.22, color="crimson", label="50% interval")

        ax.axvspan(test_part.index[0], test_part.index[-1], alpha=0.05, color="forestgreen")
        ax.set_title(f"TELCO {item_id} - Forecast vs Actual", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(item_id)
        ax.set_xlabel("Time")
        plt.tight_layout()
        fig.savefig(series_dir / f"forecast_{item_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Individual series plots saved to {series_dir}")


# =====================================================================
# Plot: Model comparison (overlay all models on test period)
# =====================================================================
def plot_model_comparison(ts_df, predictor, leaderboard, out_dir,
                          known_covariates=None, full_data=None, target_col=None,
                          history_length=500, mode="single"):
    if not HAS_MPL:
        return
    prediction_length = predictor.prediction_length
    sample_item = ts_df.item_ids[0]

    if full_data is not None and target_col is not None:
        actual_vals = full_data[target_col].values
        actual_idx = full_data.index
    else:
        item_df = ts_df.loc[sample_item]
        actual_vals = item_df["target"].values
        actual_idx = item_df.index

    n_hist = min(history_length, len(actual_vals) - prediction_length)
    hist_ts   = actual_idx[-(n_hist + prediction_length):-prediction_length]
    hist_vals = actual_vals[-(n_hist + prediction_length):-prediction_length]
    test_ts   = actual_idx[-prediction_length:]
    test_vals = actual_vals[-prediction_length:]

    model_names = leaderboard["model"].tolist()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    predict_kwargs = {"known_covariates": known_covariates} if known_covariates is not None else {}

    # --- 1) All models overlay on single plot ---
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(hist_ts, hist_vals, color="steelblue", lw=1.5, label="Train (tail)", zorder=3)
    ax.plot(test_ts, test_vals, color="forestgreen", lw=2.5, label="Actual (test)", zorder=10)
    ax.axvspan(test_ts[0], test_ts[-1], alpha=0.06, color="forestgreen")

    for ci, model_name in enumerate(model_names[:8]):
        try:
            preds = predictor.predict(ts_df, model=model_name, **predict_kwargs)
            item_preds = preds.loc[sample_item]
            score = leaderboard.loc[leaderboard["model"] == model_name, "score_test"].values
            score_str = f" ({score[0]:.3f})" if len(score) > 0 else ""
            ax.plot(item_preds.index, item_preds["mean"].values, ls="--", lw=1.5,
                    color=colors[ci % len(colors)], label=f"{model_name}{score_str}", alpha=0.85)
        except Exception:
            pass

    item_label = target_col if target_col else sample_item
    ax.set_title(f"All Models Comparison - {item_label} (Train / Actual / Forecast)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(item_label)
    ax.set_xlabel("Time")
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2) Per-model subplots (each model gets its own panel) ---
    usable_models = []
    for m in model_names[:6]:
        try:
            preds = predictor.predict(ts_df, model=m, **predict_kwargs)
            usable_models.append((m, preds))
        except Exception:
            pass

    if len(usable_models) > 0:
        n_models = len(usable_models)
        fig, axes = plt.subplots(n_models, 1, figsize=(18, 4 * n_models), squeeze=False)
        for i, (model_name, preds) in enumerate(usable_models):
            ax = axes[i, 0]
            ax.plot(hist_ts, hist_vals, color="steelblue", lw=1.2, label="Train (tail)")
            ax.plot(test_ts, test_vals, color="forestgreen", lw=2, label="Actual (test)")
            item_preds = preds.loc[sample_item]
            ax.plot(item_preds.index, item_preds["mean"].values, color="crimson",
                    lw=2, ls="--", label=f"Forecast ({model_name})")
            if "0.1" in item_preds.columns and "0.9" in item_preds.columns:
                ax.fill_between(item_preds.index,
                                item_preds["0.1"].values, item_preds["0.9"].values,
                                alpha=0.12, color="crimson", label="80% interval")
            ax.axvspan(test_ts[0], test_ts[-1], alpha=0.05, color="forestgreen")
            ax.set_title(f"{model_name} - {item_label}", fontsize=11, fontweight="bold")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel(item_label)

        plt.suptitle(f"Per-Model Forecasts (Train / Actual / Forecast)",
                     fontsize=13, fontweight="bold", y=1.005)
        plt.tight_layout()
        fig.savefig(out_dir / "per_model_forecasts.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# Plot: Ensemble weight analysis
# =====================================================================
def plot_ensemble_weights(predictor, out_dir):
    """Visualize how much each model contributes to the WeightedEnsemble."""
    if not HAS_MPL:
        return
    try:
        info = predictor.info()
        model_info = info.get("model_info", {})
        ens_info = model_info.get("WeightedEnsemble", {})
        weights = ens_info.get("model_weights", None)
        if weights is None:
            return

        models = list(weights.keys())
        vals = list(weights.values())

        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5)))
        bar_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        ax.barh(models, vals, color=bar_colors, edgecolor="white")
        ax.set_xlabel("Ensemble Weight")
        ax.set_title("WeightedEnsemble - Model Contributions", fontweight="bold")
        ax.grid(True, alpha=0.25, axis="x")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "ensemble_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Ensemble weights plot saved")
    except Exception as e:
        print(f"    [WARN] Ensemble weight plot failed: {e}")


# =====================================================================
# Plot: Metrics summary bar chart (multi mode)
# =====================================================================
def plot_metrics_summary(metrics_df: pd.DataFrame, out_dir: Path):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(["MAE", "RMSE", "R2"]):
        ax = axes[i]
        vals = metrics_df[metric].sort_values(ascending=(metric != "R2"))
        colors = ["crimson" if v < 0.8 and metric == "R2" else "steelblue" for v in vals]
        ax.barh(vals.index, vals.values, color=colors, alpha=0.85)
        ax.set_xlabel(metric)
        ax.set_title(metric, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="x")
    fig.suptitle("Per-Series Forecast Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# XAI: Residual diagnostics
# =====================================================================
def plot_residuals_ag(all_residuals: dict, ts_cols: list, out_dir: Path):
    """Histogram + time-plot of residuals for each series."""
    if not HAS_MPL:
        return
    n = len(ts_cols)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n), squeeze=False)

    for i, col in enumerate(ts_cols):
        resid = all_residuals[col]
        # Histogram
        ax_h = axes[i, 0]
        ax_h.hist(resid, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax_h.axvline(0, color="crimson", lw=1.5, ls="--")
        ax_h.set_title(f"{col} - Residual Distribution", fontsize=10, fontweight="bold")
        ax_h.set_xlabel("Residual (actual - predicted)")
        ax_h.set_ylabel("Count")
        ax_h.grid(True, alpha=0.25)
        # Time series of residuals
        ax_t = axes[i, 1]
        ax_t.plot(resid, color="steelblue", lw=0.8, alpha=0.8)
        ax_t.axhline(0, color="crimson", lw=1.5, ls="--")
        ax_t.set_title(f"{col} - Residuals Over Forecast Steps", fontsize=10, fontweight="bold")
        ax_t.set_xlabel("Step")
        ax_t.set_ylabel("Residual")
        ax_t.grid(True, alpha=0.25)

    plt.suptitle("Residual Diagnostics", fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(out_dir / "residual_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Residual diagnostics saved")


# =====================================================================
# XAI: Error by hour of day
# =====================================================================
def plot_error_by_hour(all_residuals: dict, test_index, ts_cols: list, out_dir: Path):
    """Boxplot of absolute error grouped by hour of day."""
    if not HAS_MPL:
        return
    hours = test_index.hour
    n = len(ts_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)

    for i, col in enumerate(ts_cols):
        ax = axes[i, 0]
        abs_err = np.abs(all_residuals[col])
        err_df = pd.DataFrame({"hour": hours[:len(abs_err)], "abs_error": abs_err})
        err_df.boxplot(column="abs_error", by="hour", ax=ax, grid=False,
                       boxprops=dict(color="steelblue"),
                       medianprops=dict(color="crimson", lw=2))
        ax.set_title(f"{col}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("|Error|")
        ax.grid(True, alpha=0.25)

    plt.suptitle("Absolute Forecast Error by Hour of Day", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "error_by_hour.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Error-by-hour plot saved")


# =====================================================================
# XAI: Interval coverage
# =====================================================================
def plot_interval_coverage(predictions, full_data, prediction_length, ts_cols, out_dir: Path):
    """Check whether quantile intervals contain actual values at expected rates."""
    if not HAS_MPL:
        return
    intervals = [
        ("0.1", "0.9", 0.80, "80%"),
        ("0.25", "0.75", 0.50, "50%"),
    ]
    results = []
    for item_id in ts_cols:
        actual = full_data[item_id].iloc[-prediction_length:].values
        item_preds = predictions.loc[item_id]
        row = {"series": item_id}
        for lo, hi, expected, label in intervals:
            if lo in item_preds.columns and hi in item_preds.columns:
                lo_vals = item_preds[lo].values[:len(actual)]
                hi_vals = item_preds[hi].values[:len(actual)]
                covered = np.mean((actual >= lo_vals) & (actual <= hi_vals)) * 100
                row[f"coverage_{label}"] = covered
                row[f"expected_{label}"] = expected * 100
        results.append(row)

    cov_df = pd.DataFrame(results).set_index("series")
    cov_df.to_csv(out_dir / "interval_coverage.csv")

    fig, ax = plt.subplots(figsize=(12, max(4, len(ts_cols) * 0.5)))
    x = np.arange(len(ts_cols))
    width = 0.35
    if "coverage_80%" in cov_df.columns:
        ax.barh(x - width / 2, cov_df["coverage_80%"], width, label="Actual 80% coverage",
                color="steelblue", alpha=0.8)
        ax.axvline(80, color="steelblue", ls="--", lw=1.5, label="Expected 80%")
    if "coverage_50%" in cov_df.columns:
        ax.barh(x + width / 2, cov_df["coverage_50%"], width, label="Actual 50% coverage",
                color="coral", alpha=0.8)
        ax.axvline(50, color="coral", ls="--", lw=1.5, label="Expected 50%")
    ax.set_yticks(x)
    ax.set_yticklabels(ts_cols)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Prediction Interval Coverage (actual vs expected)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / "interval_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Interval coverage plot saved")


# =====================================================================
# XAI: Cross-series error correlation
# =====================================================================
def plot_cross_series_error_corr(all_residuals: dict, ts_cols: list, out_dir: Path):
    """Heatmap of correlation between forecast errors across series."""
    if not HAS_MPL:
        return
    err_df = pd.DataFrame(all_residuals)[ts_cols]
    corr = err_df.corr()
    corr.to_csv(out_dir / "error_correlation.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ts_cols)))
    ax.set_yticks(range(len(ts_cols)))
    ax.set_xticklabels(ts_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ts_cols, fontsize=9)
    for r in range(len(ts_cols)):
        for c in range(len(ts_cols)):
            ax.text(c, r, f"{corr.values[r, c]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[r, c]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("Cross-Series Forecast Error Correlation", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "error_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Error correlation heatmap saved")


# =====================================================================
# XAI: Permutation-based covariate importance (single mode)
# =====================================================================
def plot_covariate_importance(predictor, train_tsdf, full_data, target, covariate_cols,
                              prediction_length, known_cov_builder, out_dir: Path,
                              n_repeats: int = 3):
    """Shuffle each covariate's future values and measure MAE increase."""
    if not HAS_MPL:
        return
    test_actual = full_data[target].iloc[-prediction_length:].values

    # Baseline MAE
    base_cov = known_cov_builder(covariate_cols)
    base_preds = predictor.predict(train_tsdf, known_covariates=base_cov)
    base_mae = np.mean(np.abs(test_actual - base_preds.loc["TELCO"]["mean"].values[:len(test_actual)]))

    importance = {}
    for col in covariate_cols:
        delta_list = []
        for _ in range(n_repeats):
            shuffled_data = full_data.copy()
            shuffled_data[col] = np.random.permutation(shuffled_data[col].values)
            shuf_cov = known_cov_builder(covariate_cols, override_data=shuffled_data)
            shuf_preds = predictor.predict(train_tsdf, known_covariates=shuf_cov)
            shuf_mae = np.mean(np.abs(test_actual - shuf_preds.loc["TELCO"]["mean"].values[:len(test_actual)]))
            delta_list.append(shuf_mae - base_mae)
        importance[col] = np.mean(delta_list)

    imp_df = pd.DataFrame(sorted(importance.items(), key=lambda x: x[1], reverse=True),
                          columns=["covariate", "MAE_increase"])
    imp_df.to_csv(out_dir / "covariate_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(covariate_cols) * 0.4)))
    imp_df_sorted = imp_df.sort_values("MAE_increase")
    colors = ["crimson" if v > 0 else "steelblue" for v in imp_df_sorted["MAE_increase"]]
    ax.barh(imp_df_sorted["covariate"], imp_df_sorted["MAE_increase"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("MAE Increase (higher = more important)")
    ax.set_title("Permutation-based Covariate Importance", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / "covariate_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Covariate importance plot saved")
    return imp_df


# =====================================================================
# XAI: Partial Dependence Profiles (PDP) - multi-model overlay
# =====================================================================
def plot_pdp_autogluon(predictor, train_tsdf, full_data, target, covariate_cols,
                       prediction_length, known_cov_builder, leaderboard,
                       out_dir: Path, top_n_covariates: int = 6, grid_size: int = 20):
    """PDP for each covariate across multiple models on the same plot.

    For each covariate:
      - Create a grid of values from its marginal distribution (percentiles)
      - For each grid value, set ALL future timesteps of that covariate to that value
      - Predict with each model, average the mean forecast over the horizon
      - Plot: x = covariate value, y = avg predicted target, one line per model
    """
    if not HAS_MPL:
        return

    # Select top covariates (use importance if available, else first N)
    imp_path = out_dir / "covariate_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        top_covs = imp_df.sort_values("MAE_increase", ascending=False)["covariate"].tolist()
        top_covs = [c for c in top_covs if c in covariate_cols][:top_n_covariates]
    else:
        top_covs = covariate_cols[:top_n_covariates]

    # Models to include (skip WeightedEnsemble for clarity, but keep if few models)
    model_names = leaderboard["model"].tolist()
    # Filter to models that support known_covariates prediction
    usable_models = []
    base_cov = known_cov_builder(covariate_cols)
    for m in model_names[:6]:
        try:
            _ = predictor.predict(train_tsdf, model=m, known_covariates=base_cov)
            usable_models.append(m)
        except Exception:
            pass
    if not usable_models:
        print("    [WARN] No models usable for PDP, skipping")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---------- Per-covariate PDP ----------
    n_covs = len(top_covs)
    fig, axes = plt.subplots(n_covs, 1, figsize=(14, 4.5 * n_covs), squeeze=False)
    fig.suptitle(f"Partial Dependence Profiles — Target: {target}\n"
                 f"(each covariate varied, others held at actual values)",
                 fontsize=14, fontweight="bold", y=1.005)

    for ci, cov in enumerate(top_covs):
        ax = axes[ci, 0]
        print(f"    PDP for {cov}...")

        # Grid: percentiles of the covariate's training distribution
        cov_train_vals = full_data[cov].iloc[:-prediction_length].values
        grid_vals = np.percentile(cov_train_vals,
                                  np.linspace(2, 98, grid_size))

        for mi, model_name in enumerate(usable_models):
            avg_preds = []
            for gv in grid_vals:
                # Override this covariate's future values with constant gv
                modified_data = full_data.copy()
                modified_data.iloc[-prediction_length:,
                                   modified_data.columns.get_loc(cov)] = gv
                mod_cov = known_cov_builder(covariate_cols, override_data=modified_data)
                preds = predictor.predict(train_tsdf, model=model_name,
                                          known_covariates=mod_cov)
                avg_preds.append(preds.loc["TELCO"]["mean"].values.mean())

            ax.plot(grid_vals, avg_preds, lw=2, marker="o", markersize=3,
                    color=colors[mi % len(colors)], label=model_name, alpha=0.85)

        ax.set_xlabel(f"{cov} value", fontsize=11)
        ax.set_ylabel(f"Avg predicted {target}", fontsize=11)
        ax.set_title(f"PDP: {target} ~ {cov}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_dir / "pdp_multi_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Comparison: all covariates on one plot per model ----------
    for mi, model_name in enumerate(usable_models):
        fig, ax = plt.subplots(figsize=(14, 6))
        for ci, cov in enumerate(top_covs):
            cov_train_vals = full_data[cov].iloc[:-prediction_length].values
            grid_vals = np.percentile(cov_train_vals,
                                      np.linspace(2, 98, grid_size))
            # Normalize grid to [0,1] for overlay
            gv_norm = (grid_vals - grid_vals.min()) / (grid_vals.max() - grid_vals.min() + 1e-12)

            avg_preds = []
            for gv in grid_vals:
                modified_data = full_data.copy()
                modified_data.iloc[-prediction_length:,
                                   modified_data.columns.get_loc(cov)] = gv
                mod_cov = known_cov_builder(covariate_cols, override_data=modified_data)
                preds = predictor.predict(train_tsdf, model=model_name,
                                          known_covariates=mod_cov)
                avg_preds.append(preds.loc["TELCO"]["mean"].values.mean())

            ax.plot(gv_norm, avg_preds, lw=2, marker="o", markersize=3,
                    color=colors[ci % len(colors)], label=cov, alpha=0.85)

        ax.set_xlabel("Covariate value (normalized 0-1)", fontsize=11)
        ax.set_ylabel(f"Avg predicted {target}", fontsize=11)
        ax.set_title(f"PDP Comparison — Model: {model_name}", fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"pdp_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    PDP plots saved ({len(top_covs)} covariates x {len(usable_models)} models)")


# =====================================================================
# Report
# =====================================================================
def generate_report(ctrl, metrics, leaderboard_df, elapsed, out_dir):
    target = ctrl["target"] if ctrl["mode"] == "single" else "ALL (multi)"
    lines = [
        f"# TELCO {target} - AutoGluon TimeSeries Forecast Report",
        f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Mode**: {ctrl['mode']}",
        f"**Training time**: {elapsed:.1f} seconds",
        f"**Forecast horizon**: {ctrl['prediction_length']} steps ({ctrl['prediction_length']*5} min)",
        f"\n## Metrics",
    ]
    if isinstance(metrics, dict) and "MAE" in metrics:
        lines += ["| Metric | Value |", "|--------|-------|"]
        for k, v in metrics.items():
            lines.append(f"| {k} | {v:.6f} |" if not np.isnan(v) else f"| {k} | N/A |")
    else:
        lines.append("Per-series metrics in `metrics_*.csv`.")

    lines.append(f"\n## Leaderboard (Top 10)")
    lines.append(leaderboard_df.head(10).to_markdown(index=False))
    lines.append(f"\n## Settings")
    lines.append("```json")
    lines.append(json.dumps(ctrl, indent=2, default=str))
    lines.append("```")
    mode = ctrl["mode"]
    report_path = out_dir / f"REPORT_{mode}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path}")


# =====================================================================
# Single-series mode
# =====================================================================
def run_single(ctrl):
    target = ctrl["target"]
    print(f"\n--- Mode: single | Target: {target} ---")

    print("\n[1/7] Loading data...")
    train_raw, val_raw, test_raw = load_telco_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data  = pd.concat([full_train, test_raw])
    full_data  = full_data[~full_data.index.duplicated(keep="first")].sort_index()

    ts_cols = [c for c in full_data.columns if c.startswith("TS")]
    covariate_cols = [c for c in ts_cols if c != target]
    print(f"  Total data: {len(full_data)} rows, Target: {target}, Covariates: {len(covariate_cols)}")

    print("\n[2/7] Building TimeSeriesDataFrame...")
    tsdf = to_single_series_tsdf(full_data, target, covariate_cols)
    prediction_length = ctrl["prediction_length"]
    train_tsdf = tsdf.slice_by_timestep(None, -prediction_length)

    print(f"\n[3/7] Training predictor (time_limit={ctrl['time_limit']}s)...")
    predictor = TimeSeriesPredictor(
        target="target", prediction_length=prediction_length,
        freq=ctrl["freq"], eval_metric=ctrl["eval_metric"],
        path=str(MODELS_DIR / "single"),
        quantile_levels=ctrl["quantile_levels"],
        verbosity=ctrl["verbosity"],
        known_covariates_names=covariate_cols,
    )
    fit_kwargs = {
        "time_limit": ctrl["time_limit"], "presets": ctrl["presets"],
        "num_val_windows": ctrl["num_val_windows"],
        "enable_ensemble": ctrl["enable_ensemble"],
        "random_seed": ctrl["random_seed"], "verbosity": ctrl["verbosity"],
    }
    if ctrl["hyperparameters"]:
        fit_kwargs["hyperparameters"] = ctrl["hyperparameters"]

    start = time.time()
    predictor.fit(train_tsdf, **fit_kwargs)
    elapsed = time.time() - start
    print(f"  Training complete. Time: {elapsed:.1f}s")

    # Known covariates for forecast
    future_frame = predictor.make_future_data_frame(train_tsdf)
    future_cov_vals = full_data[covariate_cols].iloc[-prediction_length:].values
    for i, col in enumerate(covariate_cols):
        future_frame[col] = future_cov_vals[:, i]
    known_covariates_df = TimeSeriesDataFrame.from_data_frame(
        future_frame, id_column="item_id", timestamp_column="timestamp",
    )

    leaderboard = predictor.leaderboard(train_tsdf, silent=True)
    print(f"\n  Leaderboard:")
    print(leaderboard.head(10).to_string(index=False))
    leaderboard.to_csv(RESULTS_DIR / f"leaderboard_single_{target}.csv", index=False)

    print(f"\n[4/7] Generating forecasts...")
    predictions = predictor.predict(train_tsdf, known_covariates=known_covariates_df)
    predictions.to_csv(RESULTS_DIR / f"forecasts_single_{target}.csv")

    test_actual = full_data[target].iloc[-prediction_length:]
    pred_mean = predictions.loc["TELCO"]["mean"].values[:len(test_actual)]
    residuals = test_actual.values - pred_mean
    metrics = compute_metrics(test_actual.values, pred_mean)
    print(f"\n  Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}" if not np.isnan(v) else f"    {k}: N/A")
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f"metrics_single_{target}.csv", index=False)

    print(f"\n[5/7] Plots...")
    train_tail = full_data[target].iloc[-(500 + prediction_length):-prediction_length]
    test_series = full_data[target].iloc[-prediction_length:]
    plot_single_forecast(train_tail, test_series, predictions.loc["TELCO"], target, PLOTS_DIR)
    plot_model_comparison(train_tsdf, predictor, leaderboard, PLOTS_DIR,
                          known_covariates=known_covariates_df,
                          full_data=full_data, target_col=target, mode="single")
    plot_ensemble_weights(predictor, PLOTS_DIR)

    # ----- XAI -----
    print(f"\n[6/7] XAI Analysis...")
    XAI_DIR = RESULTS_DIR / "xai_single"
    XAI_DIR.mkdir(exist_ok=True)

    # Residual diagnostics (single target treated as one-series dict)
    all_residuals_single = {target: residuals}
    plot_residuals_ag(all_residuals_single, [target], XAI_DIR)

    # Error by hour
    test_index = full_data.index[-prediction_length:]
    plot_error_by_hour(all_residuals_single, test_index, [target], XAI_DIR)

    # Interval coverage (item_id in predictions is "TELCO")
    single_full = pd.DataFrame({"TELCO": full_data[target].values}, index=full_data.index)
    plot_interval_coverage(predictions, single_full, prediction_length, ["TELCO"], XAI_DIR)

    # Permutation-based covariate importance
    def _build_known_cov(cov_cols, override_data=None):
        src = override_data if override_data is not None else full_data
        ff = predictor.make_future_data_frame(train_tsdf)
        cov_vals = src[cov_cols].iloc[-prediction_length:].values
        for ci, c in enumerate(cov_cols):
            ff[c] = cov_vals[:, ci]
        return TimeSeriesDataFrame.from_data_frame(
            ff, id_column="item_id", timestamp_column="timestamp")

    print("    Computing permutation covariate importance (may take a few minutes)...")
    plot_covariate_importance(
        predictor, train_tsdf, full_data, target, covariate_cols,
        prediction_length, _build_known_cov, XAI_DIR, n_repeats=3)

    # Partial Dependence Profiles (multi-model comparison)
    print("    Computing PDP (multi-model, may take several minutes)...")
    plot_pdp_autogluon(
        predictor, train_tsdf, full_data, target, covariate_cols,
        prediction_length, _build_known_cov, leaderboard, XAI_DIR,
        top_n_covariates=6, grid_size=15)

    print(f"\n[7/7] Report...")
    generate_report(ctrl, metrics, leaderboard, elapsed, RESULTS_DIR)


# =====================================================================
# Multi-series mode
# =====================================================================
def run_multi(ctrl):
    print(f"\n--- Mode: multi | All TS ---")

    print("\n[1/5] Loading data...")
    train_raw, val_raw, test_raw = load_telco_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data  = pd.concat([full_train, test_raw])
    full_data  = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    print(f"  Total: {len(full_data)} rows, {len(full_data.columns)} series")

    print("\n[2/5] Building TimeSeriesDataFrame (multi-series)...")
    tsdf = to_multi_series_tsdf(full_data)
    prediction_length = ctrl["prediction_length"]
    train_tsdf = tsdf.slice_by_timestep(None, -prediction_length)
    print(f"  TSDF shape: {tsdf.shape}, Series: {tsdf.num_items}")

    print(f"\n[3/5] Training predictor (time_limit={ctrl['time_limit']}s)...")
    predictor = TimeSeriesPredictor(
        target="target", prediction_length=prediction_length,
        freq=ctrl["freq"], eval_metric=ctrl["eval_metric"],
        path=str(MODELS_DIR / "multi"),
        quantile_levels=ctrl["quantile_levels"],
        verbosity=ctrl["verbosity"],
    )
    fit_kwargs = {
        "time_limit": ctrl["time_limit"], "presets": ctrl["presets"],
        "num_val_windows": ctrl["num_val_windows"],
        "enable_ensemble": ctrl["enable_ensemble"],
        "random_seed": ctrl["random_seed"], "verbosity": ctrl["verbosity"],
    }
    if ctrl["hyperparameters"]:
        fit_kwargs["hyperparameters"] = ctrl["hyperparameters"]

    start = time.time()
    predictor.fit(train_tsdf, **fit_kwargs)
    elapsed = time.time() - start
    print(f"  Training complete. Time: {elapsed:.1f}s")

    leaderboard = predictor.leaderboard(tsdf, silent=True)
    print(f"\n  Leaderboard:")
    print(leaderboard.head(10).to_string(index=False))
    leaderboard.to_csv(RESULTS_DIR / "leaderboard_multi.csv", index=False)

    # IMPORTANT: predict from train_tsdf (not tsdf) so forecasts align with test period
    print(f"\n[4/6] Generating forecasts...")
    predictions = predictor.predict(train_tsdf)
    predictions.to_csv(RESULTS_DIR / "forecasts_multi.csv")

    # Per-series metrics (predictions now align with test period)
    ts_cols = sorted(predictions.item_ids)
    all_metrics = {}
    all_residuals = {}
    for item_id in ts_cols:
        actual = full_data[item_id].iloc[-prediction_length:]
        pred_mean = predictions.loc[item_id]["mean"].values[:len(actual)]
        m = compute_metrics(actual.values, pred_mean)
        all_metrics[item_id] = m
        all_residuals[item_id] = actual.values - pred_mean
        print(f"  {item_id}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(RESULTS_DIR / "metrics_multi.csv")

    print(f"\n[5/6] Plots...")
    # Combined multi-series plot (train + actual + forecast)
    plot_multi_forecast(full_data, predictions, prediction_length, PLOTS_DIR)
    # Individual per-series plots (larger, with quantiles)
    plot_individual_series(full_data, predictions, prediction_length, PLOTS_DIR)
    # Model comparison (use train_tsdf so predictions align with test)
    plot_model_comparison(train_tsdf, predictor, leaderboard, PLOTS_DIR,
                          full_data=full_data, target_col=ts_cols[0], mode="multi")
    # Metrics summary
    plot_metrics_summary(metrics_df, PLOTS_DIR)
    # Ensemble weights
    plot_ensemble_weights(predictor, PLOTS_DIR)

    # ----- XAI -----
    print(f"\n[6/6] XAI Analysis...")
    XAI_DIR = RESULTS_DIR / "xai"
    XAI_DIR.mkdir(exist_ok=True)
    test_index = full_data.index[-prediction_length:]
    plot_residuals_ag(all_residuals, ts_cols, XAI_DIR)
    plot_error_by_hour(all_residuals, test_index, ts_cols, XAI_DIR)
    plot_interval_coverage(predictions, full_data, prediction_length, ts_cols, XAI_DIR)
    plot_cross_series_error_corr(all_residuals, ts_cols, XAI_DIR)

    avg_metrics = {k: metrics_df[k].mean() for k in ["MAE", "RMSE", "MAPE", "R2"]}
    generate_report(ctrl, avg_metrics, leaderboard, elapsed, RESULTS_DIR)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="AutoGluon TimeSeries TELCO Forecast")
    parser.add_argument("--target", type=str, default=CONTROL["target"])
    parser.add_argument("--mode", type=str, default=CONTROL["mode"], choices=["single", "multi"])
    parser.add_argument("--horizon", type=int, default=CONTROL["prediction_length"])
    parser.add_argument("--time_limit", type=int, default=CONTROL["time_limit"])
    parser.add_argument("--presets", type=str, default=CONTROL["presets"],
                        choices=["fast_training", "medium_quality", "high_quality", "best_quality"])
    args = parser.parse_args()

    ctrl = CONTROL.copy()
    ctrl["target"] = args.target
    ctrl["mode"] = args.mode
    ctrl["prediction_length"] = args.horizon
    ctrl["time_limit"] = args.time_limit
    ctrl["presets"] = args.presets

    print("=" * 80)
    print(f"AutoGluon TimeSeries - TELCO Forecast")
    print(f"Mode: {ctrl['mode']} | Target: {ctrl['target']} | Horizon: {ctrl['prediction_length']} steps")
    print("=" * 80)

    for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if ctrl["mode"] == "single":
        run_single(ctrl)
    else:
        run_multi(ctrl)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Models:  {MODELS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
