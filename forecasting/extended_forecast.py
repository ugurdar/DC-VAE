"""
Extended Forecasting Pipeline
==============================
Trains AutoGluon TimeSeriesPredictor with a much wider model family
than the base automl/ pipeline, then evaluates with anomaly-aware plots.

Models: Statistical (9) + Tabular ML (2) + Deep Learning (5)

Usage:
    python forecasting/extended_forecast.py
    python forecasting/extended_forecast.py --time_limit 3600 --presets best_quality
    python forecasting/extended_forecast.py --time_limit 600 --presets high_quality
"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from anomaly_utils import load_anomaly_labels, get_anomaly_mask, overlay_anomaly_regions

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
MODELS_DIR  = BASE_DIR / "models" / "multi"

PREDICTION_LENGTH = 288
FREQ = "5min"

# =====================================================================
# Control Panel
# =====================================================================
CONTROL = {
    "prediction_length": PREDICTION_LENGTH,
    "freq": FREQ,
    "eval_metric": "MASE",
    "time_limit": 3600,
    "presets": "best_quality",
    "num_val_windows": 3,
    "enable_ensemble": True,
    "random_seed": 42,
    "quantile_levels": [0.1, 0.25, 0.5, 0.75, 0.9],
    "verbosity": 2,
}

# Wide model family
HYPERPARAMETERS = {
    # --- Statistical ---
    "Naive": {},
    "SeasonalNaive": {},
    "ETS": {},
    "Theta": {},
    "AutoETS": {},
    "AutoARIMA": {"season_length": 12, "max_p": 5, "max_q": 5},  # m=12 (1hr) to avoid OOM
    "DynamicOptimizedTheta": {},
    "CrostonSBA": {},
    "NPTS": {},
    # --- Tabular ML ---
    "RecursiveTabular": {},
    "DirectTabular": {},
    # --- Deep Learning (graceful skip if no GPU/deps) ---
    "TemporalFusionTransformer": {},
    "DeepAR": {},
    "PatchTST": {},
    "SimpleFeedForward": {},
    "WaveNet": {},
}

# Fallback names for older AutoGluon versions
MODEL_NAME_FALLBACKS = {
    "DeepAR": "DeepARMXNet",
    "TemporalFusionTransformer": "TemporalFusionTransformerMXNet",
    "SimpleFeedForward": "SimpleFeedForwardMXNet",
}

# Statistical + tabular only (safe fallback)
SAFE_MODELS = [
    "Naive", "SeasonalNaive", "ETS", "Theta", "AutoETS", "AutoARIMA",
    "DynamicOptimizedTheta", "CrostonSBA", "NPTS",
    "RecursiveTabular", "DirectTabular",
]


# =====================================================================
# Data loading
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    return train, val, test


def to_multi_series_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    ts_cols = sorted([c for c in df.columns if c.startswith("TS")])
    records = []
    for col in ts_cols:
        temp = df[[col]].copy().rename(columns={col: "target"})
        temp["item_id"] = col
        temp = temp.reset_index().rename(columns={"time": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        records.append(temp)
    long_df = pd.concat(records, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id",
                                                timestamp_column="timestamp")


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mask = np.abs(actual) > 1e-10
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# =====================================================================
# Build safe hyperparameters
# =====================================================================
def build_safe_hyperparameters(requested: dict) -> dict:
    """Try each model name; use fallback names for older AutoGluon versions."""
    safe = {}
    for name, params in requested.items():
        safe[name] = params
    return safe


# =====================================================================
# Plots
# =====================================================================
def plot_multi_forecast(full_data, predictions, prediction_length,
                        labels_df, out_dir, history_steps=500):
    """Grid of all series: train tail + actual test + forecast + anomaly overlay."""
    ts_cols = sorted(predictions.item_ids)
    n = len(ts_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for i, item_id in enumerate(ts_cols):
        ax = axes[i // ncols, i % ncols]

        actual_full = full_data[item_id].values
        actual_idx = full_data.index
        n_hist = min(history_steps, len(actual_full) - prediction_length)

        hist_time = actual_idx[-(n_hist + prediction_length):-prediction_length]
        hist_vals = actual_full[-(n_hist + prediction_length):-prediction_length]
        test_time = actual_idx[-prediction_length:]
        test_vals = actual_full[-prediction_length:]

        pred_mean = predictions.loc[item_id]["mean"].values[:prediction_length]
        pred_time = test_time[:len(pred_mean)]

        ax.plot(hist_time, hist_vals, color="steelblue", lw=1, alpha=0.7, label="Train")
        ax.plot(test_time, test_vals, color="forestgreen", lw=1.5, label="Actual")
        ax.plot(pred_time, pred_mean, color="crimson", lw=1.5, ls="--", label="Forecast")

        # Prediction interval
        try:
            q10 = predictions.loc[item_id]["0.1"].values[:prediction_length]
            q90 = predictions.loc[item_id]["0.9"].values[:prediction_length]
            ax.fill_between(pred_time, q10, q90, alpha=0.15, color="crimson")
        except Exception:
            pass

        # Anomaly overlay
        visible_time = actual_idx[-(n_hist + prediction_length):]
        anomaly_mask = get_anomaly_mask(labels_df, item_id, visible_time)
        overlay_anomaly_regions(ax, visible_time, anomaly_mask, alpha=0.10)

        ax.set_title(item_id, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Extended Forecast — All Series (with anomaly regions)",
                 fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "multi_forecast_anomaly.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leaderboard(leaderboard, out_dir):
    """Horizontal bar chart of model scores."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(leaderboard) * 0.5)))
    models = leaderboard["model"].tolist()
    scores = leaderboard["score_val"].tolist()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    ax.barh(models[::-1], scores[::-1], color=colors)
    ax.set_xlabel("Score (higher is better)", fontsize=11)
    ax.set_title("Model Leaderboard", fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.2, axis="x")

    for i, (m, s) in enumerate(zip(models[::-1], scores[::-1])):
        ax.text(s, i, f" {s:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "leaderboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(metrics_df, out_dir):
    """Per-series metrics bar charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, metric in zip(axes.flat, ["MAE", "RMSE", "MAPE", "R2"]):
        if metric not in metrics_df.columns:
            ax.set_visible(False)
            continue
        vals = metrics_df[metric]
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(vals)))
        ax.bar(metrics_df.index, vals, color=colors)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Per-Series Forecast Metrics", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Extended Forecasting Pipeline")
    parser.add_argument("--time_limit", type=int, default=CONTROL["time_limit"])
    parser.add_argument("--presets", type=str, default=CONTROL["presets"])
    parser.add_argument("--num_val_windows", type=int, default=CONTROL["num_val_windows"])
    args = parser.parse_args()

    ctrl = CONTROL.copy()
    ctrl["time_limit"] = args.time_limit
    ctrl["presets"] = args.presets
    ctrl["num_val_windows"] = args.num_val_windows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extended Forecasting Pipeline")
    print("=" * 70)
    print(f"  time_limit: {ctrl['time_limit']}s")
    print(f"  presets: {ctrl['presets']}")
    print(f"  prediction_length: {ctrl['prediction_length']}")
    print(f"  models: {list(HYPERPARAMETERS.keys())}")

    # ----- 1. Load Data -----
    print("\n[1/6] Loading data...")
    train_raw, val_raw, test_raw = load_telco_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test_raw])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  Total: {len(full_data)} rows, {len(ts_cols)} series")

    # Load anomaly labels
    labels_df = load_anomaly_labels()
    print(f"  Anomaly labels loaded: {len(labels_df)} rows")

    # ----- 2. Build TSDF -----
    print("\n[2/6] Building TimeSeriesDataFrame (multi-series)...")
    tsdf = to_multi_series_tsdf(full_data)
    prediction_length = ctrl["prediction_length"]
    train_tsdf = tsdf.slice_by_timestep(None, -prediction_length)
    print(f"  TSDF shape: {tsdf.shape}, Series: {tsdf.num_items}")

    # ----- 3. Train -----
    print(f"\n[3/6] Training predictor (time_limit={ctrl['time_limit']}s, "
          f"presets={ctrl['presets']})...")

    hyperparams = build_safe_hyperparameters(HYPERPARAMETERS)

    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=prediction_length,
        freq=ctrl["freq"],
        eval_metric=ctrl["eval_metric"],
        path=str(MODELS_DIR),
        quantile_levels=ctrl["quantile_levels"],
        verbosity=ctrl["verbosity"],
    )

    fit_kwargs = {
        "time_limit": ctrl["time_limit"],
        "presets": ctrl["presets"],
        "num_val_windows": ctrl["num_val_windows"],
        "enable_ensemble": ctrl["enable_ensemble"],
        "random_seed": ctrl["random_seed"],
        "verbosity": ctrl["verbosity"],
        "hyperparameters": hyperparams,
    }

    start_time = time.time()
    try:
        predictor.fit(train_tsdf, **fit_kwargs)
    except Exception as e:
        print(f"\n  [WARN] Full training failed: {e}")
        print("  Retrying with statistical + tabular models only...")
        safe_hp = {k: v for k, v in hyperparams.items() if k in SAFE_MODELS}
        fit_kwargs["hyperparameters"] = safe_hp
        fit_kwargs["presets"] = "high_quality"
        predictor.fit(train_tsdf, **fit_kwargs)

    elapsed = time.time() - start_time
    print(f"  Training complete. Time: {elapsed:.1f}s")

    # ----- 4. Leaderboard -----
    leaderboard = predictor.leaderboard(tsdf, silent=True)
    print(f"\n  Leaderboard ({len(leaderboard)} models):")
    print(leaderboard.to_string(index=False))
    leaderboard.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)

    # ----- 5. Evaluate -----
    print(f"\n[4/6] Generating forecasts & computing metrics...")
    predictions = predictor.predict(train_tsdf)
    predictions.to_csv(RESULTS_DIR / "forecasts.csv")

    all_metrics = {}
    for item_id in ts_cols:
        actual = full_data[item_id].iloc[-prediction_length:]
        pred_mean = predictions.loc[item_id]["mean"].values[:len(actual)]
        m = compute_metrics(actual.values, pred_mean)
        all_metrics[item_id] = m
        print(f"  {item_id}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv")

    # Per-model metrics
    print(f"\n[5/6] Per-model evaluation...")
    model_names = leaderboard["model"].tolist()
    model_metrics = {}
    for model_name in model_names:
        try:
            preds = predictor.predict(train_tsdf, model=model_name)
            per_series = {}
            for item_id in ts_cols:
                actual = full_data[item_id].iloc[-prediction_length:]
                pred_mean = preds.loc[item_id]["mean"].values[:len(actual)]
                per_series[item_id] = compute_metrics(actual.values, pred_mean)
            avg_metrics = pd.DataFrame(per_series).T.mean()
            model_metrics[model_name] = avg_metrics.to_dict()
            print(f"  {model_name:30s} avg MAE={avg_metrics['MAE']:.4f} "
                  f"R2={avg_metrics['R2']:.4f}")
        except Exception as e:
            print(f"  {model_name:30s} [FAILED] {e}")

    if model_metrics:
        pd.DataFrame(model_metrics).T.to_csv(RESULTS_DIR / "per_model_metrics.csv")

    # ----- 6. Plots -----
    print(f"\n[6/6] Generating plots...")
    plot_multi_forecast(full_data, predictions, prediction_length, labels_df, PLOTS_DIR)
    print("  - Multi-series forecast with anomaly overlay")

    plot_leaderboard(leaderboard, PLOTS_DIR)
    print("  - Leaderboard")

    plot_metrics_summary(metrics_df, PLOTS_DIR)
    print("  - Metrics summary")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Models trained: {len(leaderboard)}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Models saved: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
