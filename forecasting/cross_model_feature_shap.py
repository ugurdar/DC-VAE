"""
Cross-Model Feature SHAP with Anomaly Overlay
===============================================
3-panel comparison across top-k surrogate models with anomaly regions:
  - Top: Actual + model forecasts + anomaly overlay
  - Middle: Per-model SHAP for selected feature + anomaly overlay
  - Bottom: Mean SHAP (line width ∝ disagreement) + anomaly overlay

Usage:
    python forecasting/cross_model_feature_shap.py
    python forecasting/cross_model_feature_shap.py --series TS1 --feature roll_mean_12
    python forecasting/cross_model_feature_shap.py --top_k_models 3
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from anomaly_utils import load_anomaly_labels, get_anomaly_mask, overlay_anomaly_regions

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
SHAP_DIR    = BASE_DIR / "results" / "surrogate_shap"
MODELS_DIR  = BASE_DIR / "models" / "multi"
RESULTS_DIR = BASE_DIR / "results" / "cross_model_feature_shap"

PREDICTION_LENGTH = 288
HISTORY_STEPS = 500


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


def load_shap_data(models=None):
    shap_data = {}
    for csv_path in sorted(SHAP_DIR.glob("shap_values_*.csv")):
        model_name = csv_path.stem.replace("shap_values_", "")
        if models and model_name not in models:
            continue
        shap_data[model_name] = pd.read_csv(csv_path)
    return shap_data


def load_faithfulness():
    path = SHAP_DIR / "surrogate_faithfulness.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def select_top_k_models(shap_data, faith_df, top_k=3):
    if faith_df.empty or "R2" not in faith_df.columns:
        return shap_data
    ranked = faith_df.sort_values("R2", ascending=False)
    top_models = [m for m in ranked.index if m in shap_data][:top_k]
    return {m: shap_data[m] for m in top_models}


def get_top_features(shap_data, series_id, top_n=5):
    meta_cols = {"_series", "_step"}
    all_importance = []
    for model_name, df in shap_data.items():
        series_df = df[df["_series"] == series_id]
        feat_cols = [c for c in series_df.columns if c not in meta_cols]
        all_importance.append(series_df[feat_cols].abs().mean())
    combined = pd.concat(all_importance, axis=1).mean(axis=1)
    return combined.sort_values(ascending=False).head(top_n).index.tolist()


def smooth(values, window=10):
    if len(values) <= window:
        return values
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


def plot_cross_model_feature(shap_data, faith_df, series_id, feature,
                              full_data, predictions, labels_df,
                              out_dir, smooth_window=10):
    """3-panel plot with shared x-axis and anomaly overlay on all panels."""
    # Collect SHAP values per model
    model_shap = {}
    for model_name in sorted(shap_data.keys()):
        df = shap_data[model_name]
        series_df = df[df["_series"] == series_id].sort_values("_step").reset_index(drop=True)
        if feature in series_df.columns:
            model_shap[model_name] = series_df[feature].values

    if len(model_shap) < 2:
        print(f"  [SKIP] {series_id}/{feature}: need >= 2 models")
        return

    min_len = min(len(v) for v in model_shap.values())
    for k in model_shap:
        model_shap[k] = model_shap[k][:min_len]
    n_steps = min_len

    # Time axes
    forecast_time = full_data.index[-PREDICTION_LENGTH:][:n_steps]
    n_hist = min(HISTORY_STEPS, len(full_data) - PREDICTION_LENGTH)
    hist_time = full_data.index[-(n_hist + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    hist_vals = full_data[series_id].values[-(n_hist + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    test_vals = full_data[series_id].values[-PREDICTION_LENGTH:][:n_steps]

    # R² labels
    r2_labels = {}
    for m in model_shap:
        if not faith_df.empty and m in faith_df.index:
            r2_labels[m] = f"{m} (R²={faith_df.loc[m, 'R2']:.3f})"
        else:
            r2_labels[m] = m

    # SHAP stats
    shap_matrix = np.array([model_shap[m] for m in model_shap])
    mean_shap = np.mean(shap_matrix, axis=0)
    std_shap = np.std(shap_matrix, axis=0)
    mean_smoothed = smooth(mean_shap, smooth_window)
    std_smoothed = smooth(std_shap, smooth_window)

    # Line width ∝ disagreement
    std_min, std_max = std_smoothed.min(), std_smoothed.max()
    if std_max - std_min > 1e-10:
        width_norm = (std_smoothed - std_min) / (std_max - std_min)
    else:
        width_norm = np.zeros_like(std_smoothed)
    line_widths = 1.0 + width_norm * 6.0

    # Colors
    model_colors_list = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    model_colors = {m: model_colors_list[i % len(model_colors_list)]
                    for i, m in enumerate(model_shap)}

    date_fmt = mdates.DateFormatter("%m-%d %H:%M")

    # Anomaly masks
    hist_anomaly = get_anomaly_mask(labels_df, series_id, hist_time)
    forecast_anomaly = get_anomaly_mask(labels_df, series_id, forecast_time)

    # ======== Figure: 3 panels, shared x ========
    fig, (ax_forecast, ax_shap, ax_mean) = plt.subplots(
        3, 1, figsize=(18, 13),
        gridspec_kw={"height_ratios": [1.2, 1.2, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.15)

    # Panel 1: Forecast
    ax_forecast.plot(hist_time, hist_vals, color="steelblue", lw=1.2,
                     label="Train (tail)", alpha=0.7)
    ax_forecast.plot(forecast_time, test_vals, color="forestgreen", lw=2,
                     label="Actual (test)")
    for model_name in model_shap:
        if model_name in predictions and series_id in predictions[model_name]:
            pred_vals = predictions[model_name][series_id][:n_steps]
            ax_forecast.plot(forecast_time[:len(pred_vals)], pred_vals,
                             lw=1.8, ls="--", color=model_colors[model_name],
                             label=f"Forecast ({model_name})", alpha=0.85)

    ax_forecast.axvspan(forecast_time[0], forecast_time[-1], alpha=0.06, color="forestgreen")
    ax_forecast.axvline(forecast_time[0], color="gray", lw=1, ls=":")

    # Anomaly overlay on forecast panel
    overlay_anomaly_regions(ax_forecast, hist_time, hist_anomaly, alpha=0.06)
    overlay_anomaly_regions(ax_forecast, forecast_time, forecast_anomaly, alpha=0.12,
                            label="Anomaly")

    ax_forecast.set_ylabel(series_id, fontsize=12)
    ax_forecast.set_title(f"{series_id} — Forecast + Cross-Model SHAP for '{feature}'",
                          fontweight="bold", fontsize=14)
    ax_forecast.legend(fontsize=9, loc="upper left", ncol=2)
    ax_forecast.grid(True, alpha=0.2)

    # Panel 2: Per-model SHAP
    for model_name in model_shap:
        vals = smooth(model_shap[model_name], smooth_window)
        ax_shap.plot(forecast_time, vals, lw=2.0, label=r2_labels[model_name],
                     color=model_colors[model_name], alpha=0.85)

    ax_shap.axvline(forecast_time[0], color="gray", lw=1, ls=":")
    ax_shap.axhline(0, color="gray", lw=0.8, ls="--")
    overlay_anomaly_regions(ax_shap, forecast_time, forecast_anomaly, alpha=0.10)

    ax_shap.set_ylabel("SHAP Value", fontsize=12)
    ax_shap.set_title(f"Feature '{feature}' — SHAP Values per Model",
                      fontweight="bold", fontsize=12)
    ax_shap.legend(fontsize=10, loc="best", framealpha=0.9)
    ax_shap.grid(True, alpha=0.2)

    # Panel 3: Mean SHAP with variable width
    t_mpl = mdates.date2num(forecast_time.to_pydatetime())
    points = np.array([t_mpl, mean_smoothed]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=line_widths[:-1],
                        color="#2c3e50", alpha=0.9, capstyle="round")
    ax_mean.add_collection(lc)

    ax_mean.fill_between(forecast_time, mean_smoothed - std_smoothed,
                         mean_smoothed + std_smoothed,
                         alpha=0.2, color="steelblue", label="Inter-model std")

    ax_mean.axvline(forecast_time[0], color="gray", lw=1, ls=":")
    ax_mean.axhline(0, color="gray", lw=0.8, ls="--")
    overlay_anomaly_regions(ax_mean, forecast_time, forecast_anomaly, alpha=0.10)

    y_margin = max(np.abs(mean_smoothed).max(), std_smoothed.max()) * 0.3
    y_lo = min(mean_smoothed - std_smoothed) - y_margin
    y_hi = max(mean_smoothed + std_smoothed) + y_margin
    ax_mean.set_xlim(forecast_time[0], forecast_time[-1])
    ax_mean.set_ylim(y_lo, y_hi)

    ax_mean.set_ylabel("Mean SHAP Value", fontsize=12)
    ax_mean.set_xlabel("Time", fontsize=12)
    ax_mean.set_title("Model-Averaged SHAP  (thick = high disagreement,  thin = consensus)",
                      fontweight="bold", fontsize=11)
    ax_mean.legend(fontsize=10, loc="best")
    ax_mean.grid(True, alpha=0.2)
    ax_mean.xaxis.set_major_formatter(date_fmt)
    ax_mean.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_mean.tick_params(axis="x", rotation=30, labelsize=9)

    plt.tight_layout()
    safe_feature = feature.replace("/", "_")
    fig.savefig(out_dir / f"cross_model_{series_id}_{safe_feature}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Feature SHAP with Anomaly")
    parser.add_argument("--series", type=str, default=None)
    parser.add_argument("--feature", type=str, default=None)
    parser.add_argument("--top_n_features", type=int, default=5)
    parser.add_argument("--top_k_models", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cross-Model Feature SHAP (with Anomaly)")
    print("=" * 60)

    # Load SHAP data
    print("\nLoading SHAP values...")
    shap_data = load_shap_data(args.models) if args.models else load_shap_data()
    print(f"  Models loaded: {list(shap_data.keys())}")

    faith_df = load_faithfulness()
    if not args.models:
        shap_data = select_top_k_models(shap_data, faith_df, args.top_k_models)
        print(f"  Top {args.top_k_models} by R²: {list(shap_data.keys())}")

    if len(shap_data) < 2:
        print("[ERROR] Need at least 2 models")
        return

    # Load data
    print("\nLoading TELCO data...")
    train, val, test = load_telco_data()
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])

    labels_df = load_anomaly_labels()
    print(f"  Anomaly labels: {len(labels_df)} rows")

    # Load predictor & forecasts
    print("Loading predictor & generating forecasts...")
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))
    train_val = pd.concat([train, val])
    train_val = train_val[~train_val.index.duplicated(keep="first")].sort_index()
    train_tsdf = to_multi_series_tsdf(train_val)

    predictions = {}
    for model_name in shap_data:
        try:
            preds = predictor.predict(train_tsdf, model=model_name)
            predictions[model_name] = {}
            for sid in ts_cols:
                try:
                    predictions[model_name][sid] = \
                        preds.loc[sid]["mean"].values[:PREDICTION_LENGTH]
                except Exception:
                    pass
            print(f"  {model_name}: OK")
        except Exception as e:
            print(f"  {model_name}: FAILED ({e})")

    # Series list
    first_df = next(iter(shap_data.values()))
    all_series = sorted(first_df["_series"].unique())
    series_list = [args.series] if args.series and args.series in all_series else all_series

    print(f"\nSeries: {series_list}")

    for series_id in series_list:
        print(f"\n--- {series_id} ---")
        features = [args.feature] if args.feature else \
            get_top_features(shap_data, series_id, args.top_n_features)
        if not args.feature:
            print(f"  Top features: {features}")

        for feature in features:
            print(f"  Plotting: {feature}")
            plot_cross_model_feature(
                shap_data, faith_df, series_id, feature,
                full_data, predictions, labels_df, RESULTS_DIR,
                smooth_window=args.smooth,
            )

    print(f"\nOutput: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")
    print("Done!")


if __name__ == "__main__":
    main()
