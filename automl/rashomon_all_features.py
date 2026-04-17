"""
Rashomon All-Features SHAP Plot
================================
Two-panel plot per series:
  Top:    Forecast context (train tail + actual + model forecasts)
  Bottom: ALL features' mean SHAP (across models) as separate coloured lines,
          each with a fill_between confidence band = inter-model std (Rashomon).
          Anomaly labels from TELCO_labels overlaid as vertical red lines.

Usage:
    python rashomon_all_features.py
    python rashomon_all_features.py --series TS1
    python rashomon_all_features.py --series TS1 --top_n 10
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
import matplotlib.dates as mdates

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
SHAP_DIR    = BASE_DIR / "results" / "autogluon" / "surrogate_shap"
MODELS_DIR  = BASE_DIR / "models" / "autogluon" / "multi"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "rashomon_all_features"

PREDICTION_LENGTH = 288
HISTORY_STEPS = 1500  # ~5 days of history to capture nearby anomalies


# =====================================================================
# Data loaders
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    return train, val, test


def load_anomaly_labels():
    """Load and concatenate anomaly labels for all splits."""
    dfs = []
    for split in ("train", "val", "test"):
        path = LABELS_DIR / f"TELCO_labels_{split}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["time"], index_col="time")
            df.index = df.index.tz_localize(None)
            dfs.append(df)
    if dfs:
        labels = pd.concat(dfs)
        labels = labels[~labels.index.duplicated(keep="first")].sort_index()
        return labels
    return pd.DataFrame()


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


def load_shap_data() -> dict[str, pd.DataFrame]:
    shap_data = {}
    for csv_path in sorted(SHAP_DIR.glob("shap_values_*.csv")):
        model_name = csv_path.stem.replace("shap_values_", "")
        df = pd.read_csv(csv_path)
        shap_data[model_name] = df
    return shap_data


def load_faithfulness() -> pd.DataFrame:
    path = SHAP_DIR / "surrogate_faithfulness.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def select_top_k_models(shap_data: dict, faith_df: pd.DataFrame,
                         top_k: int = 5) -> dict[str, pd.DataFrame]:
    if faith_df.empty or "R2" not in faith_df.columns:
        return shap_data
    ranked = faith_df.sort_values("R2", ascending=False)
    top_models = [m for m in ranked.index if m in shap_data][:top_k]
    return {m: shap_data[m] for m in top_models}


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) <= window:
        return values
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


def get_ranked_features(shap_data: dict, series_id: str) -> list[str]:
    """Rank features by mean |SHAP| across all models."""
    meta_cols = {"_series", "_step"}
    all_importance = []
    for df in shap_data.values():
        sdf = df[df["_series"] == series_id]
        feat_cols = [c for c in sdf.columns if c not in meta_cols]
        all_importance.append(sdf[feat_cols].abs().mean())
    combined = pd.concat(all_importance, axis=1).mean(axis=1)
    return combined.sort_values(ascending=False).index.tolist()


# =====================================================================
# Colour palette for many features
# =====================================================================
FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#ffe119", "#ffd8b1", "#000000",
]


# =====================================================================
# Main plot function
# =====================================================================
def plot_rashomon_all_features(
    shap_data: dict,
    faith_df: pd.DataFrame,
    series_id: str,
    full_data: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    anomaly_labels: pd.DataFrame,
    out_dir: Path,
    top_n: int = 10,
    smooth_window: int = 10,
):
    """
    2-panel plot:
      Top:    Train tail + Actual test + model forecasts
      Bottom: All features' mean SHAP lines with Rashomon CI + anomaly markers
    """
    meta_cols = {"_series", "_step"}

    # Rank features
    ranked_features = get_ranked_features(shap_data, series_id)
    features_to_plot = ranked_features[:top_n]
    print(f"  Top {top_n} features: {features_to_plot}")

    # Build per-feature: mean & std across models
    feature_mean = {}
    feature_std = {}
    n_steps = None

    for feat in features_to_plot:
        model_vals = []
        for model_name, df in shap_data.items():
            sdf = df[df["_series"] == series_id].sort_values("_step").reset_index(drop=True)
            if feat in sdf.columns:
                model_vals.append(sdf[feat].values)
        if len(model_vals) < 2:
            continue
        min_len = min(len(v) for v in model_vals)
        if n_steps is None:
            n_steps = min_len
        else:
            n_steps = min(n_steps, min_len)
        stacked = np.array([v[:n_steps] for v in model_vals])
        feature_mean[feat] = np.mean(stacked, axis=0)
        feature_std[feat] = np.std(stacked, axis=0)

    if not feature_mean:
        print(f"  [SKIP] {series_id}: no features with >= 2 models")
        return

    # Trim all to same n_steps
    for feat in list(feature_mean.keys()):
        feature_mean[feat] = feature_mean[feat][:n_steps]
        feature_std[feat] = feature_std[feat][:n_steps]

    # --- Time axes ---
    forecast_time = full_data.index[-PREDICTION_LENGTH:][:n_steps]
    n_hist = min(HISTORY_STEPS, len(full_data) - PREDICTION_LENGTH)
    hist_time = full_data.index[-(n_hist + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    hist_vals = full_data[series_id].values[-(n_hist + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    test_vals = full_data[series_id].values[-PREDICTION_LENGTH:][:n_steps]

    # --- Anomaly times in forecast window ---
    anomaly_times_in_window = []
    if not anomaly_labels.empty and series_id in anomaly_labels.columns:
        ts_labels = anomaly_labels[series_id]
        # Filter to forecast window
        mask = (ts_labels.index >= forecast_time[0]) & (ts_labels.index <= forecast_time[-1])
        anomaly_ts = ts_labels[mask]
        anomaly_times_in_window = anomaly_ts[anomaly_ts > 0].index.tolist()

    # --- Model colours for forecast panel ---
    model_colors_list = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    model_names = sorted(shap_data.keys())
    model_colors = {m: model_colors_list[i % len(model_colors_list)]
                    for i, m in enumerate(model_names)}

    # R² labels
    r2_labels = {}
    for m in model_names:
        if not faith_df.empty and m in faith_df.index:
            r2_labels[m] = f"{m} (R²={faith_df.loc[m, 'R2']:.3f})"
        else:
            r2_labels[m] = m

    date_fmt = mdates.DateFormatter("%m-%d %H:%M")

    # --- Anomaly times across FULL visible range (history + forecast) ---
    all_visible_start = hist_time[0] if len(hist_time) > 0 else forecast_time[0]
    all_visible_end = forecast_time[-1]
    all_anomaly_times = []
    if not anomaly_labels.empty and series_id in anomaly_labels.columns:
        ts_labels = anomaly_labels[series_id]
        mask = (ts_labels.index >= all_visible_start) & (ts_labels.index <= all_visible_end)
        anom_ts = ts_labels[mask]
        all_anomaly_times = anom_ts[anom_ts > 0].index.tolist()
    print(f"  Anomalies in visible range: {len(all_anomaly_times)}")

    # ======== Figure: 2 panels (independent x-axes) ========
    fig, (ax_forecast, ax_shap) = plt.subplots(
        2, 1, figsize=(20, 13),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    fig.subplots_adjust(hspace=0.25)

    # ---- Panel 1: Forecast context ----
    ax_forecast.plot(hist_time, hist_vals, color="steelblue", lw=1.2,
                     label="Train (tail)", alpha=0.7)
    ax_forecast.plot(forecast_time, test_vals, color="forestgreen", lw=2,
                     label="Actual (test)")

    for model_name in model_names:
        if model_name in predictions and series_id in predictions[model_name]:
            pred_vals = predictions[model_name][series_id][:n_steps]
            ax_forecast.plot(forecast_time[:len(pred_vals)], pred_vals,
                             lw=1.8, ls="--", color=model_colors[model_name],
                             label=f"Forecast ({r2_labels[model_name]})", alpha=0.85)

    ax_forecast.axvspan(forecast_time[0], forecast_time[-1],
                        alpha=0.06, color="forestgreen")
    ax_forecast.axvline(forecast_time[0], color="gray", lw=1, ls=":")

    # Anomaly markers on forecast panel (full visible range)
    for i, at in enumerate(all_anomaly_times):
        ax_forecast.axvline(at, color="red", lw=1.0, alpha=0.6,
                            label="Anomaly" if i == 0 else None)

    ax_forecast.set_ylabel(series_id, fontsize=12)
    ax_forecast.set_title(
        f"{series_id} — Forecast + Rashomon SHAP (All Features, Top {top_n})",
        fontweight="bold", fontsize=14,
    )
    ax_forecast.legend(fontsize=8, loc="upper left", ncol=3)
    ax_forecast.grid(True, alpha=0.2)
    ax_forecast.xaxis.set_major_formatter(date_fmt)
    ax_forecast.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax_forecast.tick_params(axis="x", rotation=30, labelsize=9)

    # ---- Panel 2: All features' SHAP with Rashomon CI ----
    feat_list = list(feature_mean.keys())
    for i, feat in enumerate(feat_list):
        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        mean_s = smooth(feature_mean[feat], smooth_window)
        std_s = smooth(feature_std[feat], smooth_window)

        ax_shap.plot(forecast_time, mean_s, lw=2.0, color=color,
                     label=feat, alpha=0.9)
        ax_shap.fill_between(
            forecast_time,
            mean_s - std_s,
            mean_s + std_s,
            alpha=0.12, color=color,
        )

    ax_shap.axhline(0, color="gray", lw=0.8, ls="--")

    # Anomaly markers on SHAP panel (forecast window only)
    for i, at in enumerate(anomaly_times_in_window):
        ax_shap.axvline(at, color="red", lw=1.0, alpha=0.6,
                         label="Anomaly" if i == 0 else None)

    ax_shap.set_ylabel("SHAP Value (mean across models)", fontsize=12)
    ax_shap.set_xlabel("Time", fontsize=12)
    ax_shap.set_title(
        f"Rashomon SHAP — Each line = feature, shaded band = inter-model uncertainty (±1σ)",
        fontweight="bold", fontsize=12,
    )

    # Legend: features + anomaly marker
    ax_shap.legend(
        fontsize=8, loc="upper left", ncol=3, framealpha=0.9,
        title="Features (band = Rashomon uncertainty)",
        title_fontsize=9,
    )
    ax_shap.grid(True, alpha=0.2)

    # x-axis formatting for SHAP panel
    ax_shap.xaxis.set_major_formatter(date_fmt)
    ax_shap.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_shap.tick_params(axis="x", rotation=30, labelsize=9)

    out_path = out_dir / f"rashomon_all_features_{series_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Rashomon All-Features SHAP")
    parser.add_argument("--series", type=str, default=None,
                        help="Series to analyze (e.g., TS1). Default: all")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Top N features by importance (default: 10)")
    parser.add_argument("--top_k_models", type=int, default=5,
                        help="Use top K models by R² (default: 5 = all)")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Smoothing window size")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Rashomon All-Features SHAP Analysis")
    print("=" * 60)

    # Load SHAP data
    print("\nLoading SHAP values...")
    shap_data = load_shap_data()
    print(f"  Models loaded: {list(shap_data.keys())}")

    faith_df = load_faithfulness()
    shap_data = select_top_k_models(shap_data, faith_df, args.top_k_models)
    print(f"  Selected models: {list(shap_data.keys())}")

    if not faith_df.empty:
        print("\nSurrogate Faithfulness:")
        for model in shap_data:
            if model in faith_df.index:
                print(f"  {model:25s} R²={faith_df.loc[model, 'R2']:.4f}")

    if len(shap_data) < 2:
        print("[ERROR] Need >= 2 models for Rashomon analysis")
        return

    # Load actual data
    print("\nLoading TELCO data...")
    train, val, test = load_telco_data()
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    full_data.index = full_data.index.tz_localize(None)
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])

    # Load anomaly labels
    print("Loading anomaly labels...")
    anomaly_labels = load_anomaly_labels()
    if not anomaly_labels.empty:
        print(f"  Labels shape: {anomaly_labels.shape}")
    else:
        print("  [WARN] No anomaly labels found")

    # Load predictor & forecasts
    print("Loading AutoGluon predictor & generating forecasts...")
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
            print(f"  {model_name}: failed ({e})")

    # Series list
    first_df = next(iter(shap_data.values()))
    all_series = sorted(first_df["_series"].unique())
    if args.series:
        series_list = [s for s in [args.series] if s in all_series]
        if not series_list:
            print(f"[ERROR] Series '{args.series}' not in {all_series}")
            return
    else:
        series_list = all_series

    print(f"\nGenerating plots for: {series_list}")

    for series_id in series_list:
        print(f"\n--- {series_id} ---")
        plot_rashomon_all_features(
            shap_data, faith_df, series_id, full_data, predictions,
            anomaly_labels, RESULTS_DIR,
            top_n=args.top_n, smooth_window=args.smooth,
        )

    print(f"\nOutput: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")
    print("Done!")


if __name__ == "__main__":
    main()
