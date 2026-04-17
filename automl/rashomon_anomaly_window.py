"""
Rashomon SHAP with Anomaly Overlay
===================================
Recomputes surrogate SHAP at a custom forecast origin that covers
anomaly-rich periods, then plots all-features Rashomon + anomaly markers.

Usage:
    python rashomon_anomaly_window.py
    python rashomon_anomaly_window.py --series TS1 --forecast_date 2021-06-16
    python rashomon_anomaly_window.py --series TS1 --top_n 10
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

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
MODELS_DIR  = BASE_DIR / "models" / "autogluon" / "multi"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "rashomon_anomaly"

PREDICTION_LENGTH = 288
FREQ = "5min"

TARGET_LAGS = [1, 2, 3, 6, 12, 24, 72, 144, 288]
ROLLING_WINDOWS = [12, 72, 288]
HISTORY_STEPS = 1500

FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#ffe119", "#ffd8b1", "#000000",
]


# =====================================================================
# Data
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    for df in (train, val, test):
        df.index = df.index.tz_localize(None)
    return train, val, test


def load_anomaly_labels():
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


# =====================================================================
# Feature Engineering
# =====================================================================
def build_features_for_series(values, timestamps, forecast_start_idx,
                               prediction_length, series_id):
    rows = []
    for h in range(prediction_length):
        t = forecast_start_idx + h
        feat = {}
        feat["step"] = h + 1
        feat["step_norm"] = (h + 1) / prediction_length

        for lag in TARGET_LAGS:
            idx = t - lag
            feat[f"lag_{lag}"] = values[idx] if idx >= 0 else np.nan

        for w in ROLLING_WINDOWS:
            start = max(0, t - w)
            wv = values[start:t]
            if len(wv) > 0:
                feat[f"roll_mean_{w}"] = np.mean(wv)
                feat[f"roll_std_{w}"] = np.std(wv)
                feat[f"roll_min_{w}"] = np.min(wv)
                feat[f"roll_max_{w}"] = np.max(wv)
            else:
                for stat in ("mean", "std", "min", "max"):
                    feat[f"roll_{stat}_{w}"] = np.nan

        feat["diff_1"] = (values[t-1] - values[t-2]) if t >= 2 else 0.0
        feat["diff_12"] = (values[t-1] - values[t-12]) if t >= 12 else np.nan
        feat["diff_288"] = (values[t-1] - values[t-288]) if t >= 288 else np.nan

        ts = timestamps[t]
        feat["hour"] = ts.hour
        feat["minute"] = ts.minute
        feat["day_of_week"] = ts.dayofweek
        feat["is_weekend"] = int(ts.dayofweek >= 5)
        feat["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)

        train_vals = values[:forecast_start_idx]
        feat["series_mean"] = np.mean(train_vals)
        feat["series_std"] = np.std(train_vals)
        feat["series_last"] = values[forecast_start_idx - 1]
        feat["series_id_num"] = 0  # Will be overwritten

        rows.append(feat)
    return pd.DataFrame(rows)


def build_all_features(full_data, forecast_start_idx, prediction_length):
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    all_frames = []
    series_map = {s: i for i, s in enumerate(ts_cols)}

    for col in ts_cols:
        values = full_data[col].values
        timestamps = full_data.index
        feat_df = build_features_for_series(
            values, timestamps, forecast_start_idx, prediction_length, col
        )
        feat_df["series_id_num"] = series_map[col]
        feat_df["_series"] = col
        feat_df["_step_idx"] = range(prediction_length)
        all_frames.append(feat_df)

    return pd.concat(all_frames, ignore_index=True)


# =====================================================================
# Smoothing
# =====================================================================
def smooth(values, window=10):
    if len(values) <= window:
        return values
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


# =====================================================================
# Plot
# =====================================================================
def plot_rashomon(shap_data, series_id, full_data, predictions,
                  anomaly_labels, forecast_time, hist_time, hist_vals,
                  test_vals, faith_info, out_dir, top_n=10, smooth_window=10):
    meta_cols = {"_series", "_step"}

    # Rank features by mean |SHAP| across models
    all_importance = []
    for model_name, df in shap_data.items():
        sdf = df[df["_series"] == series_id]
        feat_cols = [c for c in sdf.columns if c not in meta_cols]
        all_importance.append(sdf[feat_cols].abs().mean())
    combined = pd.concat(all_importance, axis=1).mean(axis=1)
    ranked = combined.sort_values(ascending=False)
    features_to_plot = ranked.head(top_n).index.tolist()
    print(f"  Top {top_n} features: {features_to_plot}")

    # Build per-feature mean & std across models
    feature_mean = {}
    feature_std = {}
    n_steps = None

    for feat in features_to_plot:
        model_vals = []
        for df in shap_data.values():
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
        feature_mean[feat] = np.mean(stacked, axis=0)[:n_steps]
        feature_std[feat] = np.std(stacked, axis=0)[:n_steps]

    if not feature_mean:
        print(f"  [SKIP] {series_id}: insufficient data")
        return

    forecast_time = forecast_time[:n_steps]
    test_vals = test_vals[:n_steps]

    # Anomaly times in visible range
    all_visible_start = hist_time[0] if len(hist_time) > 0 else forecast_time[0]
    all_visible_end = forecast_time[-1]
    all_anomaly_times = []
    forecast_anomaly_times = []
    if not anomaly_labels.empty and series_id in anomaly_labels.columns:
        ts_labels = anomaly_labels[series_id]
        mask_all = (ts_labels.index >= all_visible_start) & (ts_labels.index <= all_visible_end)
        all_anomaly_times = ts_labels[mask_all][ts_labels[mask_all] > 0].index.tolist()
        mask_fc = (ts_labels.index >= forecast_time[0]) & (ts_labels.index <= forecast_time[-1])
        forecast_anomaly_times = ts_labels[mask_fc][ts_labels[mask_fc] > 0].index.tolist()

    print(f"  Anomalies: {len(all_anomaly_times)} in history+forecast, "
          f"{len(forecast_anomaly_times)} in forecast window")

    # Model colours
    model_names = sorted(shap_data.keys())
    model_colors_list = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    model_colors = {m: model_colors_list[i % len(model_colors_list)]
                    for i, m in enumerate(model_names)}

    r2_labels = {}
    for m in model_names:
        if m in faith_info:
            r2_labels[m] = f"{m} (R²={faith_info[m]:.3f})"
        else:
            r2_labels[m] = m

    date_fmt = mdates.DateFormatter("%m-%d %H:%M")

    # ======== Figure ========
    fig, (ax_forecast, ax_shap) = plt.subplots(
        2, 1, figsize=(20, 13),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    fig.subplots_adjust(hspace=0.25)

    # ---- Panel 1: Forecast ----
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

    for i, at in enumerate(all_anomaly_times):
        ax_forecast.axvline(at, color="red", lw=1.0, alpha=0.5,
                            label="Anomaly" if i == 0 else None)

    ax_forecast.set_ylabel(series_id, fontsize=12)
    ax_forecast.set_title(
        f"{series_id} — Forecast + Rashomon SHAP (Top {top_n} Features)",
        fontweight="bold", fontsize=14,
    )
    ax_forecast.legend(fontsize=8, loc="upper left", ncol=3)
    ax_forecast.grid(True, alpha=0.2)
    ax_forecast.xaxis.set_major_formatter(date_fmt)
    ax_forecast.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax_forecast.tick_params(axis="x", rotation=30, labelsize=9)

    # ---- Panel 2: All features SHAP + Rashomon CI + Anomalies ----
    feat_list = list(feature_mean.keys())
    for i, feat in enumerate(feat_list):
        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        mean_s = smooth(feature_mean[feat], smooth_window)
        std_s = smooth(feature_std[feat], smooth_window)
        ax_shap.plot(forecast_time, mean_s, lw=2.0, color=color,
                     label=feat, alpha=0.9)
        ax_shap.fill_between(forecast_time, mean_s - std_s, mean_s + std_s,
                             alpha=0.12, color=color)

    ax_shap.axhline(0, color="gray", lw=0.8, ls="--")

    # Anomaly markers on SHAP panel
    for i, at in enumerate(forecast_anomaly_times):
        ax_shap.axvline(at, color="red", lw=1.0, alpha=0.5,
                         label="Anomaly" if i == 0 else None)

    ax_shap.set_ylabel("SHAP Value (mean across models)", fontsize=12)
    ax_shap.set_xlabel("Time", fontsize=12)
    ax_shap.set_title(
        "Rashomon SHAP — Each line = feature, band = inter-model uncertainty (±1σ), "
        "red lines = anomalies",
        fontweight="bold", fontsize=11,
    )
    ax_shap.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.9,
                   title="Features (band = Rashomon uncertainty)", title_fontsize=9)
    ax_shap.grid(True, alpha=0.2)
    ax_shap.xaxis.set_major_formatter(date_fmt)
    ax_shap.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_shap.tick_params(axis="x", rotation=30, labelsize=9)

    out_path = out_dir / f"rashomon_anomaly_{series_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="TS1")
    parser.add_argument("--forecast_date", type=str, default="2021-06-16",
                        help="Date to start forecast window (default: 2021-06-16, 145 anomalies)")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series_id = args.series

    print("=" * 60)
    print(f"Rashomon SHAP — Anomaly Window ({args.forecast_date})")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    train, val, test = load_telco_data()
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])

    anomaly_labels = load_anomaly_labels()
    if series_id in anomaly_labels.columns:
        fc_date = pd.Timestamp(args.forecast_date)
        day_mask = anomaly_labels.index.date == fc_date.date()
        n_anom = anomaly_labels.loc[day_mask, series_id].sum()
        print(f"  Anomalies on {args.forecast_date} for {series_id}: {int(n_anom)}")

    # Find forecast start index
    forecast_start_ts = pd.Timestamp(args.forecast_date)
    idx_mask = full_data.index >= forecast_start_ts
    if not idx_mask.any():
        print(f"[ERROR] Date {args.forecast_date} not in data range")
        return
    forecast_start_idx = idx_mask.argmax()
    actual_pl = min(PREDICTION_LENGTH, len(full_data) - forecast_start_idx)
    print(f"  Forecast start: {full_data.index[forecast_start_idx]} (idx={forecast_start_idx})")
    print(f"  Forecast length: {actual_pl}")

    # Build TSDF up to forecast start for predictions
    print("\n[2/5] Getting model predictions...")
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))
    leaderboard = predictor.leaderboard(silent=True)
    all_models = leaderboard["model"].tolist()

    # Build train TSDF (data before forecast_start)
    train_data = full_data.iloc[:forecast_start_idx]
    train_tsdf = to_multi_series_tsdf(train_data)
    print(f"  Train TSDF: {train_tsdf.shape}")

    predictions = {}
    for model_name in all_models:
        try:
            preds = predictor.predict(train_tsdf, model=model_name)
            predictions[model_name] = {}
            for sid in ts_cols:
                try:
                    predictions[model_name][sid] = \
                        preds.loc[sid]["mean"].values[:actual_pl]
                except Exception:
                    pass
            print(f"  {model_name}: OK")
        except Exception as e:
            print(f"  {model_name}: failed ({e})")

    # Build features
    print("\n[3/5] Building features...")
    feature_df = build_all_features(full_data, forecast_start_idx, actual_pl)
    meta_cols_list = ["_series", "_step_idx"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols_list]
    X_all = feature_df[feature_cols].fillna(feature_df[feature_cols].median())
    print(f"  Feature matrix: {X_all.shape}")

    # Train surrogates and compute SHAP
    print("\n[4/5] Computing surrogate SHAP...")
    shap_data = {}
    faith_info = {}

    for model_name in list(predictions.keys()):
        y_all = []
        for sid in ts_cols:
            if sid in predictions[model_name]:
                pv = predictions[model_name][sid]
                y_all.append(pv[:actual_pl])
            else:
                y_all.append(np.full(actual_pl, np.nan))
        y_target = np.concatenate(y_all)
        valid_mask = ~np.isnan(y_target)

        if valid_mask.sum() < 50:
            print(f"  {model_name}: skipped (too few valid predictions)")
            continue

        X_valid = X_all[valid_mask].reset_index(drop=True)
        y_valid = y_target[valid_mask]

        # Train surrogate
        surrogate = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, random_state=42, verbose=-1,
        )
        surrogate.fit(X_valid, y_valid)

        # Faithfulness
        y_pred = surrogate.predict(X_valid)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        faith_info[model_name] = r2
        print(f"  {model_name}: R²={r2:.4f}")

        # SHAP
        explainer = shap.TreeExplainer(surrogate)
        sv = explainer.shap_values(X_valid)

        # Build SHAP DataFrame
        shap_df = pd.DataFrame(sv, columns=feature_cols)
        valid_feature_df = feature_df[valid_mask].reset_index(drop=True)
        shap_df["_series"] = valid_feature_df["_series"].values
        shap_df["_step"] = valid_feature_df["_step_idx"].values
        shap_data[model_name] = shap_df

    if len(shap_data) < 2:
        print("[ERROR] Need >= 2 models for Rashomon analysis")
        return

    # Time axes
    forecast_time = full_data.index[forecast_start_idx:forecast_start_idx + actual_pl]
    n_hist = min(HISTORY_STEPS, forecast_start_idx)
    hist_time = full_data.index[forecast_start_idx - n_hist:forecast_start_idx]
    hist_vals = full_data[series_id].values[forecast_start_idx - n_hist:forecast_start_idx]
    test_vals = full_data[series_id].values[forecast_start_idx:forecast_start_idx + actual_pl]

    # Plot
    print(f"\n[5/5] Plotting...")
    plot_rashomon(
        shap_data, series_id, full_data, predictions,
        anomaly_labels, forecast_time, hist_time, hist_vals,
        test_vals, faith_info, RESULTS_DIR,
        top_n=args.top_n, smooth_window=args.smooth,
    )

    # Save SHAP CSVs
    for model_name, df in shap_data.items():
        df.to_csv(RESULTS_DIR / f"shap_values_{model_name}.csv", index=False)

    print(f"\nOutput: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
