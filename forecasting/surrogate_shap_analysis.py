"""
Surrogate SHAP Analysis — Extended Forecasting Pipeline
========================================================
Applies TreeSHAP to ALL trained models via LightGBM surrogate approach.
Anomaly regions from TELCO labels are overlaid on all temporal plots.

Usage:
    python forecasting/surrogate_shap_analysis.py
    python forecasting/surrogate_shap_analysis.py --models DirectTabular RecursiveTabular
    python forecasting/surrogate_shap_analysis.py --top_n 20
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

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
RESULTS_DIR = BASE_DIR / "results" / "surrogate_shap"
MODELS_DIR  = BASE_DIR / "models" / "multi"

PREDICTION_LENGTH = 288
FREQ = "5min"

TARGET_LAGS = [1, 2, 3, 6, 12, 24, 72, 144, 288]
ROLLING_WINDOWS = [12, 72, 288]


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
# Feature Engineering
# =====================================================================
def build_features_for_series(values, timestamps, forecast_start_idx,
                              prediction_length, series_id):
    rows = []
    for h in range(prediction_length):
        t = forecast_start_idx + h
        feat = {"step": h + 1, "step_norm": (h + 1) / prediction_length}

        for lag in TARGET_LAGS:
            idx = t - lag
            feat[f"lag_{lag}"] = values[idx] if idx >= 0 else np.nan

        for w in ROLLING_WINDOWS:
            start = max(0, t - w)
            window_vals = values[start:t]
            if len(window_vals) > 0:
                feat[f"roll_mean_{w}"] = np.mean(window_vals)
                feat[f"roll_std_{w}"] = np.std(window_vals)
                feat[f"roll_min_{w}"] = np.min(window_vals)
                feat[f"roll_max_{w}"] = np.max(window_vals)
            else:
                for s in ["mean", "std", "min", "max"]:
                    feat[f"roll_{s}_{w}"] = np.nan

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
        feat["series_id"] = series_id
        rows.append(feat)
    return pd.DataFrame(rows)


def build_all_features(full_data, prediction_length):
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    all_frames = []
    forecast_start_idx = len(full_data) - prediction_length
    for col in ts_cols:
        feat_df = build_features_for_series(
            full_data[col].values, full_data.index,
            forecast_start_idx, prediction_length, col
        )
        feat_df["_series"] = col
        feat_df["_step_idx"] = range(prediction_length)
        all_frames.append(feat_df)
    return pd.concat(all_frames, ignore_index=True)


# =====================================================================
# Surrogate Training & SHAP
# =====================================================================
def train_surrogate(X, y, model_name):
    surrogate = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, random_state=42, verbose=-1,
    )
    surrogate.fit(X, y)
    return surrogate


def evaluate_surrogate(surrogate, X, y_true):
    y_pred = surrogate.predict(X)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Correlation": corr}


def compute_shap_values(surrogate, X):
    explainer = shap.TreeExplainer(surrogate)
    return explainer.shap_values(X)


# =====================================================================
# Plots — with anomaly overlays
# =====================================================================
def plot_shap_bar(shap_vals, feature_names, model_name, out_dir, top_n=20):
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    names = [feature_names[i] for i in order]
    vals = mean_abs[order]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(vals)))
    ax.barh(range(len(names))[::-1], vals, color=colors)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance — {model_name}", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_bar_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_with_shap(shap_vals, feature_df, feature_names, model_name,
                            full_data, predictions_dict, test_timestamps,
                            labels_df, out_dir, top_n=5, history_steps=500):
    """3-panel plot per series: forecast + SHAP lines + stacked area, with anomaly overlay."""
    mean_abs_global = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_global)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]

    series_list = sorted(feature_df["_series"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    for series_id in series_list:
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values]
        n_steps = series_shap.shape[0]
        time_forecast = test_timestamps[:n_steps]

        actual_full = full_data[series_id].values
        actual_idx = full_data.index
        n_hist = min(history_steps, len(actual_full) - PREDICTION_LENGTH)
        hist_time = actual_idx[-(n_hist + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        hist_vals = actual_full[-(n_hist + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        test_vals = actual_full[-PREDICTION_LENGTH:]
        pred_vals = predictions_dict.get(series_id, None)

        fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                                 gridspec_kw={"height_ratios": [2, 1.5, 1]})

        # Panel 1: Forecast
        ax = axes[0]
        ax.plot(hist_time, hist_vals, color="steelblue", lw=1.2, label="Train (tail)")
        ax.plot(time_forecast[:len(test_vals)], test_vals[:len(time_forecast)],
                color="forestgreen", lw=2, label="Actual (test)")
        if pred_vals is not None:
            ax.plot(time_forecast[:len(pred_vals)], pred_vals[:len(time_forecast)],
                    color="crimson", lw=2, ls="--", label=f"Forecast ({model_name})")
        ax.axvspan(time_forecast[0], time_forecast[-1], alpha=0.05, color="forestgreen")

        # Anomaly overlay
        hist_anomaly = get_anomaly_mask(labels_df, series_id, hist_time)
        overlay_anomaly_regions(ax, hist_time, hist_anomaly, alpha=0.06)
        forecast_anomaly = get_anomaly_mask(labels_df, series_id, time_forecast)
        overlay_anomaly_regions(ax, time_forecast, forecast_anomaly, alpha=0.12,
                                label="Anomaly")

        ax.set_ylabel(series_id, fontsize=11)
        ax.set_title(f"{series_id} — Forecast + SHAP — {model_name}",
                     fontweight="bold", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)

        # Panel 2: SHAP Lines
        ax2 = axes[1]
        for rank, (feat_idx, fname) in enumerate(zip(order, top_feat_names)):
            vals = series_shap[:, feat_idx]
            if len(vals) > 30:
                w = max(1, len(vals) // 30)
                vals = pd.Series(vals).rolling(w, min_periods=1, center=True).mean().values
            ax2.plot(time_forecast, vals, lw=1.5, label=fname, color=colors[rank], alpha=0.85)
        ax2.axhline(0, color="gray", lw=0.8, ls="--")
        overlay_anomaly_regions(ax2, time_forecast, forecast_anomaly, alpha=0.10)
        ax2.set_ylabel("SHAP Value", fontsize=11)
        ax2.set_title("Feature SHAP Contributions Over Time", fontweight="bold", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right", ncol=2)
        ax2.grid(True, alpha=0.2)

        # Panel 3: Stacked Area
        ax3 = axes[2]
        abs_shap = np.abs(series_shap[:, order])
        smoothed = np.zeros_like(abs_shap)
        if abs_shap.shape[0] > 30:
            w = max(1, abs_shap.shape[0] // 30)
            for j in range(abs_shap.shape[1]):
                smoothed[:, j] = pd.Series(abs_shap[:, j]).rolling(
                    w, min_periods=1, center=True).mean().values
        else:
            smoothed = abs_shap

        ax3.stackplot(time_forecast, smoothed.T, labels=top_feat_names,
                      colors=colors[:top_n], alpha=0.7)
        overlay_anomaly_regions(ax3, time_forecast, forecast_anomaly, alpha=0.10)
        ax3.set_ylabel("|SHAP| (stacked)", fontsize=11)
        ax3.set_xlabel("Time", fontsize=11)
        ax3.set_title("Stacked Feature Contributions (absolute)", fontweight="bold", fontsize=10)
        ax3.legend(fontsize=7, loc="upper right", ncol=2)
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        fig.savefig(out_dir / f"forecast_shap_{model_name}_{series_id}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_shap_vs_horizon(shap_vals, feature_df, feature_names, model_name, out_dir, top_n=6):
    mean_abs_global = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_global)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]
    steps = feature_df["_step_idx"].values
    unique_steps = np.sort(np.unique(steps))

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    for rank, (feat_idx, fname) in enumerate(zip(order, top_feat_names)):
        step_means = []
        for s in unique_steps:
            mask = steps == s
            step_means.append(np.mean(np.abs(shap_vals[mask, feat_idx])))
        step_means = np.array(step_means)
        if len(step_means) > 20:
            window = max(1, len(step_means) // 30)
            step_means = pd.Series(step_means).rolling(window, min_periods=1, center=True).mean().values
        ax.plot(unique_steps + 1, step_means, lw=1.8, label=fname, color=colors[rank], alpha=0.85)

    ax.set_xlabel("Forecast Step (h)", fontsize=11)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Feature Importance vs Forecast Horizon — {model_name}", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_vs_horizon_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model_comparison(all_shap, feature_names, out_dir, top_n=15):
    if len(all_shap) < 2:
        return
    records = {}
    for model_name, sv in all_shap.items():
        mean_abs = np.mean(np.abs(sv), axis=0)
        records[model_name] = {feature_names[i]: mean_abs[i] for i in range(len(feature_names))}
    df = pd.DataFrame(records)
    avg = df.mean(axis=1).sort_values(ascending=False)
    top_feats = avg.head(top_n).index.tolist()
    df = df.loc[top_feats]
    normed = df.div(df.max(axis=0), axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(records) * 1.8), max(6, top_n * 0.4)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    for i in range(len(top_feats)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normed.iloc[i, j] > 0.6 else "black")
    ax.set_title("Cross-Model SHAP Comparison (Normalized)", fontweight="bold", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalized |SHAP|")
    plt.tight_layout()
    fig.savefig(out_dir / "cross_model_shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_faithfulness(faithfulness, out_dir):
    models = list(faithfulness.keys())
    r2_vals = [faithfulness[m]["R2"] for m in models]
    corr_vals = [faithfulness[m]["Correlation"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(models) * 0.4)))
    ax = axes[0]
    colors = ["#2ca02c" if v > 0.9 else "#ff7f0e" if v > 0.7 else "#d62728" for v in r2_vals]
    ax.barh(models, r2_vals, color=colors, edgecolor="white")
    ax.axvline(0.9, color="green", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("R²")
    ax.set_title("Surrogate Faithfulness (R²)", fontweight="bold")
    for i, v in enumerate(r2_vals):
        ax.text(max(v + 0.01, 0.02), i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")

    ax = axes[1]
    colors = ["#2ca02c" if v > 0.95 else "#ff7f0e" if v > 0.85 else "#d62728" for v in corr_vals]
    ax.barh(models, corr_vals, color=colors, edgecolor="white")
    ax.set_xlabel("Pearson Correlation")
    ax.set_title("Surrogate Faithfulness (Correlation)", fontweight="bold")
    for i, v in enumerate(corr_vals):
        ax.text(max(v + 0.01, 0.02), i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("Surrogate Model Faithfulness", fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / "surrogate_faithfulness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_anomaly_comparison(shap_vals, feature_df, feature_names, model_name,
                                  labels_df, test_timestamps, out_dir, top_n=10):
    """Bar chart: mean |SHAP| during normal vs anomaly periods."""
    series_list = sorted(feature_df["_series"].unique())

    for series_id in series_list:
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values]
        n_steps = series_shap.shape[0]
        forecast_time = test_timestamps[:n_steps]

        anomaly_mask = get_anomaly_mask(labels_df, series_id, forecast_time)
        if not anomaly_mask.any() or anomaly_mask.all():
            continue

        normal_shap = np.mean(np.abs(series_shap[~anomaly_mask]), axis=0)
        anomaly_shap = np.mean(np.abs(series_shap[anomaly_mask]), axis=0)

        combined = normal_shap + anomaly_shap
        order = np.argsort(combined)[::-1][:top_n]
        names = [feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
        y = np.arange(len(names))
        bar_h = 0.35
        ax.barh(y - bar_h/2, normal_shap[order], bar_h, color="steelblue",
                label="Normal", edgecolor="white")
        ax.barh(y + bar_h/2, anomaly_shap[order], bar_h, color="crimson",
                label="Anomaly", edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(f"SHAP: Normal vs Anomaly — {model_name} — {series_id}",
                     fontweight="bold", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(out_dir / f"shap_anomaly_vs_normal_{model_name}_{series_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Surrogate SHAP for Extended Forecasting")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Surrogate SHAP Analysis — Extended Forecasting")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    train, val, test = load_telco_data()
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  Series: {ts_cols}")

    labels_df = load_anomaly_labels()
    print(f"  Anomaly labels: {len(labels_df)} rows")

    # Load predictor
    print("\n[2/6] Loading predictor...")
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))
    leaderboard = predictor.leaderboard(silent=True)
    all_models = leaderboard["model"].tolist()
    print(f"  Available models: {all_models}")

    selected_models = [m for m in args.models if m in all_models] if args.models else all_models
    print(f"  Selected: {selected_models}")

    # Build features
    print("\n[3/6] Building features...")
    feature_df = build_all_features(full_data, PREDICTION_LENGTH)
    series_map = {s: i for i, s in enumerate(sorted(feature_df["series_id"].unique()))}
    feature_df["series_id_num"] = feature_df["series_id"].map(series_map)

    meta_cols = ["_series", "_step_idx", "series_id"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    X_all = feature_df[feature_cols].copy().fillna(feature_df[feature_cols].median())
    print(f"  Feature matrix: {X_all.shape}")

    test_timestamps = full_data.index[-PREDICTION_LENGTH:]

    # Prepare TSDF for predictions
    train_tsdf = to_multi_series_tsdf(pd.concat([train, val]).pipe(
        lambda d: d[~d.index.duplicated(keep="first")].sort_index()
    ))

    # SHAP loop
    print("\n[4/6] Computing surrogate SHAP...")
    all_shap_values = {}
    all_faithfulness = {}

    for model_name in selected_models:
        print(f"\n  === {model_name} ===")
        try:
            preds = predictor.predict(train_tsdf, model=model_name)
        except Exception as e:
            print(f"    [ERROR] Prediction failed: {e}")
            continue

        y_all = []
        predictions_dict = {}
        for series_id in ts_cols:
            try:
                item_preds = preds.loc[series_id]["mean"].values
                if len(item_preds) >= PREDICTION_LENGTH:
                    item_preds = item_preds[:PREDICTION_LENGTH]
                y_all.append(item_preds)
                predictions_dict[series_id] = item_preds
            except Exception:
                y_all.append(np.full(PREDICTION_LENGTH, np.nan))

        y_target = np.concatenate(y_all)
        valid_mask = ~np.isnan(y_target)
        if valid_mask.sum() < 100:
            print(f"    [WARN] Too few valid predictions, skipping")
            continue

        X_valid = X_all[valid_mask].reset_index(drop=True)
        y_valid = y_target[valid_mask]

        print(f"    Training surrogate ({len(y_valid)} samples)...")
        surrogate = train_surrogate(X_valid, y_valid, model_name)
        metrics = evaluate_surrogate(surrogate, X_valid, y_valid)
        all_faithfulness[model_name] = metrics
        print(f"    R²={metrics['R2']:.4f}, Corr={metrics['Correlation']:.4f}")

        if metrics["R2"] < 0.5:
            print(f"    [WARN] Low R² — SHAP may not be faithful")

        print(f"    Computing TreeSHAP...")
        sv = compute_shap_values(surrogate, X_valid)
        all_shap_values[model_name] = sv

        fn = feature_cols
        valid_feature_df = feature_df[valid_mask].reset_index(drop=True)

        print(f"    Generating plots...")
        plot_shap_bar(sv, fn, model_name, RESULTS_DIR, top_n=args.top_n)
        plot_shap_vs_horizon(sv, valid_feature_df, fn, model_name, RESULTS_DIR)
        plot_forecast_with_shap(sv, valid_feature_df, fn, model_name,
                                full_data, predictions_dict, test_timestamps,
                                labels_df, RESULTS_DIR)
        plot_shap_anomaly_comparison(sv, valid_feature_df, fn, model_name,
                                      labels_df, test_timestamps, RESULTS_DIR)

        # Export SHAP values
        shap_df = pd.DataFrame(sv, columns=fn)
        shap_df["_series"] = valid_feature_df["_series"].values
        shap_df["_step"] = valid_feature_df["_step_idx"].values
        shap_df.to_csv(RESULTS_DIR / f"shap_values_{model_name}.csv", index=False)
        print(f"    Exported: shap_values_{model_name}.csv")

    # Cross-model comparison
    print(f"\n[5/6] Cross-model comparison...")
    if len(all_shap_values) >= 2:
        plot_cross_model_comparison(all_shap_values, feature_cols, RESULTS_DIR)

    if all_faithfulness:
        plot_faithfulness(all_faithfulness, RESULTS_DIR)
        pd.DataFrame(all_faithfulness).T.to_csv(RESULTS_DIR / "surrogate_faithfulness.csv")

    # Summary
    print("\n[6/6] Summary")
    print("=" * 70)
    print(f"Output: {RESULTS_DIR}")
    print(f"\nSurrogate Faithfulness:")
    for m, f in all_faithfulness.items():
        print(f"  {m:30s} R²={f['R2']:.4f} Corr={f['Correlation']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
