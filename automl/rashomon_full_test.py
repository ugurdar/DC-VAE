"""
Rashomon SHAP over FULL test set
=================================
Rolling-origin forecasts across the entire test period, then surrogate
SHAP for every test timestep.  Anomaly overlay from TELCO_labels.

Usage:
    python rashomon_full_test.py
    python rashomon_full_test.py --series TS1 --top_n 10
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import time as _time

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
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
MODELS_DIR  = BASE_DIR / "models" / "autogluon" / "multi"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "rashomon_full_test"

PREDICTION_LENGTH = 288
TARGET_LAGS = [1, 2, 3, 6, 12, 24, 72, 144, 288]
ROLLING_WINDOWS = [12, 72, 288]

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
        return labels[~labels.index.duplicated(keep="first")].sort_index()
    return pd.DataFrame()


def to_multi_series_tsdf(df):
    ts_cols = sorted([c for c in df.columns if c.startswith("TS")])
    records = []
    for col in ts_cols:
        temp = df[[col]].copy().rename(columns={col: "target"})
        temp["item_id"] = col
        temp = temp.reset_index().rename(columns={"time": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        records.append(temp)
    long_df = pd.concat(records, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(
        long_df, id_column="item_id", timestamp_column="timestamp"
    )


# =====================================================================
# Feature Engineering — for a RANGE of timesteps
# =====================================================================
def build_features_range(values, timestamps, start_idx, end_idx, series_id, series_num):
    """Build features for timesteps [start_idx, end_idx)."""
    rows = []
    for t in range(start_idx, end_idx):
        feat = {}
        feat["step"] = t - start_idx + 1
        feat["step_norm"] = feat["step"] / (end_idx - start_idx)

        for lag in TARGET_LAGS:
            idx = t - lag
            feat[f"lag_{lag}"] = values[idx] if idx >= 0 else np.nan

        for w in ROLLING_WINDOWS:
            s = max(0, t - w)
            wv = values[s:t]
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

        # Use data before test start as context
        ctx = values[:start_idx] if start_idx > 0 else values[:max(1, t)]
        feat["series_mean"] = np.mean(ctx)
        feat["series_std"] = np.std(ctx)
        feat["series_last"] = values[t - 1] if t > 0 else values[0]
        feat["series_id_num"] = series_num

        rows.append(feat)
    return pd.DataFrame(rows)


# =====================================================================
# Rolling predictions
# =====================================================================
def get_rolling_predictions(predictor, full_data, test_start_idx, model_name,
                             ts_cols, step=288):
    """
    Slide forecast origin through test set in steps of `step`.
    Returns dict: {series_id: np.array of predictions for full test}.
    """
    n_total = len(full_data)
    test_len = n_total - test_start_idx
    all_preds = {sid: np.full(test_len, np.nan) for sid in ts_cols}

    origins = list(range(test_start_idx, n_total, step))
    print(f"    Rolling: {len(origins)} windows (step={step})", flush=True)

    for i, origin in enumerate(origins):
        train_slice = full_data.iloc[:origin]
        tsdf = to_multi_series_tsdf(train_slice)
        try:
            preds = predictor.predict(tsdf, model=model_name)
            for sid in ts_cols:
                try:
                    vals = preds.loc[sid]["mean"].values
                    out_start = origin - test_start_idx
                    out_end = min(out_start + len(vals), test_len)
                    n_fill = out_end - out_start
                    all_preds[sid][out_start:out_end] = vals[:n_fill]
                except Exception:
                    pass
        except Exception as e:
            print(f"    Window {i}: failed ({e})", flush=True)

        if (i + 1) % 10 == 0:
            print(f"    Window {i+1}/{len(origins)} done", flush=True)

    return all_preds


# =====================================================================
# Plot
# =====================================================================
def smooth(values, window=15):
    if len(values) <= window:
        return values
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


def get_anomaly_regions(anomaly_labels, series_id, time_axis):
    """Return (anom_arr, starts, ends) for contiguous anomaly blocks."""
    anomaly_times = []
    if not anomaly_labels.empty and series_id in anomaly_labels.columns:
        ts_labels = anomaly_labels[series_id]
        mask = (ts_labels.index >= time_axis[0]) & (ts_labels.index <= time_axis[-1])
        anomaly_times = ts_labels[mask][ts_labels[mask] > 0].index.tolist()

    anom_arr = np.zeros(len(time_axis), dtype=bool)
    time_set = {t: i for i, t in enumerate(time_axis)}
    for at in anomaly_times:
        if at in time_set:
            anom_arr[time_set[at]] = True

    starts, ends = np.array([], dtype=int), np.array([], dtype=int)
    if anom_arr.any():
        changes = np.diff(anom_arr.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if anom_arr[0]:
            starts = np.insert(starts, 0, 0)
        if anom_arr[-1]:
            ends = np.append(ends, len(anom_arr))

    return anomaly_times, starts, ends


def shade_anomalies(ax, time_axis, starts, ends, color="black", alpha=0.15, label=True):
    """Add anomaly shading to an axis."""
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(time_axis[s], time_axis[min(e, len(time_axis)-1)],
                   alpha=alpha, color=color,
                   label="Anomaly" if (label and i == 0) else None)


def plot_rashomon_full(shap_data, series_id, test_timestamps, test_vals,
                        all_predictions, anomaly_labels, faith_info, out_dir,
                        top_n=10, smooth_window=15):
    """
    4-panel plot:
      1) Time series + model predictions + anomalies (black)
      2) All-features SHAP with Rashomon CI + anomalies
      3) Per-model uncertainty (total |SHAP| per model) + anomalies
      4) Total Rashomon uncertainty (inter-model std) + anomalies
    """
    meta_cols = {"_series", "_step"}
    model_names = sorted(shap_data.keys())

    # Rank features
    all_importance = []
    for df in shap_data.values():
        sdf = df[df["_series"] == series_id]
        feat_cols = [c for c in sdf.columns if c not in meta_cols]
        all_importance.append(sdf[feat_cols].abs().mean())
    combined = pd.concat(all_importance, axis=1).mean(axis=1)
    features_to_plot = combined.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"  Top {top_n}: {features_to_plot}")

    # Collect per-model SHAP matrices: {model: (n_steps, n_features)}
    model_shap_matrices = {}
    n_steps = None
    for model_name, df in shap_data.items():
        sdf = df[df["_series"] == series_id].sort_values("_step").reset_index(drop=True)
        feat_cols = [c for c in sdf.columns if c not in meta_cols]
        mat = sdf[feat_cols].values  # (steps, features)
        if n_steps is None:
            n_steps = len(mat)
        else:
            n_steps = min(n_steps, len(mat))
        model_shap_matrices[model_name] = mat

    # Trim all to same length
    for m in model_shap_matrices:
        model_shap_matrices[m] = model_shap_matrices[m][:n_steps]

    # Per-feature mean & std across models
    feature_mean = {}
    feature_std = {}
    for feat in features_to_plot:
        model_vals = []
        for model_name, df in shap_data.items():
            sdf = df[df["_series"] == series_id].sort_values("_step").reset_index(drop=True)
            if feat in sdf.columns:
                model_vals.append(sdf[feat].values[:n_steps])
        if len(model_vals) < 2:
            continue
        stacked = np.array(model_vals)
        feature_mean[feat] = np.mean(stacked, axis=0)
        feature_std[feat] = np.std(stacked, axis=0)

    if not feature_mean:
        return

    time_axis = test_timestamps[:n_steps]
    vals = test_vals[:n_steps]

    # Per-model uncertainty: total |SHAP| at each timestep
    model_total_shap = {}
    for model_name, mat in model_shap_matrices.items():
        model_total_shap[model_name] = np.sum(np.abs(mat), axis=1)

    # Total Rashomon uncertainty: mean inter-model std across features
    # Stack all models: (n_models, n_steps, n_features)
    first_df = next(iter(shap_data.values()))
    sdf0 = first_df[first_df["_series"] == series_id].sort_values("_step").reset_index(drop=True)
    all_feat_cols = [c for c in sdf0.columns if c not in meta_cols]

    model_stack = []  # (n_models, n_steps, n_features)
    for model_name in model_names:
        df = shap_data[model_name]
        sdf = df[df["_series"] == series_id].sort_values("_step").reset_index(drop=True)
        model_stack.append(sdf[all_feat_cols].values[:n_steps])
    model_stack = np.array(model_stack)  # (M, T, F)

    # Inter-model std per feature per timestep, then mean across features
    inter_model_std = np.std(model_stack, axis=0)  # (T, F)
    total_rashomon = np.mean(inter_model_std, axis=1)  # (T,)
    # Also max across features for "worst case" disagreement
    max_rashomon = np.max(inter_model_std, axis=1)  # (T,)

    # Anomaly regions
    anomaly_times, starts, ends = get_anomaly_regions(anomaly_labels, series_id, time_axis)
    print(f"  Anomalies in range: {len(anomaly_times)}")

    # Model colours
    MODEL_COLORS = {"DirectTabular": "#1f77b4", "RecursiveTabular": "#d62728",
                    "WeightedEnsemble": "#2ca02c"}
    for m in model_names:
        if m not in MODEL_COLORS:
            MODEL_COLORS[m] = "#ff7f0e"

    date_fmt = mdates.DateFormatter("%m-%d")

    # ======== 4-panel figure ========
    fig, (ax_ts, ax_shap, ax_model_unc, ax_total_unc) = plt.subplots(
        4, 1, figsize=(24, 20),
        gridspec_kw={"height_ratios": [1.2, 1.5, 1, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.18)

    # ---- Panel 1: Time series + model predictions + anomalies ----
    ax_ts.plot(time_axis, vals, color="steelblue", lw=0.8, alpha=0.9, label="Actual")
    for model_name in model_names:
        if model_name in all_predictions and series_id in all_predictions[model_name]:
            pred = all_predictions[model_name][series_id][:n_steps]
            r2_str = f" (R²={faith_info[model_name]:.3f})" if model_name in faith_info else ""
            ax_ts.plot(time_axis[:len(pred)], pred, lw=0.7, alpha=0.7,
                       color=MODEL_COLORS.get(model_name, "#ff7f0e"),
                       ls="--", label=f"{model_name}{r2_str}")
    shade_anomalies(ax_ts, time_axis, starts, ends, color="black", alpha=0.20)
    ax_ts.set_ylabel(series_id, fontsize=12)
    ax_ts.set_title(f"{series_id} — Actual vs Model Forecasts + Rashomon SHAP Analysis",
                    fontweight="bold", fontsize=14)
    ax_ts.legend(fontsize=8, loc="upper right", ncol=2)
    ax_ts.grid(True, alpha=0.2)

    # ---- Panel 2: All-features SHAP with Rashomon bands + anomalies ----
    feat_list = list(feature_mean.keys())
    for i, feat in enumerate(feat_list):
        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        mean_s = smooth(feature_mean[feat], smooth_window)
        std_s = smooth(feature_std[feat], smooth_window)
        ax_shap.plot(time_axis, mean_s, lw=1.5, color=color, label=feat, alpha=0.9)
        ax_shap.fill_between(time_axis, mean_s - std_s, mean_s + std_s,
                             alpha=0.10, color=color)
    ax_shap.axhline(0, color="gray", lw=0.8, ls="--")
    shade_anomalies(ax_shap, time_axis, starts, ends, color="black", alpha=0.12)
    ax_shap.set_ylabel("SHAP Value", fontsize=12)
    ax_shap.set_title("Feature SHAP (mean across models, band = Rashomon ±1σ)",
                      fontweight="bold", fontsize=12)
    ax_shap.legend(fontsize=7, loc="upper left", ncol=4, framealpha=0.9,
                   title="Features", title_fontsize=8)
    ax_shap.grid(True, alpha=0.2)

    # ---- Panel 3: Per-model uncertainty (total |SHAP|) + anomalies ----
    for model_name in model_names:
        vals_m = smooth(model_total_shap[model_name], smooth_window)
        r2_str = f" (R²={faith_info[model_name]:.3f})" if model_name in faith_info else ""
        ax_model_unc.plot(time_axis, vals_m, lw=1.5,
                          color=MODEL_COLORS.get(model_name, "#ff7f0e"),
                          label=f"{model_name}{r2_str}", alpha=0.9)
    shade_anomalies(ax_model_unc, time_axis, starts, ends, color="black", alpha=0.12)
    ax_model_unc.set_ylabel("Total |SHAP|", fontsize=12)
    ax_model_unc.set_title("Per-Model Explanation Magnitude (Total |SHAP| over time)",
                           fontweight="bold", fontsize=12)
    ax_model_unc.legend(fontsize=9, loc="upper right")
    ax_model_unc.grid(True, alpha=0.2)

    # ---- Panel 4: Total Rashomon uncertainty + anomalies ----
    total_s = smooth(total_rashomon, smooth_window)
    max_s = smooth(max_rashomon, smooth_window)
    ax_total_unc.fill_between(time_axis, 0, max_s, alpha=0.15, color="crimson",
                              label="Max feature disagreement")
    ax_total_unc.fill_between(time_axis, 0, total_s, alpha=0.3, color="steelblue",
                              label="Mean feature disagreement")
    ax_total_unc.plot(time_axis, total_s, lw=1.5, color="steelblue", alpha=0.9)
    ax_total_unc.plot(time_axis, max_s, lw=1.0, color="crimson", alpha=0.7, ls="--")
    shade_anomalies(ax_total_unc, time_axis, starts, ends, color="black", alpha=0.12)
    ax_total_unc.set_ylabel("Inter-Model σ", fontsize=12)
    ax_total_unc.set_xlabel("Time", fontsize=12)
    ax_total_unc.set_title("Total Rashomon Uncertainty (inter-model SHAP disagreement)",
                           fontweight="bold", fontsize=12)
    ax_total_unc.legend(fontsize=9, loc="upper right")
    ax_total_unc.grid(True, alpha=0.2)

    # Shared x-axis
    ax_total_unc.xaxis.set_major_formatter(date_fmt)
    ax_total_unc.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_total_unc.tick_params(axis="x", rotation=30, labelsize=9)

    out_path = out_dir / f"rashomon_full_{series_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="TS1")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=15)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to use (default: top 3 by leaderboard)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Rashomon SHAP — Full Test Set")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    train, val, test = load_telco_data()
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])

    train_val = pd.concat([train, val])
    train_val = train_val[~train_val.index.duplicated(keep="first")].sort_index()
    test_start_idx = len(train_val)
    test_len = len(test)
    test_timestamps = full_data.index[test_start_idx:]
    print(f"  Train+val: {test_start_idx}, Test: {test_len}")
    print(f"  Test range: {test_timestamps[0]} -> {test_timestamps[-1]}")

    anomaly_labels = load_anomaly_labels()

    # Load predictor
    print("\n[2/6] Loading predictor...")
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))
    lb = predictor.leaderboard(silent=True)
    print(lb[["model", "score_val"]].to_string())

    if args.models:
        selected = args.models
    else:
        # Top 3 non-ensemble models
        non_ens = lb[~lb["model"].str.contains("Ensemble")]
        selected = non_ens["model"].head(3).tolist()
    print(f"\n  Selected models: {selected}")

    # Rolling predictions
    print("\n[3/6] Rolling predictions over test set...")
    all_predictions = {}
    t0 = _time.time()
    for model_name in selected:
        print(f"\n  --- {model_name} ---")
        preds = get_rolling_predictions(
            predictor, full_data, test_start_idx, model_name,
            ts_cols, step=PREDICTION_LENGTH,
        )
        all_predictions[model_name] = preds
    print(f"\n  Predictions done in {_time.time()-t0:.0f}s")

    # Build features for full test
    print("\n[4/6] Building features for full test set...")
    series_map = {s: i for i, s in enumerate(ts_cols)}
    all_feat_frames = []
    for sid in ts_cols:
        values = full_data[sid].values
        timestamps = full_data.index
        feat_df = build_features_range(
            values, timestamps, test_start_idx,
            test_start_idx + test_len, sid, series_map[sid],
        )
        feat_df["_series"] = sid
        feat_df["_step"] = range(test_len)
        all_feat_frames.append(feat_df)
    feature_df = pd.concat(all_feat_frames, ignore_index=True)
    meta_cols = ["_series", "_step"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    X_all = feature_df[feature_cols].fillna(feature_df[feature_cols].median())
    print(f"  Feature matrix: {X_all.shape}")

    # Train surrogates + SHAP
    print("\n[5/6] Surrogate SHAP per model...")
    shap_results = {}
    faith_info = {}

    for model_name in selected:
        print(f"\n  --- {model_name} ---")
        y_all = []
        for sid in ts_cols:
            y_all.append(all_predictions[model_name][sid][:test_len])
        y_target = np.concatenate(y_all)
        valid = ~np.isnan(y_target)
        print(f"    Valid predictions: {valid.sum()}/{len(y_target)}")

        if valid.sum() < 100:
            print(f"    Skipping (too few)")
            continue

        X_v = X_all[valid].reset_index(drop=True)
        y_v = y_target[valid]

        surrogate = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, random_state=42, verbose=-1,
        )
        surrogate.fit(X_v, y_v)

        y_pred = surrogate.predict(X_v)
        ss_res = np.sum((y_v - y_pred) ** 2)
        ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        faith_info[model_name] = r2
        print(f"    Surrogate R²={r2:.4f}")

        print(f"    Computing SHAP...")
        explainer = shap.TreeExplainer(surrogate)
        sv = explainer.shap_values(X_v)

        shap_df = pd.DataFrame(sv, columns=feature_cols)
        valid_meta = feature_df[valid].reset_index(drop=True)
        shap_df["_series"] = valid_meta["_series"].values
        shap_df["_step"] = valid_meta["_step"].values
        shap_results[model_name] = shap_df

        shap_df.to_csv(RESULTS_DIR / f"shap_values_{model_name}.csv", index=False)
        print(f"    Saved SHAP CSV")

    if len(shap_results) < 2:
        print("[ERROR] Need >= 2 models")
        return

    # Plot
    print(f"\n[6/6] Plotting {args.series}...")
    series_id = args.series
    test_vals = full_data[series_id].values[test_start_idx:test_start_idx + test_len]

    plot_rashomon_full(
        shap_results, series_id, test_timestamps, test_vals,
        all_predictions, anomaly_labels, faith_info, RESULTS_DIR,
        top_n=args.top_n, smooth_window=args.smooth,
    )

    print(f"\nOutput: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
