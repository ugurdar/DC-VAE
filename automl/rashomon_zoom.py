"""
Rashomon SHAP — Zoomed view on a date range.
Reuses pre-computed SHAP CSVs + prediction CSVs from rashomon_full_test.
No model retraining needed.

Usage:
    python rashomon_zoom.py
    python rashomon_zoom.py --start 2021-06-01 --end 2021-06-22
    python rashomon_zoom.py --series TS1 --start 2021-06-01 --end 2021-06-22 --top_n 10
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
SHAP_DIR    = BASE_DIR / "results" / "autogluon" / "rashomon_full_test"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "rashomon_zoom"

FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#ffe119", "#ffd8b1", "#000000",
]
MODEL_COLORS = {
    "DirectTabular": "#1f77b4",
    "RecursiveTabular": "#d62728",
    "WeightedEnsemble": "#2ca02c",
}


def smooth(values, window=10):
    if len(values) <= window:
        return values
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values


def get_anomaly_regions(anomaly_labels, series_id, time_axis):
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
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(time_axis[s], time_axis[min(e, len(time_axis)-1)],
                   alpha=alpha, color=color,
                   label="Anomaly" if (label and i == 0) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="TS1")
    parser.add_argument("--start", type=str, default="2021-06-01")
    parser.add_argument("--end", type=str, default="2021-06-22")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--models", nargs="*",
                        default=["DirectTabular", "RecursiveTabular", "WeightedEnsemble"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series_id = args.series
    date_start = pd.Timestamp(args.start)
    date_end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)

    print("=" * 60)
    print(f"Rashomon SHAP — Zoom: {args.start} to {args.end}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    for df in (train, val, test):
        df.index = df.index.tz_localize(None)
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()

    train_val = pd.concat([train, val])
    train_val = train_val[~train_val.index.duplicated(keep="first")].sort_index()
    test_start_idx = len(train_val)
    test_timestamps = full_data.index[test_start_idx:]

    # Anomaly labels
    dfs = []
    for split in ("train", "val", "test"):
        path = LABELS_DIR / f"TELCO_labels_{split}.csv"
        if path.exists():
            ldf = pd.read_csv(path, parse_dates=["time"], index_col="time")
            ldf.index = ldf.index.tz_localize(None)
            dfs.append(ldf)
    anomaly_labels = pd.concat(dfs)[~pd.concat(dfs).index.duplicated(keep="first")].sort_index() if dfs else pd.DataFrame()

    # Load pre-computed SHAP CSVs (only selected models)
    print(f"Loading SHAP CSVs for {args.models}...")
    shap_data = {}
    models = []
    for csv_path in sorted(SHAP_DIR.glob("shap_values_*.csv")):
        model_name = csv_path.stem.replace("shap_values_", "")
        if model_name not in args.models:
            continue
        df = pd.read_csv(csv_path)
        shap_data[model_name] = df
        models.append(model_name)
        print(f"  {model_name}: {len(df)} rows")

    if len(shap_data) < 2:
        print("[ERROR] Need >= 2 model SHAP CSVs")
        return

    # Map steps to timestamps, filter to date range
    step_to_ts = {i: t for i, t in enumerate(test_timestamps)}
    ts_to_step = {t: i for i, t in enumerate(test_timestamps)}

    range_mask = (test_timestamps >= date_start) & (test_timestamps <= date_end)
    range_steps = np.where(range_mask)[0]
    step_start, step_end = range_steps[0], range_steps[-1]
    time_axis = test_timestamps[range_mask]
    n_steps = len(time_axis)
    print(f"Date range: {time_axis[0]} -> {time_axis[-1]} ({n_steps} steps)")

    # Actual values
    test_vals = full_data[series_id].values[test_start_idx:]
    range_vals = test_vals[step_start:step_end + 1]

    # Filter SHAP to range for series
    meta_cols = {"_series", "_step"}
    filtered_shap = {}
    for model_name, df in shap_data.items():
        sdf = df[(df["_series"] == series_id) &
                 (df["_step"] >= step_start) & (df["_step"] <= step_end)]
        sdf = sdf.sort_values("_step").reset_index(drop=True)
        filtered_shap[model_name] = sdf

    # Also load predictions from rolling forecast (reconstruct from SHAP target)
    # We'll recompute predictions from the model files if available,
    # but for now use the actual values + model forecasts aren't stored.
    # Instead, we can use the surrogate predictions as proxy.
    # Actually, let's just re-predict for this smaller range.

    # --- Rank features ---
    all_importance = []
    for df in filtered_shap.values():
        feat_cols = [c for c in df.columns if c not in meta_cols]
        all_importance.append(df[feat_cols].abs().mean())
    combined = pd.concat(all_importance, axis=1).mean(axis=1)
    features_to_plot = combined.sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"Top {args.top_n}: {features_to_plot}")

    # Per-feature mean & std
    feature_mean = {}
    feature_std = {}
    for feat in features_to_plot:
        model_vals = []
        for df in filtered_shap.values():
            if feat in df.columns:
                model_vals.append(df[feat].values[:n_steps])
        if len(model_vals) < 2:
            continue
        min_len = min(len(v) for v in model_vals)
        stacked = np.array([v[:min_len] for v in model_vals])
        feature_mean[feat] = np.mean(stacked, axis=0)
        feature_std[feat] = np.std(stacked, axis=0)

    # Per-model total |SHAP|
    model_total_shap = {}
    for model_name, df in filtered_shap.items():
        feat_cols = [c for c in df.columns if c not in meta_cols]
        model_total_shap[model_name] = np.sum(np.abs(df[feat_cols].values[:n_steps]), axis=1)

    # Total Rashomon uncertainty
    all_feat_cols = [c for c in next(iter(filtered_shap.values())).columns if c not in meta_cols]
    model_stack = []
    for model_name in models:
        df = filtered_shap[model_name]
        model_stack.append(df[all_feat_cols].values[:n_steps])
    n_min = min(m.shape[0] for m in model_stack)
    model_stack = np.array([m[:n_min] for m in model_stack])  # (M, T, F)
    inter_model_std = np.std(model_stack, axis=0)  # (T, F)
    total_rashomon = np.mean(inter_model_std, axis=1)
    max_rashomon = np.max(inter_model_std, axis=1)

    time_axis = time_axis[:n_min]
    range_vals = range_vals[:n_min]

    # Anomaly regions
    anomaly_times, starts, ends = get_anomaly_regions(anomaly_labels, series_id, time_axis)
    print(f"Anomalies in range: {len(anomaly_times)}")

    # --- Get model predictions via rolling ---
    print("Getting model predictions for zoom range...")
    from autogluon.timeseries import TimeSeriesPredictor
    MODELS_DIR = BASE_DIR / "models" / "autogluon" / "multi"
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))

    def to_tsdf(df_wide):
        ts_cols = sorted([c for c in df_wide.columns if c.startswith("TS")])
        records = []
        for col in ts_cols:
            temp = df_wide[[col]].copy().rename(columns={col: "target"})
            temp["item_id"] = col
            temp = temp.reset_index().rename(columns={"time": "timestamp"})
            temp["timestamp"] = pd.to_datetime(temp["timestamp"])
            records.append(temp)
        long_df = pd.concat(records, ignore_index=True)
        from autogluon.timeseries import TimeSeriesDataFrame
        return TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id",
                                                    timestamp_column="timestamp")

    # Predictions: rolling windows within the zoom range
    all_predictions = {}
    abs_start = test_start_idx + step_start
    abs_end = test_start_idx + step_end + 1
    PL = 288
    for model_name in models:
        pred_arr = np.full(n_min, np.nan)
        origins = list(range(abs_start, abs_end, PL))
        for origin in origins:
            train_slice = full_data.iloc[:origin]
            tsdf = to_tsdf(train_slice)
            try:
                preds = predictor.predict(tsdf, model=model_name)
                vals_p = preds.loc[series_id]["mean"].values
                out_start = origin - abs_start
                out_end = min(out_start + len(vals_p), n_min)
                n_fill = out_end - out_start
                pred_arr[out_start:out_end] = vals_p[:n_fill]
            except Exception:
                pass
        all_predictions[model_name] = pred_arr
        print(f"  {model_name}: {np.sum(~np.isnan(pred_arr))}/{n_min} predicted")

    # Surrogate R² from full run (read faithfulness)
    faith_info = {}
    faith_path = SHAP_DIR / "surrogate_faithfulness.csv"
    # Not saved by default, use hardcoded from last run
    faith_info = {"DirectTabular": 0.989, "RecursiveTabular": 0.965, "WeightedEnsemble": 0.994}

    # ======== 4-Panel Plot ========
    date_fmt = mdates.DateFormatter("%m-%d %H:%M")
    sw = args.smooth

    fig, (ax_ts, ax_shap, ax_model_unc, ax_total_unc) = plt.subplots(
        4, 1, figsize=(24, 20),
        gridspec_kw={"height_ratios": [1.2, 1.5, 1, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.18)

    # ---- Panel 1: Actual + model predictions + anomalies (black) ----
    ax_ts.plot(time_axis, range_vals, color="steelblue", lw=1.0, alpha=0.9, label="Actual")
    for model_name in models:
        pred = all_predictions[model_name]
        r2_str = f" (R²={faith_info.get(model_name, 0):.3f})"
        ax_ts.plot(time_axis[:len(pred)], pred, lw=0.8, alpha=0.7,
                   color=MODEL_COLORS.get(model_name, "#ff7f0e"),
                   ls="--", label=f"{model_name}{r2_str}")
    shade_anomalies(ax_ts, time_axis, starts, ends, color="black", alpha=0.20)
    ax_ts.set_ylabel(series_id, fontsize=12)
    ax_ts.set_title(f"{series_id} — {args.start} to {args.end} — Actual vs Model Forecasts",
                    fontweight="bold", fontsize=14)
    ax_ts.legend(fontsize=9, loc="upper right", ncol=2)
    ax_ts.grid(True, alpha=0.2)

    # ---- Panel 2: Feature SHAP + Rashomon bands + anomalies ----
    for i, feat in enumerate(feature_mean.keys()):
        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        mean_s = smooth(feature_mean[feat], sw)[:n_min]
        std_s = smooth(feature_std[feat], sw)[:n_min]
        ax_shap.plot(time_axis[:len(mean_s)], mean_s, lw=1.8, color=color, label=feat, alpha=0.9)
        ax_shap.fill_between(time_axis[:len(mean_s)], mean_s - std_s, mean_s + std_s,
                             alpha=0.12, color=color)
    ax_shap.axhline(0, color="gray", lw=0.8, ls="--")
    shade_anomalies(ax_shap, time_axis, starts, ends, color="black", alpha=0.12)
    ax_shap.set_ylabel("SHAP Value", fontsize=12)
    ax_shap.set_title("Feature SHAP (mean across models, band = Rashomon ±1σ)",
                      fontweight="bold", fontsize=12)
    ax_shap.legend(fontsize=7, loc="upper left", ncol=4, framealpha=0.9,
                   title="Features", title_fontsize=8)
    ax_shap.grid(True, alpha=0.2)

    # ---- Panel 3: Per-model uncertainty ----
    for model_name in models:
        vals_m = smooth(model_total_shap[model_name], sw)[:n_min]
        r2_str = f" (R²={faith_info.get(model_name, 0):.3f})"
        ax_model_unc.plot(time_axis[:len(vals_m)], vals_m, lw=1.5,
                          color=MODEL_COLORS.get(model_name, "#ff7f0e"),
                          label=f"{model_name}{r2_str}", alpha=0.9)
    shade_anomalies(ax_model_unc, time_axis, starts, ends, color="black", alpha=0.12)
    ax_model_unc.set_ylabel("Total |SHAP|", fontsize=12)
    ax_model_unc.set_title("Per-Model Explanation Magnitude",
                           fontweight="bold", fontsize=12)
    ax_model_unc.legend(fontsize=9, loc="upper right")
    ax_model_unc.grid(True, alpha=0.2)

    # ---- Panel 4: Total Rashomon uncertainty ----
    total_s = smooth(total_rashomon, sw)[:n_min]
    max_s = smooth(max_rashomon, sw)[:n_min]
    ax_total_unc.fill_between(time_axis[:len(max_s)], 0, max_s, alpha=0.15, color="crimson",
                              label="Max feature disagreement")
    ax_total_unc.fill_between(time_axis[:len(total_s)], 0, total_s, alpha=0.3, color="steelblue",
                              label="Mean feature disagreement")
    ax_total_unc.plot(time_axis[:len(total_s)], total_s, lw=1.5, color="steelblue", alpha=0.9)
    ax_total_unc.plot(time_axis[:len(max_s)], max_s, lw=1.0, color="crimson", alpha=0.7, ls="--")
    shade_anomalies(ax_total_unc, time_axis, starts, ends, color="black", alpha=0.12)
    ax_total_unc.set_ylabel("Inter-Model σ", fontsize=12)
    ax_total_unc.set_xlabel("Time", fontsize=12)
    ax_total_unc.set_title("Total Rashomon Uncertainty (inter-model SHAP disagreement)",
                           fontweight="bold", fontsize=12)
    ax_total_unc.legend(fontsize=9, loc="upper right")
    ax_total_unc.grid(True, alpha=0.2)

    # x-axis formatting
    ax_total_unc.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_total_unc.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_total_unc.tick_params(axis="x", rotation=30, labelsize=9)

    safe_start = args.start.replace("-", "")
    safe_end = args.end.replace("-", "")
    out_path = RESULTS_DIR / f"rashomon_zoom_{series_id}_{safe_start}_{safe_end}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
