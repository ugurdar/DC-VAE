"""
Rashomon Per-Feature Uncertainty
=================================
Top panel: time series + model predictions + anomalies
Below: one subplot per feature showing mean SHAP + Rashomon uncertainty band.

Usage:
    python rashomon_per_feature.py                                    # full test
    python rashomon_per_feature.py --start 2021-06-01 --end 2021-06-22  # zoom
    python rashomon_per_feature.py --series TS1 --top_n 8
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
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
SHAP_DIR    = BASE_DIR / "results" / "autogluon" / "rashomon_full_test"
MODELS_DIR  = BASE_DIR / "models" / "autogluon" / "multi"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "rashomon_per_feature"

PREDICTION_LENGTH = 288

FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]
MODEL_COLORS = {
    "DirectTabular": "#1f77b4",
    "RecursiveTabular": "#d62728",
    "WeightedEnsemble": "#2ca02c",
}
FAITH = {"DirectTabular": 0.989, "RecursiveTabular": 0.965, "WeightedEnsemble": 0.994}


def smooth(v, w=10):
    if len(v) <= w:
        return v
    return pd.Series(v).rolling(w, min_periods=1, center=True).mean().values


def load_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    for d in (train, val, test):
        d.index = d.index.tz_localize(None)
    full = pd.concat([train, val, test])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    tv = pd.concat([train, val])
    tv = tv[~tv.index.duplicated(keep="first")].sort_index()
    return full, len(tv)


def load_labels():
    dfs = []
    for s in ("train", "val", "test"):
        p = LABELS_DIR / f"TELCO_labels_{s}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["time"], index_col="time")
            d.index = d.index.tz_localize(None)
            dfs.append(d)
    if dfs:
        lb = pd.concat(dfs)
        return lb[~lb.index.duplicated(keep="first")].sort_index()
    return pd.DataFrame()


def load_shap(models):
    data = {}
    for p in sorted(SHAP_DIR.glob("shap_values_*.csv")):
        m = p.stem.replace("shap_values_", "")
        if m not in models:
            continue
        data[m] = pd.read_csv(p)
    return data


def to_tsdf(df_wide):
    ts_cols = sorted([c for c in df_wide.columns if c.startswith("TS")])
    recs = []
    for col in ts_cols:
        t = df_wide[[col]].copy().rename(columns={col: "target"})
        t["item_id"] = col
        t = t.reset_index().rename(columns={"time": "timestamp"})
        t["timestamp"] = pd.to_datetime(t["timestamp"])
        recs.append(t)
    return TimeSeriesDataFrame.from_data_frame(
        pd.concat(recs, ignore_index=True),
        id_column="item_id", timestamp_column="timestamp",
    )


def get_anomaly_regions(labels, sid, time_axis):
    starts, ends = np.array([], dtype=int), np.array([], dtype=int)
    if labels.empty or sid not in labels.columns:
        return [], starts, ends
    ts_l = labels[sid]
    mask = (ts_l.index >= time_axis[0]) & (ts_l.index <= time_axis[-1])
    atimes = ts_l[mask][ts_l[mask] > 0].index.tolist()
    arr = np.zeros(len(time_axis), dtype=bool)
    tset = {t: i for i, t in enumerate(time_axis)}
    for a in atimes:
        if a in tset:
            arr[tset[a]] = True
    if arr.any():
        ch = np.diff(arr.astype(int))
        starts = np.where(ch == 1)[0] + 1
        ends = np.where(ch == -1)[0] + 1
        if arr[0]:
            starts = np.insert(starts, 0, 0)
        if arr[-1]:
            ends = np.append(ends, len(arr))
    return atimes, starts, ends


def shade_anom(ax, ta, starts, ends, label=True):
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(ta[s], ta[min(e, len(ta)-1)], alpha=0.18, color="black",
                   label="Anomaly" if (label and i == 0) else None)


def rolling_preds(predictor, full_data, abs_start, abs_end, model_name, sid):
    arr = np.full(abs_end - abs_start, np.nan)
    for origin in range(abs_start, abs_end, PREDICTION_LENGTH):
        tsdf = to_tsdf(full_data.iloc[:origin])
        try:
            pr = predictor.predict(tsdf, model=model_name)
            v = pr.loc[sid]["mean"].values
            o = origin - abs_start
            n = min(o + len(v), len(arr)) - o
            arr[o:o+n] = v[:n]
        except Exception:
            pass
    return arr


# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--series", default="TS1")
    pa.add_argument("--start", default=None, help="Zoom start date (e.g. 2021-06-01)")
    pa.add_argument("--end", default=None, help="Zoom end date (e.g. 2021-06-22)")
    pa.add_argument("--top_n", type=int, default=10)
    pa.add_argument("--smooth", type=int, default=10)
    pa.add_argument("--models", nargs="*",
                    default=["DirectTabular", "RecursiveTabular", "WeightedEnsemble"])
    args = pa.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sid = args.series
    sw = args.smooth
    is_zoom = args.start is not None and args.end is not None

    tag = "zoom" if is_zoom else "full"
    print("=" * 60)
    print(f"Rashomon Per-Feature Uncertainty — {tag}")
    print("=" * 60)

    # Data
    full_data, test_start_idx = load_data()
    test_ts = full_data.index[test_start_idx:]
    test_len = len(test_ts)
    labels = load_labels()

    # SHAP
    shap_data = load_shap(args.models)
    models = list(shap_data.keys())
    print(f"Models: {models}")

    # Date range
    if is_zoom:
        ds = pd.Timestamp(args.start)
        de = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        rmask = (test_ts >= ds) & (test_ts <= de)
        range_steps = np.where(rmask)[0]
        step_s, step_e = range_steps[0], range_steps[-1]
    else:
        step_s, step_e = 0, test_len - 1

    time_axis = test_ts[step_s:step_e + 1]
    n = len(time_axis)
    test_vals = full_data[sid].values[test_start_idx + step_s:test_start_idx + step_e + 1]

    meta = {"_series", "_step"}

    # Filter SHAP
    filtered = {}
    for m, df in shap_data.items():
        sdf = df[(df["_series"] == sid) & (df["_step"] >= step_s) & (df["_step"] <= step_e)]
        filtered[m] = sdf.sort_values("_step").reset_index(drop=True)

    # Rank features
    imps = []
    for df in filtered.values():
        fc = [c for c in df.columns if c not in meta]
        imps.append(df[fc].abs().mean())
    combined = pd.concat(imps, axis=1).mean(axis=1)
    feats = combined.sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"Top {args.top_n}: {feats}")

    # Per-feature: per-model values, mean, std
    feat_per_model = {}  # {feat: {model: values}}
    feat_mean = {}
    feat_std = {}
    for feat in feats:
        feat_per_model[feat] = {}
        mvals = []
        for m, df in filtered.items():
            if feat in df.columns:
                v = df[feat].values[:n]
                feat_per_model[feat][m] = v
                mvals.append(v)
        if len(mvals) >= 2:
            mn = min(len(v) for v in mvals)
            stacked = np.array([v[:mn] for v in mvals])
            feat_mean[feat] = np.mean(stacked, axis=0)
            feat_std[feat] = np.std(stacked, axis=0)

    # Predictions
    print("Getting predictions...")
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))
    abs_s = test_start_idx + step_s
    abs_e = test_start_idx + step_e + 1
    preds = {}
    for m in models:
        preds[m] = rolling_preds(predictor, full_data, abs_s, abs_e, m, sid)
        ok = np.sum(~np.isnan(preds[m]))
        print(f"  {m}: {ok}/{n}")

    # Anomalies
    atimes, astarts, aends = get_anomaly_regions(labels, sid, time_axis)
    print(f"Anomalies: {len(atimes)}")

    # ======== Plot ========
    n_feats = len(feat_mean)
    n_panels = 1 + n_feats  # 1 TS + N features
    heights = [1.5] + [1] * n_feats
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(24, 3 + 2.5 * n_feats),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.15)

    # x-axis formatter
    if is_zoom:
        date_fmt = mdates.DateFormatter("%m-%d")
        locator = mdates.DayLocator(interval=2)
    else:
        date_fmt = mdates.DateFormatter("%m-%d")
        locator = mdates.WeekdayLocator(interval=1)

    # ---- Panel 0: Time series + predictions + anomalies ----
    ax = axes[0]
    ax.plot(time_axis, test_vals[:n], color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    for m in models:
        r2 = FAITH.get(m, 0)
        ax.plot(time_axis[:len(preds[m])], preds[m], lw=0.7, alpha=0.7, ls="--",
                color=MODEL_COLORS.get(m, "#ff7f0e"),
                label=f"{m} (R²={r2:.3f})")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(sid, fontsize=11)
    title_range = f"{args.start} to {args.end}" if is_zoom else "Full Test Set"
    ax.set_title(f"{sid} — {title_range} — Actual vs Forecasts + Per-Feature Rashomon",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

    # ---- Feature panels ----
    feat_list = list(feat_mean.keys())
    for i, feat in enumerate(feat_list):
        ax = axes[i + 1]
        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        mn = smooth(feat_mean[feat], sw)
        sd = smooth(feat_std[feat], sw)
        flen = len(mn)

        # Mean line + Rashomon band
        ax.plot(time_axis[:flen], mn, lw=1.5, color=color, alpha=0.9, label=f"{feat} (mean)")
        ax.fill_between(time_axis[:flen], mn - sd, mn + sd,
                        alpha=0.20, color=color, label="±1σ Rashomon")

        # Per-model lines (thin, dashed)
        for m in models:
            if m in feat_per_model[feat]:
                v = smooth(feat_per_model[feat][m][:flen], sw)
                ax.plot(time_axis[:len(v)], v, lw=0.6, alpha=0.5, ls="--",
                        color=MODEL_COLORS.get(m, "#999"))

        ax.axhline(0, color="gray", lw=0.5, ls="--")
        shade_anom(ax, time_axis, astarts, aends, label=(i == 0))
        ax.set_ylabel(feat, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15)
        # Small legend only on first feature panel
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    # Bottom axis labels
    axes[-1].set_xlabel("Time", fontsize=11)
    axes[-1].xaxis.set_major_formatter(date_fmt)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].tick_params(axis="x", rotation=30, labelsize=9)

    # Save
    if is_zoom:
        ss = args.start.replace("-", "")
        se = args.end.replace("-", "")
        fname = f"rashomon_perfeature_{sid}_{ss}_{se}.png"
    else:
        fname = f"rashomon_perfeature_{sid}_full.png"

    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    print("Done!")


if __name__ == "__main__":
    main()
