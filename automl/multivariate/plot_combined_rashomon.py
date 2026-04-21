"""
Combined Rashomon Plot — All Models (sklearn + H2O)
====================================================
Loads pre-computed SHAP CSVs from both diverse/ and h2o/ results.
{n_models} models total: LightGBM, XGBoost, RandomForest, Ridge, DecisionTree,
                 GBM(H2O), DRF, XRT, DeepLearning, GLM

Usage:
    python plot_combined_rashomon.py                                  # full
    python plot_combined_rashomon.py --start 2021-06-01 --end 2021-06-25  # zoom
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
DIVERSE_DIR  = BASE_DIR / "results" / "diverse"
H2O_DIR      = BASE_DIR / "results" / "h2o"
AUTOGLUON_DIR = BASE_DIR / "results"
RESULTS_DIR  = BASE_DIR / "results" / "combined"

TARGET = "TS1"

MODEL_COLORS = {
    # sklearn
    "LightGBM":         "#1f77b4",
    "XGBoost":          "#d62728",
    "RandomForest":     "#2ca02c",
    "Ridge":            "#ff7f0e",
    "DecisionTree":     "#9467bd",
    # H2O
    "GBM_H2O":         "#17becf",
    "DRF_H2O":         "#bcbd22",
    "XRT_H2O":         "#e377c2",
    "DeepLearning_H2O": "#7f7f7f",
    "GLM_H2O":         "#8c564b",
    # AutoGluon
    "LightGBMXT_BAG_L1_AG":      "#aec7e8",
    "RandomForestMSE_BAG_L1_AG":  "#98df8a",
    "LightGBM_BAG_L1_AG":         "#ff9896",
    "WeightedEnsemble_L2_AG":     "#c5b0d5",
    "WeightedEnsemble_L3_AG":     "#c49c94",
    "LightGBMXT_BAG_L2_AG":       "#dbdb8d",
}

# Short display names
MODEL_SHORT = {
    "LightGBM": "LightGBM", "XGBoost": "XGBoost",
    "RandomForest": "RandomForest", "Ridge": "Ridge",
    "DecisionTree": "DecisionTree",
    "GBM_H2O": "GBM (H2O)", "DRF_H2O": "DRF (H2O)",
    "XRT_H2O": "XRT (H2O)", "DeepLearning_H2O": "DeepLearning (H2O)",
    "GLM_H2O": "GLM (H2O)",
    "LightGBMXT_BAG_L1_AG": "LightGBMXT L1 (AG)",
    "RandomForestMSE_BAG_L1_AG": "RF L1 (AG)",
    "LightGBM_BAG_L1_AG": "LightGBM L1 (AG)",
    "WeightedEnsemble_L2_AG": "WE L2 (AG)",
    "WeightedEnsemble_L3_AG": "WE L3 (AG)",
    "LightGBMXT_BAG_L2_AG": "LightGBMXT L2 (AG)",
}

FEAT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
]


def load_test():
    d = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",
                    parse_dates=["time"], index_col="time")
    d.index = d.index.tz_localize(None)
    return d


def load_labels():
    parts = []
    for s in ("train", "val", "test"):
        p = LABELS_DIR / f"TELCO_labels_{s}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["time"], index_col="time")
            d.index = d.index.tz_localize(None)
            parts.append(d)
    lb = pd.concat(parts)
    return lb[~lb.index.duplicated(keep="first")].sort_index()


def load_all_shap():
    shap_data = {}
    # sklearn diverse
    for p in sorted(DIVERSE_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace(f"shap_", "").replace(f"_{TARGET}", "")
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        df.index = df.index.tz_localize(None)
        shap_data[name] = df
    # H2O
    for p in sorted(H2O_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace(f"shap_", "").replace(f"_{TARGET}", "")
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        df.index = df.index.tz_localize(None)
        shap_data[name + "_H2O"] = df
    # AutoGluon
    ag_files = sorted(AUTOGLUON_DIR.glob(f"shap_*_{TARGET}.csv"))
    for p in ag_files:
        name = p.stem.replace(f"shap_", "").replace(f"_{TARGET}", "")
        # Skip anomaly correlation CSV
        if "anomaly" in name:
            continue
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        df.index = df.index.tz_localize(None)
        shap_data[name + "_AG"] = df
    return shap_data


def get_anomaly_regions(labels, sid, time_axis):
    if labels.empty or sid not in labels.columns:
        return [], np.array([]), np.array([])
    ts_l = labels[sid]
    mask = (ts_l.index >= time_axis[0]) & (ts_l.index <= time_axis[-1])
    atimes = ts_l[mask][ts_l[mask] > 0].index.tolist()
    arr = np.zeros(len(time_axis), dtype=bool)
    tset = {t: i for i, t in enumerate(time_axis)}
    for a in atimes:
        if a in tset:
            arr[tset[a]] = True
    starts, ends = np.array([], dtype=int), np.array([], dtype=int)
    if arr.any():
        ch = np.diff(arr.astype(int))
        starts = np.where(ch == 1)[0] + 1
        ends   = np.where(ch == -1)[0] + 1
        if arr[0]:  starts = np.insert(starts, 0, 0)
        if arr[-1]: ends   = np.append(ends, len(arr))
    return atimes, starts, ends


def shade_anom(ax, ta, starts, ends, label=True):
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(ta[s], ta[min(e, len(ta)-1)], alpha=0.18, color="black",
                   label="Anomaly" if (label and i == 0) else None)


def smooth(v, w=15):
    if len(v) <= w:
        return v
    return pd.Series(v).rolling(w, min_periods=1, center=True).mean().values


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--start", default=None)
    pa.add_argument("--end", default=None)
    args = pa.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combined Rashomon — sklearn + H2O + AutoGluon")
    print("=" * 60)

    test = load_test()
    labels = load_labels()
    features = [c for c in test.columns if c != TARGET]

    shap_data = load_all_shap()
    model_names = list(shap_data.keys())
    n_models = len(model_names)
    print(f"Models ({n_models}): {model_names}")

    # Date filter
    is_zoom = args.start is not None and args.end is not None
    if is_zoom:
        ds = pd.Timestamp(args.start)
        de = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        mask = (test.index >= ds) & (test.index <= de)
        time_axis = test.index[mask]
        test_vals = test[TARGET][mask].values
        shap_plot = {m: df.loc[mask] for m, df in shap_data.items()}
        tag = f"zoom_{args.start.replace('-','')}_{args.end.replace('-','')}"
        sw = 10
    else:
        time_axis = test.index
        test_vals = test[TARGET].values
        shap_plot = shap_data
        tag = "full"
        sw = 15

    n = len(time_axis)
    atimes, astarts, aends = get_anomaly_regions(labels, TARGET, time_axis)
    print(f"Timesteps: {n}, Anomalies: {len(atimes)}")

    # Feature ranking
    imps = [df.abs().mean() for df in shap_plot.values()]
    combined = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined.sort_values(ascending=False).index.tolist()
    top_feats = [f for f in top_feats if f in features]
    print(f"Feature ranking: {top_feats}")

    title_range = f"{args.start} to {args.end}" if is_zoom else "Full Test"

    # ===== Plot 1: 4-panel summary =====
    fig, axes = plt.subplots(4, 1, figsize=(24, 20),
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1, 1]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Panel 1: Actual + anomalies
    ax = axes[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.set_title(f"{TARGET} — Combined Rashomon: {n_models} Models (sklearn + H2O + AG) — {title_range}",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Panel 2: Per-feature SHAP + Rashomon band
    ax = axes[1]
    for fi, feat in enumerate(top_feats):
        vals = [shap_plot[m][feat].values[:n] for m in model_names
                if feat in shap_plot[m].columns]
        if len(vals) >= 2:
            stacked = np.array(vals)
            mn = smooth(np.mean(stacked, axis=0), sw)
            sd = smooth(np.std(stacked, axis=0), sw)
            c = FEAT_COLORS[fi % len(FEAT_COLORS)]
            ax.plot(time_axis[:len(mn)], mn, lw=1.2, color=c, alpha=0.85, label=feat)
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd, alpha=0.15, color=c)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("SHAP value", fontsize=11)
    ax.set_title(f"Per-Feature SHAP (mean ± Rashomon σ across {n_models} models)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.15)

    # Panel 3: Per-model total |SHAP|
    ax = axes[2]
    for m in model_names:
        fc = [c for c in shap_plot[m].columns if c in features]
        total = smooth(shap_plot[m][fc].abs().sum(axis=1).values[:n], sw)
        ax.plot(time_axis[:len(total)], total, lw=1.0,
                color=MODEL_COLORS.get(m, "#999"), alpha=0.8,
                label=MODEL_SHORT.get(m, m))
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Total |SHAP|", fontsize=11)
    ax.set_title(f"Per-Model Total Feature Attribution ({n_models} models)", fontsize=11)
    ax.legend(fontsize=6, loc="upper right", ncol=6, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # Panel 4: Total Rashomon uncertainty
    ax = axes[3]
    all_vals = []
    for m in model_names:
        fc = [c for c in shap_plot[m].columns if c in features]
        all_vals.append(shap_plot[m][fc].values[:n])
    stacked = np.stack(all_vals)
    total_std = smooth(np.std(stacked, axis=0).mean(axis=1), sw)
    ax.fill_between(time_axis[:len(total_std)], 0, total_std,
                    alpha=0.4, color="#ff7f0e", label=f"Rashomon σ ({n_models} models)")
    ax.plot(time_axis[:len(total_std)], total_std, lw=1.0, color="#ff7f0e")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Rashomon σ", fontsize=11)
    ax.set_title(f"Total Rashomon Uncertainty ({n_models} models)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time", fontsize=11)
    if is_zoom:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].tick_params(axis="x", rotation=30, labelsize=9)

    out1 = RESULTS_DIR / f"combined_rashomon_{TARGET}_{tag}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ===== Plot 2: Per-feature subplots =====
    n_feats = len(top_feats)
    heights = [1.5] + [1] * n_feats
    fig2, axes2 = plt.subplots(
        1 + n_feats, 1,
        figsize=(24, 3 + 2.5 * n_feats),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    fig2.subplots_adjust(hspace=0.15)

    # Top panel: actual
    ax = axes2[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.set_title(f"{TARGET} — Per-Feature Rashomon ({n_models} models, {title_range})",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    for i, feat in enumerate(top_feats):
        ax = axes2[i + 1]
        c = FEAT_COLORS[i % len(FEAT_COLORS)]
        vals = []
        for m in model_names:
            if feat in shap_plot[m].columns:
                v = shap_plot[m][feat].values[:n]
                vals.append(v)
                lbl = MODEL_SHORT.get(m, m) if i == 0 else None
                ax.plot(time_axis[:len(v)], smooth(v, sw), lw=0.6, alpha=0.5,
                        ls="--", color=MODEL_COLORS.get(m, "#999"), label=lbl)
        if len(vals) >= 2:
            stacked = np.array(vals)
            mn = smooth(np.mean(stacked, axis=0), sw)
            sd = smooth(np.std(stacked, axis=0), sw)
            ax.plot(time_axis[:len(mn)], mn, lw=1.5, color=c, alpha=0.9,
                    label=f"{feat} (mean)" if i == 0 else None)
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd,
                            alpha=0.25, color=c,
                            label="±1σ Rashomon" if i == 0 else None)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        shade_anom(ax, time_axis, astarts, aends, label=(i == 0))
        ax.set_ylabel(feat, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=4, framealpha=0.9)

    axes2[-1].set_xlabel("Time", fontsize=11)
    if is_zoom:
        axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes2[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes2[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes2[-1].tick_params(axis="x", rotation=30, labelsize=9)

    out2 = RESULTS_DIR / f"combined_perfeature_{TARGET}_{tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ===== Summary stats =====
    print("\n" + "=" * 60)
    print("Per-Feature Rashomon σ (mean across time)")
    print("=" * 60)
    for feat in top_feats:
        vals = [shap_plot[m][feat].values[:n] for m in model_names
                if feat in shap_plot[m].columns]
        if len(vals) >= 2:
            sigma = np.std(np.array(vals), axis=0).mean()
            print(f"  {feat:>5s}: σ = {sigma:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
