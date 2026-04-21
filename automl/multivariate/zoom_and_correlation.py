"""
Multivariate Zoom + SHAP–Anomaly Correlation
=============================================
1) Zoomed 4-panel plot (reuses pre-computed SHAP CSVs)
2) Per-feature SHAP vs anomaly correlation analysis:
   - Welch t-test, Cohen's d, point-biserial correlation
   - Box plots, rolling correlation

Usage:
    python zoom_and_correlation.py
    python zoom_and_correlation.py --start 2021-06-01 --end 2021-06-25
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings

import numpy as np
import pandas as pd
from scipy import stats

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
SHAP_DIR    = BASE_DIR / "results"
RESULTS_DIR = BASE_DIR / "results"

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff"]
MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def smooth(v, w=10):
    if len(v) <= w:
        return v
    return pd.Series(v).rolling(w, min_periods=1, center=True).mean().values


def load_data():
    dfs = []
    for s in ("train", "val", "test"):
        d = pd.read_csv(DATA_DIR / f"TELCO_data_{s}.csv",
                        parse_dates=["time"], index_col="time")
        d.index = d.index.tz_localize(None)
        dfs.append(d)
    full = pd.concat(dfs)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return dfs[0], dfs[1], dfs[2], full


def load_labels():
    parts = []
    for s in ("train", "val", "test"):
        p = LABELS_DIR / f"TELCO_labels_{s}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["time"], index_col="time")
            d.index = d.index.tz_localize(None)
            parts.append(d)
    if parts:
        lb = pd.concat(parts)
        return lb[~lb.index.duplicated(keep="first")].sort_index()
    return pd.DataFrame()


def load_shap(models):
    data = {}
    for p in sorted(SHAP_DIR.glob("shap_*_TS1.csv")):
        m = p.stem.replace("shap_", "").replace("_TS1", "")
        if m in models:
            df = pd.read_csv(p, parse_dates=["time"], index_col="time")
            df.index = df.index.tz_localize(None)
            data[m] = df
    return data


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


def load_predictions(predictor_path, models, df_test, target):
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(predictor_path)
    preds = {}
    r2s = {}
    y = df_test[target].values
    for m in models:
        p = predictor.predict(df_test, model=m).values
        preds[m] = p
        r2s[m] = 1 - np.sum((y - p)**2) / np.sum((y - y.mean())**2)
    return preds, r2s


# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--target", default="TS1")
    pa.add_argument("--start", default="2021-06-01")
    pa.add_argument("--end", default="2021-06-25")
    pa.add_argument("--top_n", type=int, default=11)
    pa.add_argument("--smooth", type=int, default=10)
    pa.add_argument("--models", nargs="*",
                    default=["LightGBMXT_BAG_L1", "RandomForestMSE_BAG_L1", "LightGBM_BAG_L1"])
    args = pa.parse_args()
    target = args.target
    sw = args.smooth

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Multivariate Zoom + Correlation — {target}")
    print(f"Date range: {args.start} to {args.end}")
    print("=" * 60)

    # Load
    train, val, test, full = load_data()
    labels = load_labels()
    shap_data = load_shap(args.models)
    models = list(shap_data.keys())
    print(f"Models: {models}")

    features = [c for c in test.columns if c != target]

    # Date filter
    ds = pd.Timestamp(args.start)
    de = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)

    test_mask = (test.index >= ds) & (test.index <= de)
    test_zoom = test[test_mask]
    time_axis = test_zoom.index
    test_vals = test_zoom[target].values
    n = len(time_axis)
    print(f"Zoom timesteps: {n}")

    # SHAP filter
    shap_zoom = {}
    for m, df in shap_data.items():
        mask = (df.index >= ds) & (df.index <= de)
        shap_zoom[m] = df[mask]

    # Feature ranking
    imps = []
    for df in shap_zoom.values():
        fc = [c for c in df.columns if c in features]
        imps.append(df[fc].abs().mean())
    combined = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined.sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"Top features: {top_feats}")

    # Predictions
    print("Getting predictions...")
    model_path = str(BASE_DIR / "models" / target)
    preds, r2s = load_predictions(model_path, models, test_zoom, target)

    # Anomalies
    atimes, astarts, aends = get_anomaly_regions(labels, target, time_axis)
    print(f"Anomalies in range: {len(atimes)}")

    # ===== PART 1: Zoom Plot =====
    print("\n--- Zoom Plot ---")
    fig, axes = plt.subplots(4, 1, figsize=(22, 16),
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1, 1]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Panel 1: Actual + predictions
    ax = axes[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=1.0, alpha=0.9, label="Actual")
    for i, m in enumerate(models):
        ax.plot(time_axis, preds[m], lw=0.8, alpha=0.7, ls="--",
                color=MODEL_COLORS[i], label=f"{m} (R²={r2s[m]:.3f})")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(target, fontsize=11)
    ax.set_title(f"{target} — Multivariate Zoom ({args.start} to {args.end})",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel 2: Per-feature SHAP
    ax = axes[1]
    for fi, feat in enumerate(top_feats):
        vals_per_model = []
        for m in models:
            if feat in shap_zoom[m].columns:
                v = shap_zoom[m][feat].values[:n]
                vals_per_model.append(v)
        if len(vals_per_model) >= 2:
            mn_len = min(len(v) for v in vals_per_model)
            stacked = np.array([v[:mn_len] for v in vals_per_model])
            mn = smooth(np.mean(stacked, axis=0), sw)
            sd = smooth(np.std(stacked, axis=0), sw)
            c = COLORS[fi % len(COLORS)]
            ax.plot(time_axis[:len(mn)], mn, lw=1.2, color=c, alpha=0.85, label=feat)
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd, alpha=0.12, color=c)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("SHAP value", fontsize=11)
    ax.set_title("Per-Feature SHAP (mean ± Rashomon σ)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.15)

    # Panel 3: Per-model total |SHAP|
    ax = axes[2]
    for i, m in enumerate(models):
        fc = [c for c in shap_zoom[m].columns if c in features]
        total = smooth(shap_zoom[m][fc].abs().sum(axis=1).values[:n], sw)
        ax.plot(time_axis[:len(total)], total, lw=1.0, color=MODEL_COLORS[i],
                alpha=0.8, label=m)
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Total |SHAP|", fontsize=11)
    ax.set_title("Per-Model Total Feature Attribution", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)

    # Panel 4: Total Rashomon uncertainty
    ax = axes[3]
    all_vals = []
    for m in models:
        fc = [c for c in shap_zoom[m].columns if c in features]
        all_vals.append(shap_zoom[m][fc].values[:n])
    mn_len = min(v.shape[0] for v in all_vals)
    stacked = np.stack([v[:mn_len] for v in all_vals])
    total_std = smooth(np.std(stacked, axis=0).mean(axis=1), sw)
    ax.fill_between(time_axis[:len(total_std)], 0, total_std,
                    alpha=0.4, color="#ff7f0e", label="Rashomon σ")
    ax.plot(time_axis[:len(total_std)], total_std, lw=1.0, color="#ff7f0e")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Rashomon σ", fontsize=11)
    ax.set_title("Total Rashomon Uncertainty", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time", fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].tick_params(axis="x", rotation=30, labelsize=9)

    ss = args.start.replace("-", "")
    se = args.end.replace("-", "")
    out_zoom = RESULTS_DIR / f"multivariate_zoom_{target}_{ss}_{se}.png"
    fig.savefig(out_zoom, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_zoom}")

    # ===== PART 2: SHAP–Anomaly Correlation (full test set) =====
    print("\n--- SHAP–Anomaly Correlation (full test set) ---")

    # Use full test SHAP
    shap_full = shap_data
    time_full = test.index
    n_full = len(time_full)

    # Anomaly binary vector
    anom_vec = np.zeros(n_full, dtype=bool)
    if not labels.empty and target in labels.columns:
        ts_l = labels[target]
        tset = {t: i for i, t in enumerate(time_full)}
        for t in ts_l[ts_l > 0].index:
            if t in tset:
                anom_vec[tset[t]] = True
    n_anom = anom_vec.sum()
    n_norm = (~anom_vec).sum()
    print(f"Full test: {n_full} steps, anomaly={n_anom}, normal={n_norm}")

    # Per-feature analysis
    results = []
    for feat in features:
        # Mean SHAP across models
        vals_per_model = []
        for m in models:
            if feat in shap_full[m].columns:
                v = shap_full[m][feat].values[:n_full]
                vals_per_model.append(v)
        if len(vals_per_model) < 2:
            continue
        mn_len = min(len(v) for v in vals_per_model)
        stacked = np.array([v[:mn_len] for v in vals_per_model])

        mean_shap = np.mean(stacked, axis=0)
        abs_mean_shap = np.abs(mean_shap)
        rashomon_std = np.std(stacked, axis=0)

        anom_mask = anom_vec[:mn_len]

        # Mean |SHAP| during anomaly vs normal
        shap_anom = abs_mean_shap[anom_mask]
        shap_norm = abs_mean_shap[~anom_mask]

        # Rashomon std during anomaly vs normal
        rash_anom = rashomon_std[anom_mask]
        rash_norm = rashomon_std[~anom_mask]

        # t-test on |SHAP|
        t_shap, p_shap = stats.ttest_ind(shap_anom, shap_norm, equal_var=False)
        # t-test on Rashomon
        t_rash, p_rash = stats.ttest_ind(rash_anom, rash_norm, equal_var=False)

        # Cohen's d for |SHAP|
        pooled = np.sqrt((shap_anom.var() + shap_norm.var()) / 2)
        d_shap = (shap_anom.mean() - shap_norm.mean()) / pooled if pooled > 0 else 0

        # Point-biserial correlation
        r_pb, p_pb = stats.pointbiserialr(anom_mask, abs_mean_shap)

        ratio = shap_anom.mean() / shap_norm.mean() if shap_norm.mean() > 0 else np.inf

        results.append({
            "feature": feat,
            "shap_anom_mean": shap_anom.mean(),
            "shap_norm_mean": shap_norm.mean(),
            "ratio": ratio,
            "cohens_d": d_shap,
            "t_stat": t_shap,
            "p_value": p_shap,
            "point_biserial_r": r_pb,
            "pb_p_value": p_pb,
            "rashomon_anom": rash_anom.mean(),
            "rashomon_norm": rash_norm.mean(),
            "rashomon_p": p_rash,
        })

    res_df = pd.DataFrame(results).sort_values("p_value")
    res_df.to_csv(RESULTS_DIR / f"shap_anomaly_corr_{target}.csv", index=False)

    print("\nPer-feature |SHAP| — Anomaly vs Normal:")
    print(f"{'Feature':>8s}  {'Anom':>8s}  {'Norm':>8s}  {'Ratio':>6s}  {'d':>6s}  {'p':>10s}  {'r_pb':>7s}")
    for _, r in res_df.iterrows():
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "ns"
        print(f"{r.feature:>8s}  {r.shap_anom_mean:>8.4f}  {r.shap_norm_mean:>8.4f}  "
              f"{r.ratio:>5.2f}x  {r.cohens_d:>6.2f}  {r.p_value:>9.1e} {sig}  {r.point_biserial_r:>7.3f}")

    # ===== Correlation Plots =====
    sig_feats = res_df[res_df.p_value < 0.05]["feature"].tolist()
    all_feats = res_df["feature"].tolist()

    # Plot 1: Box plot — |SHAP| anomaly vs normal for each feature
    fig, ax = plt.subplots(figsize=(14, 6))
    box_data = []
    box_labels = []
    box_colors = []
    for feat in all_feats:
        vals_per_model = []
        for m in models:
            if feat in shap_full[m].columns:
                vals_per_model.append(shap_full[m][feat].values[:n_full])
        if len(vals_per_model) < 2:
            continue
        mn_len = min(len(v) for v in vals_per_model)
        mean_abs = np.abs(np.mean(np.array([v[:mn_len] for v in vals_per_model]), axis=0))
        anom_mask = anom_vec[:mn_len]
        box_data.append(mean_abs[anom_mask])
        box_data.append(mean_abs[~anom_mask])
        box_labels.append(f"{feat}\nanom")
        box_labels.append(f"{feat}\nnorm")

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#d62728" if i % 2 == 0 else "#1f77b4")
        patch.set_alpha(0.6)
    ax.set_ylabel("|SHAP| value", fontsize=11)
    ax.set_title(f"{target} — |SHAP| Distribution: Anomaly (red) vs Normal (blue)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    out1 = RESULTS_DIR / f"shap_boxplot_{target}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # Plot 2: Bar chart — Cohen's d per feature
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = ["#d62728" if p < 0.05 else "#aaaaaa" for p in res_df.p_value]
    ax.barh(res_df.feature, res_df.cohens_d, color=colors_bar, alpha=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(0.2, color="gray", lw=0.5, ls="--", label="Small effect (0.2)")
    ax.axvline(-0.2, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Cohen's d (positive = higher |SHAP| during anomaly)", fontsize=10)
    ax.set_title(f"{target} — Effect Size: |SHAP| Anomaly vs Normal (red = p<0.05)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    out2 = RESULTS_DIR / f"shap_effect_size_{target}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # Plot 3: Rolling correlation between |SHAP| and anomaly indicator
    roll_w = 288  # 1 day
    fig, axes_r = plt.subplots(len(all_feats), 1,
                                figsize=(22, 2 * len(all_feats)),
                                sharex=True)
    if len(all_feats) == 1:
        axes_r = [axes_r]

    for fi, feat in enumerate(all_feats):
        ax = axes_r[fi]
        vals_per_model = []
        for m in models:
            if feat in shap_full[m].columns:
                vals_per_model.append(shap_full[m][feat].values[:n_full])
        if len(vals_per_model) < 2:
            continue
        mn_len = min(len(v) for v in vals_per_model)
        mean_abs = np.abs(np.mean(np.array([v[:mn_len] for v in vals_per_model]), axis=0))
        anom_ind = anom_vec[:mn_len].astype(float)

        # Rolling correlation
        s_shap = pd.Series(mean_abs, index=time_full[:mn_len])
        s_anom = pd.Series(anom_ind, index=time_full[:mn_len])
        roll_corr = s_shap.rolling(roll_w, min_periods=50).corr(s_anom)

        c = COLORS[fi % len(COLORS)]
        ax.plot(time_full[:mn_len], roll_corr.values, lw=1.0, color=c)
        ax.axhline(0, color="gray", lw=0.5, ls="--")

        # Anomaly shading
        at, ast, aen = get_anomaly_regions(labels, target, time_full[:mn_len])
        shade_anom(ax, time_full[:mn_len], ast, aen, label=(fi == 0))

        ax.set_ylabel(feat, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.6, 0.6)
        ax.grid(True, alpha=0.15)

    axes_r[0].set_title(f"{target} — Rolling Correlation (w={roll_w}): |SHAP| vs Anomaly",
                         fontsize=12, fontweight="bold")
    axes_r[-1].set_xlabel("Time", fontsize=11)
    axes_r[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes_r[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes_r[-1].tick_params(axis="x", rotation=30, labelsize=9)
    plt.tight_layout()
    out3 = RESULTS_DIR / f"shap_rolling_corr_{target}.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    sig_pos = res_df[(res_df.p_value < 0.05) & (res_df.cohens_d > 0)]
    sig_neg = res_df[(res_df.p_value < 0.05) & (res_df.cohens_d < 0)]
    print(f"Features with significantly HIGHER |SHAP| during anomaly ({len(sig_pos)}):")
    for _, r in sig_pos.iterrows():
        print(f"  {r.feature}: d={r.cohens_d:.2f}, ratio={r.ratio:.2f}x, r_pb={r.point_biserial_r:.3f}")
    print(f"Features with significantly LOWER |SHAP| during anomaly ({len(sig_neg)}):")
    for _, r in sig_neg.iterrows():
        print(f"  {r.feature}: d={r.cohens_d:.2f}, ratio={r.ratio:.2f}x, r_pb={r.point_biserial_r:.3f}")
    ns = res_df[res_df.p_value >= 0.05]
    print(f"Not significant ({len(ns)}): {', '.join(ns.feature.tolist())}")
    print("\nDone!")


if __name__ == "__main__":
    main()
