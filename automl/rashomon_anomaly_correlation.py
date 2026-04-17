"""
Rashomon Uncertainty vs Anomaly Correlation Analysis
=====================================================
Quantifies whether inter-model SHAP disagreement (Rashomon uncertainty)
is systematically higher during anomaly periods.

Analyses:
  1. Per-feature: mean uncertainty at anomaly vs normal timesteps + t-test
  2. ROC-AUC: Can Rashomon uncertainty detect anomalies?
  3. Rolling correlation between uncertainty and anomaly indicator
  4. Visualization: box plots, ROC curves, temporal correlation

Usage:
    python rashomon_anomaly_correlation.py
    python rashomon_anomaly_correlation.py --series TS1
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

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
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "rashomon_anomaly_corr"

MODELS = ["DirectTabular", "RecursiveTabular", "WeightedEnsemble"]

FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


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


def load_shap():
    data = {}
    for p in sorted(SHAP_DIR.glob("shap_values_*.csv")):
        m = p.stem.replace("shap_values_", "")
        if m not in MODELS:
            continue
        data[m] = pd.read_csv(p)
    return data


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--series", default="TS1")
    pa.add_argument("--top_n", type=int, default=10)
    args = pa.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sid = args.series

    print("=" * 60)
    print(f"Rashomon Uncertainty vs Anomaly — {sid}")
    print("=" * 60)

    full_data, test_start_idx = load_data()
    test_ts = full_data.index[test_start_idx:]
    labels = load_labels()
    shap_data = load_shap()
    models = list(shap_data.keys())
    print(f"Models: {models}")

    meta = {"_series", "_step"}

    # Get feature columns
    first_df = next(iter(shap_data.values()))
    sdf0 = first_df[first_df["_series"] == sid].sort_values("_step").reset_index(drop=True)
    all_feats = [c for c in sdf0.columns if c not in meta]
    n_steps = len(sdf0)

    # Stack SHAP: (n_models, n_steps, n_features)
    model_stack = []
    for m in models:
        df = shap_data[m]
        sdf = df[df["_series"] == sid].sort_values("_step").reset_index(drop=True)
        model_stack.append(sdf[all_feats].values[:n_steps])
    model_stack = np.array(model_stack)

    # Inter-model std per feature per timestep
    inter_std = np.std(model_stack, axis=0)  # (T, F)
    inter_std_df = pd.DataFrame(inter_std, columns=all_feats)

    # Total uncertainty = mean across features
    total_unc = np.mean(inter_std, axis=1)

    # Anomaly indicator aligned to test timestamps
    time_axis = test_ts[:n_steps]
    anom_indicator = np.zeros(n_steps, dtype=int)
    if not labels.empty and sid in labels.columns:
        ts_l = labels[sid]
        tset = {t: i for i, t in enumerate(time_axis)}
        for t in ts_l[ts_l > 0].index:
            if t in tset:
                anom_indicator[tset[t]] = 1

    n_anom = anom_indicator.sum()
    n_norm = n_steps - n_anom
    print(f"Timesteps: {n_steps} (anomaly={n_anom}, normal={n_norm})")

    # Rank features by importance
    imps = []
    for m in models:
        df = shap_data[m]
        sdf = df[df["_series"] == sid]
        imps.append(sdf[all_feats].abs().mean())
    combined = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined.sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"Top {args.top_n}: {top_feats}")

    # =====================================================================
    # Analysis 1: Per-feature uncertainty at anomaly vs normal
    # =====================================================================
    print("\n" + "=" * 60)
    print("1. Per-Feature Uncertainty: Anomaly vs Normal")
    print("=" * 60)

    results = []
    for feat in top_feats:
        fi = all_feats.index(feat)
        unc_anom = inter_std[anom_indicator == 1, fi]
        unc_norm = inter_std[anom_indicator == 0, fi]

        mean_a = np.mean(unc_anom)
        mean_n = np.mean(unc_norm)
        ratio = mean_a / mean_n if mean_n > 1e-10 else np.inf

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(unc_anom, unc_norm, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(unc_anom) + np.var(unc_norm)) / 2)
        cohens_d = (mean_a - mean_n) / pooled_std if pooled_std > 1e-10 else 0

        # Point-biserial correlation
        r_pb, p_pb = stats.pointbiserialr(anom_indicator, inter_std[:, fi])

        results.append({
            "feature": feat,
            "mean_anomaly": mean_a,
            "mean_normal": mean_n,
            "ratio": ratio,
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "r_pointbiserial": r_pb,
            "p_pointbiserial": p_pb,
        })

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {feat:20s}  anom={mean_a:.4f}  norm={mean_n:.4f}  "
              f"ratio={ratio:.2f}x  d={cohens_d:.2f}  p={p_val:.1e} {sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / f"uncertainty_vs_anomaly_{sid}.csv", index=False)

    # Total uncertainty
    unc_anom_tot = total_unc[anom_indicator == 1]
    unc_norm_tot = total_unc[anom_indicator == 0]
    t_tot, p_tot = stats.ttest_ind(unc_anom_tot, unc_norm_tot, equal_var=False)
    print(f"\n  {'TOTAL':20s}  anom={unc_anom_tot.mean():.4f}  norm={unc_norm_tot.mean():.4f}  "
          f"ratio={unc_anom_tot.mean()/unc_norm_tot.mean():.2f}x  p={p_tot:.1e}")

    # =====================================================================
    # Analysis 2: ROC-AUC — can uncertainty detect anomalies?
    # =====================================================================
    print("\n" + "=" * 60)
    print("2. ROC-AUC: Uncertainty as Anomaly Detector")
    print("=" * 60)

    auc_results = []
    for feat in top_feats:
        fi = all_feats.index(feat)
        try:
            auc = roc_auc_score(anom_indicator, inter_std[:, fi])
        except ValueError:
            auc = 0.5
        auc_results.append({"feature": feat, "AUC": auc})
        print(f"  {feat:20s}  AUC={auc:.4f}")

    # Total
    try:
        auc_total = roc_auc_score(anom_indicator, total_unc)
    except ValueError:
        auc_total = 0.5
    print(f"  {'TOTAL':20s}  AUC={auc_total:.4f}")
    auc_results.append({"feature": "TOTAL", "AUC": auc_total})

    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(RESULTS_DIR / f"auc_scores_{sid}.csv", index=False)

    # =====================================================================
    # Plot 1: Box plots — uncertainty at anomaly vs normal
    # =====================================================================
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=False)
    axes = axes.flatten()

    for i, feat in enumerate(top_feats[:10]):
        ax = axes[i]
        fi = all_feats.index(feat)
        data_box = [inter_std[anom_indicator == 0, fi],
                     inter_std[anom_indicator == 1, fi]]
        bp = ax.boxplot(data_box, labels=["Normal", "Anomaly"],
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor("#a8d5e2")
        bp["boxes"][1].set_facecolor("#ff6b6b")

        p = results_df[results_df["feature"] == feat]["p_value"].values[0]
        d = results_df[results_df["feature"] == feat]["cohens_d"].values[0]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{feat}\nd={d:.2f} {sig}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Inter-model σ" if i % 5 == 0 else "", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"{sid} — Rashomon Uncertainty: Anomaly vs Normal\n"
                 f"(d = Cohen's d effect size, * = p<0.05, ** = p<0.01, *** = p<0.001)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"boxplot_{sid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: boxplot_{sid}.png")

    # =====================================================================
    # Plot 2: AUC bar chart
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    auc_sorted = auc_df.sort_values("AUC", ascending=True)
    colors = ["#ff6b6b" if f == "TOTAL" else "#4363d8" for f in auc_sorted["feature"]]
    ax.barh(auc_sorted["feature"], auc_sorted["AUC"], color=colors, edgecolor="white")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="Random (AUC=0.5)")
    for i, (_, row) in enumerate(auc_sorted.iterrows()):
        ax.text(row["AUC"] + 0.005, i, f"{row['AUC']:.3f}", va="center", fontsize=9)
    ax.set_xlabel("ROC-AUC", fontsize=11)
    ax.set_title(f"{sid} — Can Rashomon Uncertainty Detect Anomalies?\n"
                 f"(AUC > 0.5 = uncertainty higher during anomalies)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.4, max(auc_df["AUC"].max() + 0.05, 0.7))
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"auc_bar_{sid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: auc_bar_{sid}.png")

    # =====================================================================
    # Plot 3: ROC curves for top features + total
    # =====================================================================
    fig, ax = plt.subplots(figsize=(9, 8))
    # Total
    fpr, tpr, _ = roc_curve(anom_indicator, total_unc)
    ax.plot(fpr, tpr, lw=2.5, color="black", label=f"TOTAL (AUC={auc_total:.3f})")
    # Per feature
    for i, feat in enumerate(top_feats[:6]):
        fi = all_feats.index(feat)
        auc_val = auc_df[auc_df["feature"] == feat]["AUC"].values[0]
        fpr, tpr, _ = roc_curve(anom_indicator, inter_std[:, fi])
        ax.plot(fpr, tpr, lw=1.5, color=FEATURE_COLORS[i], alpha=0.8,
                label=f"{feat} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{sid} — ROC: Rashomon Uncertainty as Anomaly Detector",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"roc_{sid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: roc_{sid}.png")

    # =====================================================================
    # Plot 4: Rolling correlation (uncertainty vs anomaly over time)
    # =====================================================================
    window = 288  # 1 day rolling window
    anom_s = pd.Series(anom_indicator, index=time_axis)
    unc_s = pd.Series(total_unc, index=time_axis)

    roll_corr = anom_s.rolling(window, min_periods=50).corr(unc_s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Top: time series + anomalies
    test_vals = full_data[sid].values[test_start_idx:test_start_idx + n_steps]
    ax1.plot(time_axis, test_vals, color="steelblue", lw=0.7, alpha=0.8, label=sid)
    # Anomaly shading
    anom_arr = anom_indicator.astype(bool)
    if anom_arr.any():
        ch = np.diff(anom_arr.astype(int))
        starts = np.where(ch == 1)[0] + 1
        ends = np.where(ch == -1)[0] + 1
        if anom_arr[0]:
            starts = np.insert(starts, 0, 0)
        if anom_arr[-1]:
            ends = np.append(ends, len(anom_arr))
        for j, (s, e) in enumerate(zip(starts, ends)):
            ax1.axvspan(time_axis[s], time_axis[min(e, len(time_axis)-1)],
                        alpha=0.2, color="black",
                        label="Anomaly" if j == 0 else None)
    ax1.set_ylabel(sid, fontsize=11)
    ax1.set_title(f"{sid} — Rolling Correlation: Rashomon Uncertainty ↔ Anomaly (window={window})",
                  fontweight="bold", fontsize=13)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.2)

    # Bottom: rolling correlation
    ax2.plot(time_axis, roll_corr.values, color="crimson", lw=1.2)
    ax2.fill_between(time_axis, 0, roll_corr.values,
                     where=roll_corr.values > 0, alpha=0.3, color="crimson")
    ax2.fill_between(time_axis, 0, roll_corr.values,
                     where=roll_corr.values < 0, alpha=0.3, color="steelblue")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    # Anomaly shading
    if anom_arr.any():
        for j, (s, e) in enumerate(zip(starts, ends)):
            ax2.axvspan(time_axis[s], time_axis[min(e, len(time_axis)-1)],
                        alpha=0.12, color="black")
    ax2.set_ylabel("Rolling Pearson r", fontsize=11)
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylim(-0.5, 0.8)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax2.tick_params(axis="x", rotation=30, labelsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"rolling_corr_{sid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: rolling_corr_{sid}.png")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    sig_feats = results_df[results_df["p_value"] < 0.05]
    print(f"Features with significantly higher uncertainty during anomalies: "
          f"{len(sig_feats)}/{len(results_df)}")
    if len(sig_feats) > 0:
        print(sig_feats[["feature", "ratio", "cohens_d", "p_value"]].to_string(index=False))
    print(f"\nTotal uncertainty AUC: {auc_total:.4f}")
    best = auc_df[auc_df["feature"] != "TOTAL"].sort_values("AUC", ascending=False).iloc[0]
    print(f"Best single feature AUC: {best['feature']} = {best['AUC']:.4f}")
    print(f"\nOutput: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
