"""
SHAP–Anomaly Analysis (10 Models Combined)
============================================
Loads all 10 model SHAP CSVs (sklearn + H2O) and analyzes:
1. Per-feature |SHAP| anomaly vs normal (t-test, Cohen's d, point-biserial)
2. Per-feature Rashomon σ anomaly vs normal
3. Anomaly classification using SHAP, uncertainty, and combined
4. Visualizations: box plots, effect sizes, ROC/PR curves

Usage:
    python shap_anomaly_analysis_10models.py
"""
from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
DIVERSE_DIR = BASE_DIR / "results" / "diverse"
H2O_DIR     = BASE_DIR / "results" / "h2o"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_10models"

TARGET = "TS1"

MODEL_SHORT = {
    "LightGBM": "LightGBM", "XGBoost": "XGBoost",
    "RandomForest": "RandomForest", "Ridge": "Ridge",
    "DecisionTree": "DecisionTree",
    "GBM_H2O": "GBM(H2O)", "DRF_H2O": "DRF(H2O)",
    "XRT_H2O": "XRT(H2O)", "DeepLearning_H2O": "DL(H2O)",
    "GLM_H2O": "GLM(H2O)",
}


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


def load_test():
    d = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",
                    parse_dates=["time"], index_col="time")
    d.index = d.index.tz_localize(None)
    return d


def load_all_shap():
    data = {}
    for p in sorted(DIVERSE_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace("shap_", "").replace(f"_{TARGET}", "")
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        df.index = df.index.tz_localize(None)
        data[name] = df
    for p in sorted(H2O_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace("shap_", "").replace(f"_{TARGET}", "")
        data[name + "_H2O"] = pd.read_csv(p, parse_dates=["time"], index_col="time")
        data[name + "_H2O"].index = data[name + "_H2O"].index.tz_localize(None)
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


def train_clf(X_train, X_test, y_train, y_test, name):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    n_pos = max(y_train.sum(), 1)
    scale_pos = (len(y_train) - n_pos) / n_pos

    ds_tr = lgb.Dataset(Xtr, label=y_train.values)
    ds_te = lgb.Dataset(Xte, label=y_test.values, reference=ds_tr)
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "verbosity": -1,
         "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
         "scale_pos_weight": scale_pos, "min_child_samples": 20},
        ds_tr, num_boost_round=500, valid_sets=[ds_te],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_prob = model.predict(Xte)
    thrs = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thrs]
    best_thr = thrs[np.argmax(f1s)]
    y_pred = (y_prob >= best_thr).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    prec = cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
    rec = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0

    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    return {"name": name, "auc": auc, "ap": ap, "f1": max(f1s),
            "precision": prec, "recall": rec, "threshold": best_thr,
            "cm": cm, "y_prob": y_prob, "y_pred": y_pred, "importance": imp}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHAP–Anomaly Analysis (10 models)")
    print("=" * 60)

    test = load_test()
    labels = load_labels()
    shap_data = load_all_shap()
    model_names = list(shap_data.keys())
    features = [c for c in test.columns if c != TARGET]
    n = len(test)

    print(f"Models ({len(model_names)}): {model_names}")
    print(f"Features: {features}")

    # Anomaly vector
    anom_vec = np.zeros(n, dtype=bool)
    if TARGET in labels.columns:
        ts_l = labels[TARGET]
        tset = {t: i for i, t in enumerate(test.index)}
        for t in ts_l[ts_l > 0].index:
            if t in tset:
                anom_vec[tset[t]] = True
    n_anom = anom_vec.sum()
    print(f"Test: {n} steps, anomaly={n_anom}, normal={n - n_anom}")

    # ── Stack SHAP values ────────────────────────────────────────
    # (n_models, n_timesteps, n_features)
    stacked = np.stack([shap_data[m][features].values[:n] for m in model_names])

    mean_shap = np.mean(stacked, axis=0)        # (n, 11)
    abs_mean_shap = np.abs(mean_shap)            # (n, 11)
    rashomon_std = np.std(stacked, axis=0)       # (n, 11)

    # ── PART 1: Per-feature statistics ───────────────────────────
    print("\n" + "=" * 60)
    print("1. Per-Feature |SHAP|: Anomaly vs Normal")
    print("=" * 60)
    print(f"{'Feature':>5s}  {'Anom':>8s}  {'Norm':>8s}  {'Ratio':>6s}  "
          f"{'d':>6s}  {'p':>10s}  {'r_pb':>6s}")

    results = []
    for fi, feat in enumerate(features):
        sa = abs_mean_shap[anom_vec, fi]
        sn = abs_mean_shap[~anom_vec, fi]
        t, p = stats.ttest_ind(sa, sn, equal_var=False)
        pooled = np.sqrt((sa.var() + sn.var()) / 2)
        d = (sa.mean() - sn.mean()) / pooled if pooled > 0 else 0
        r_pb, p_pb = stats.pointbiserialr(anom_vec, abs_mean_shap[:, fi])
        ratio = sa.mean() / sn.mean() if sn.mean() > 0 else np.inf

        # Rashomon
        ra = rashomon_std[anom_vec, fi]
        rn = rashomon_std[~anom_vec, fi]
        t_r, p_r = stats.ttest_ind(ra, rn, equal_var=False)
        pooled_r = np.sqrt((ra.var() + rn.var()) / 2)
        d_r = (ra.mean() - rn.mean()) / pooled_r if pooled_r > 0 else 0

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{feat:>5s}  {sa.mean():>8.4f}  {sn.mean():>8.4f}  "
              f"{ratio:>5.2f}x  {d:>6.2f}  {p:>9.1e} {sig}  {r_pb:>6.3f}")

        results.append({
            "feature": feat,
            "shap_anom": sa.mean(), "shap_norm": sn.mean(),
            "ratio": ratio, "cohens_d": d, "p_value": p, "r_pb": r_pb,
            "rash_anom": ra.mean(), "rash_norm": rn.mean(),
            "rash_d": d_r, "rash_p": p_r,
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "per_feature_stats.csv", index=False)

    print(f"\n{'Feature':>5s}  {'Rash_A':>8s}  {'Rash_N':>8s}  "
          f"{'d':>6s}  {'p':>10s}")
    for _, r in res_df.iterrows():
        sig = "***" if r.rash_p < 0.001 else "**" if r.rash_p < 0.01 else "*" if r.rash_p < 0.05 else "ns"
        print(f"{r.feature:>5s}  {r.rash_anom:>8.4f}  {r.rash_norm:>8.4f}  "
              f"{r.rash_d:>6.2f}  {r.rash_p:>9.1e} {sig}")

    # ── PART 2: Classification ───────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Anomaly Classification")
    print("=" * 60)

    # Build feature sets
    # A) Per-model |SHAP| (10×11 = 110)
    shap_feats = pd.DataFrame(index=test.index)
    for m in model_names:
        short = MODEL_SHORT.get(m, m)
        for feat in features:
            shap_feats[f"|shap|_{short}_{feat}"] = shap_data[m][feat].abs().values[:n]

    # B) Mean |SHAP| (11)
    mean_feats = pd.DataFrame(abs_mean_shap, index=test.index,
                               columns=[f"mean_|shap|_{f}" for f in features])

    # C) Rashomon σ (11) + total (1)
    rash_feats = pd.DataFrame(rashomon_std, index=test.index,
                               columns=[f"rashomon_σ_{f}" for f in features])
    rash_feats["total_rashomon_σ"] = rashomon_std.mean(axis=1)

    # Configs
    configs = {
        "All SHAP (110)":         shap_feats,
        "Mean |SHAP| (11)":       mean_feats,
        "Rashomon σ (12)":        rash_feats,
        "Mean + Rashomon (23)":   pd.concat([mean_feats, rash_feats], axis=1),
        "All Combined (133)":     pd.concat([shap_feats, rash_feats], axis=1),
    }

    y = labels[TARGET].reindex(test.index).fillna(0).astype(int)
    y = (y > 0).astype(int)
    split = int(n * 0.6)
    y_train, y_test_y = y.iloc[:split], y.iloc[split:]

    print(f"Train: {split} (anom={y_train.sum()}), Test: {n-split} (anom={y_test_y.sum()})")

    clf_results = []
    for name, X in configs.items():
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        r = train_clf(Xtr, Xte, y_train, y_test_y, name)
        clf_results.append(r)
        print(f"\n  {name}:")
        print(f"    AUC={r['auc']:.4f}  AP={r['ap']:.4f}  F1={r['f1']:.4f}  "
              f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}")

    # Comparison table
    comp = pd.DataFrame([{
        "Model": r["name"], "ROC-AUC": r["auc"], "PR-AUC": r["ap"],
        "F1": r["f1"], "Precision": r["precision"], "Recall": r["recall"],
    } for r in clf_results])
    comp.to_csv(RESULTS_DIR / "classification_comparison.csv", index=False)
    print("\n" + comp.to_string(index=False))

    # ── PLOTS ────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Plot 1: Effect size bar chart (|SHAP| and Rashomon σ side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    res_sorted = res_df.sort_values("cohens_d")
    colors1 = ["#d62728" if p < 0.05 else "#aaaaaa" for p in res_sorted.p_value]
    ax1.barh(res_sorted.feature, res_sorted.cohens_d, color=colors1, alpha=0.8)
    ax1.axvline(0, color="black", lw=0.5)
    ax1.axvline(0.2, color="gray", lw=0.5, ls="--")
    ax1.axvline(-0.2, color="gray", lw=0.5, ls="--")
    ax1.set_xlabel("Cohen's d")
    ax1.set_title("|SHAP| Effect Size\n(+) = higher during anomaly", fontweight="bold")
    ax1.grid(True, alpha=0.15, axis="x")

    res_sorted2 = res_df.sort_values("rash_d")
    colors2 = ["#ff7f0e" if p < 0.05 else "#aaaaaa" for p in res_sorted2.rash_p]
    ax2.barh(res_sorted2.feature, res_sorted2.rash_d, color=colors2, alpha=0.8)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.axvline(0.2, color="gray", lw=0.5, ls="--")
    ax2.axvline(-0.2, color="gray", lw=0.5, ls="--")
    ax2.set_xlabel("Cohen's d")
    ax2.set_title("Rashomon σ Effect Size\n(+) = more disagreement during anomaly", fontweight="bold")
    ax2.grid(True, alpha=0.15, axis="x")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Box plots — top features |SHAP| anomaly vs normal
    top5 = res_df.sort_values("p_value").head(5).feature.tolist()
    fig, axes_b = plt.subplots(1, len(top5), figsize=(4 * len(top5), 5))
    for i, feat in enumerate(top5):
        fi = features.index(feat)
        ax = axes_b[i]
        data_a = abs_mean_shap[anom_vec, fi]
        data_n = abs_mean_shap[~anom_vec, fi]
        bp = ax.boxplot([data_a, data_n], labels=["Anomaly", "Normal"],
                        patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("#d62728")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#1f77b4")
        bp["boxes"][1].set_alpha(0.6)
        d_val = res_df[res_df.feature == feat].cohens_d.values[0]
        p_val = res_df[res_df.feature == feat].p_value.values[0]
        ax.set_title(f"{feat}\nd={d_val:.2f}, p={p_val:.1e}", fontsize=10)
        ax.set_ylabel("|SHAP|" if i == 0 else "")
        ax.grid(True, alpha=0.15)
    plt.suptitle("Top 5 Features: |SHAP| Anomaly vs Normal (10 models)", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "boxplots_top5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: ROC + PR comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors_clf = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for i, r in enumerate(clf_results):
        c = colors_clf[i % len(colors_clf)]
        fpr, tpr, _ = roc_curve(y_test_y, r["y_prob"])
        ax1.plot(fpr, tpr, lw=2, color=c, label=f"{r['name']} ({r['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test_y, r["y_prob"])
        ax2.plot(rec, prec, lw=2, color=c, label=f"{r['name']} ({r['ap']:.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color="gray")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curves", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)

    ax2.axhline(y_test_y.mean(), ls="--", color="gray")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_pr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Metrics bar comparison
    fig, axes_m = plt.subplots(1, 3, figsize=(16, 5))
    for i, (metric, key) in enumerate([("ROC-AUC", "auc"), ("PR-AUC", "ap"), ("F1", "f1")]):
        ax = axes_m[i]
        vals = [r[key] for r in clf_results]
        names = [r["name"].split("(")[0].strip() for r in clf_results]
        bars = ax.bar(range(len(vals)), vals, color=colors_clf[:len(vals)], alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=7, rotation=20)
        ax.set_title(metric, fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.15, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: Best model feature importance
    best = max(clf_results, key=lambda r: r["f1"])
    imp = best["importance"].head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = []
    for f in imp.feature:
        if "rashomon" in f:
            bar_colors.append("#ff7f0e")
        elif "mean_|shap|" in f:
            bar_colors.append("#2ca02c")
        else:
            bar_colors.append("#1f77b4")
    ax.barh(range(len(imp)), imp.importance.values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.feature.values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Best Model ({best['name']}) — Top 20 Features", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1f77b4", label="Per-model |SHAP|"),
        Patch(facecolor="#2ca02c", label="Mean |SHAP|"),
        Patch(facecolor="#ff7f0e", label="Rashomon σ"),
    ], fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "best_model_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
