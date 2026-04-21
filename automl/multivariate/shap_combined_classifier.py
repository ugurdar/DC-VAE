"""
Combined SHAP + Uncertainty Anomaly Classifier
===============================================
Compares 3 approaches side by side:
  1) SHAP only (|SHAP| features)
  2) Uncertainty only (Rashomon σ)
  3) Combined (SHAP + Uncertainty)

Usage:
    python shap_combined_classifier.py
"""
from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
import lightgbm as lgb

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
SHAP_DIR    = BASE_DIR / "results"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_clf_combined"

MODELS = ["WeightedEnsemble_L2", "WeightedEnsemble_L3", "LightGBMXT_BAG_L2"]
TARGET = "TS1"


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


def load_shap():
    data = {}
    for p in sorted(SHAP_DIR.glob(f"shap_*_{TARGET}.csv")):
        m = p.stem.replace("shap_", "").replace(f"_{TARGET}", "")
        if m in MODELS:
            df = pd.read_csv(p, parse_dates=["time"], index_col="time")
            df.index = df.index.tz_localize(None)
            data[m] = df
    return data


def build_all_features(shap_data):
    models = list(shap_data.keys())
    idx = shap_data[models[0]].index
    ts_feats = shap_data[models[0]].columns.tolist()
    stacked = np.stack([shap_data[m].values for m in models])

    # --- SHAP features ---
    # Mean |SHAP| across models per feature
    mean_abs = pd.DataFrame(
        np.mean(np.abs(stacked), axis=0), index=idx,
        columns=[f"mean_|shap|_{c}" for c in ts_feats]
    )
    # Per-model total |SHAP|
    total_shap_parts = []
    for m in models:
        total = shap_data[m].abs().sum(axis=1)
        total_shap_parts.append(total.to_frame(f"total_|shap|_{m}"))

    # --- Uncertainty features ---
    # Rashomon σ per feature
    rash_std = pd.DataFrame(
        np.std(stacked, axis=0), index=idx,
        columns=[f"rashomon_σ_{c}" for c in ts_feats]
    )
    # Total Rashomon σ
    total_rash = pd.Series(
        np.std(stacked, axis=0).mean(axis=1),
        index=idx, name="total_rashomon_σ"
    )

    # Build 3 feature sets
    X_shap = pd.concat([mean_abs] + total_shap_parts, axis=1)
    X_unc = pd.concat([rash_std, total_rash.to_frame()], axis=1)
    X_combined = pd.concat([X_shap, X_unc], axis=1)

    return X_shap, X_unc, X_combined


def train_and_evaluate(X_train, X_test, y_train, y_test, name):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos if n_pos > 0 else 1

    ds_train = lgb.Dataset(Xtr, label=y_train.values)
    ds_val = lgb.Dataset(Xte, label=y_test.values, reference=ds_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "scale_pos_weight": scale_pos,
        "min_child_samples": 20,
    }
    model = lgb.train(
        params, ds_train, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_prob = model.predict(Xte)

    # Best threshold
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_thr = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)
    y_pred = (y_prob >= best_thr).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    prec_val = cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
    rec_val = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0

    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    return {
        "name": name,
        "model": model,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "auc": auc,
        "ap": ap,
        "f1": best_f1,
        "threshold": best_thr,
        "precision": prec_val,
        "recall": rec_val,
        "cm": cm,
        "importance": imp,
        "n_features": X_train.shape[1],
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combined SHAP + Uncertainty Classifier")
    print("=" * 60)

    shap_data = load_shap()
    models = list(shap_data.keys())
    print(f"Models: {models}")

    labels = load_labels()

    X_shap, X_unc, X_combined = build_all_features(shap_data)
    print(f"SHAP features:       {X_shap.shape[1]}")
    print(f"Uncertainty features: {X_unc.shape[1]}")
    print(f"Combined features:   {X_combined.shape[1]}")

    # Labels
    y = labels[TARGET].reindex(X_combined.index).fillna(0).astype(int)
    y = (y > 0).astype(int)
    print(f"Class: {y.value_counts().to_dict()}")

    # Temporal split
    n = len(X_combined)
    split_idx = int(n * 0.6)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train: {split_idx} (anom={y_train.sum()}), Test: {n - split_idx} (anom={y_test.sum()})")

    # Train all 3
    print("\n--- Training 3 models ---")
    results = []
    for name, X in [("SHAP Only", X_shap), ("Uncertainty Only", X_unc), ("SHAP + Uncertainty", X_combined)]:
        Xtr, Xte = X.iloc[:split_idx], X.iloc[split_idx:]
        r = train_and_evaluate(Xtr, Xte, y_train, y_test, name)
        results.append(r)
        print(f"\n  {name} ({r['n_features']} features):")
        print(f"    ROC-AUC={r['auc']:.4f}  PR-AUC={r['ap']:.4f}  F1={r['f1']:.4f}")
        print(f"    Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  Thr={r['threshold']:.2f}")
        print(f"    CM: TP={r['cm'][1,1]} FP={r['cm'][0,1]} FN={r['cm'][1,0]} TN={r['cm'][0,0]}")

    # ===== Comparison Table =====
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    comp = pd.DataFrame([{
        "Model": r["name"],
        "Features": r["n_features"],
        "ROC-AUC": r["auc"],
        "PR-AUC": r["ap"],
        "F1": r["f1"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "TP": r["cm"][1,1],
        "FP": r["cm"][0,1],
        "FN": r["cm"][1,0],
    } for r in results])
    print(comp.to_string(index=False))
    comp.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    # ===== Plots =====
    time_test = X_combined.iloc[split_idx:].index
    colors = {"SHAP Only": "#1f77b4", "Uncertainty Only": "#ff7f0e", "SHAP + Uncertainty": "#2ca02c"}

    # 1. ROC comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for r in results:
        c = colors[r["name"]]
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax1.plot(fpr, tpr, lw=2, color=c,
                 label=f"{r['name']} (AUC={r['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color="gray")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curves — Comparison", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    for r in results:
        c = colors[r["name"]]
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax2.plot(rec, prec, lw=2, color=c,
                 label=f"{r['name']} (AP={r['ap']:.3f})")
    ax2.axhline(y_test.mean(), ls="--", color="gray", label=f"Baseline={y_test.mean():.3f}")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curves — Comparison", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_pr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_map = {"ROC-AUC": "auc", "PR-AUC": "ap", "F1": "f1"}
    for i, (metric, key) in enumerate(metrics_map.items()):
        ax = axes[i]
        vals = [r[key] for r in results]

        bars = ax.bar(range(3), vals,
                      color=[colors[r["name"]] for r in results], alpha=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([r["name"] for r in results], fontsize=9, rotation=15)
        ax.set_title(metric, fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.15, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Combined model — feature importance
    r_comb = results[2]  # SHAP + Uncertainty
    imp = r_comb["importance"]
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = imp.head(20)
    bar_colors = []
    for f in top20.feature:
        if "rashomon" in f:
            bar_colors.append("#ff7f0e")
        elif "total_|shap|" in f:
            bar_colors.append("#9467bd")
        else:
            bar_colors.append("#1f77b4")
    ax.barh(range(len(top20)), top20.importance.values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.feature.values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Combined Model — Top 20 Features", fontweight="bold")
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Mean |SHAP|"),
        Patch(facecolor="#ff7f0e", label="Rashomon σ"),
        Patch(facecolor="#9467bd", label="Total |SHAP|"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "combined_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Temporal predictions — all 3 models
    fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True)
    anom_mask = y_test.values.astype(bool)
    starts_a, ends_a = np.array([], dtype=int), np.array([], dtype=int)
    if anom_mask.any():
        ch = np.diff(anom_mask.astype(int))
        starts_a = np.where(ch == 1)[0] + 1
        ends_a = np.where(ch == -1)[0] + 1
        if anom_mask[0]: starts_a = np.insert(starts_a, 0, 0)
        if anom_mask[-1]: ends_a = np.append(ends_a, len(anom_mask))

    for ri, r in enumerate(results):
        ax = axes[ri]
        c = colors[r["name"]]
        ax.plot(time_test, r["y_prob"], lw=0.8, color=c, alpha=0.7, label="P(anomaly)")
        ax.axhline(r["threshold"], ls="--", color="red", lw=1,
                   label=f"Thr={r['threshold']:.2f}")
        for i, (s, e) in enumerate(zip(starts_a, ends_a)):
            ax.axvspan(time_test[s], time_test[min(e, len(time_test)-1)],
                       alpha=0.18, color="black",
                       label="True Anomaly" if i == 0 else None)
        ax.set_ylabel("P(anomaly)", fontsize=10)
        ax.set_title(f"{r['name']} — AUC={r['auc']:.3f}, F1={r['f1']:.3f}",
                     fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time", fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].tick_params(axis="x", rotation=30, labelsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "temporal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
