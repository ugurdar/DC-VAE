"""
SHAP Uncertainty-based Anomaly Classifier
==========================================
Uses ONLY Rashomon uncertainty (inter-model SHAP disagreement)
as features to classify anomalies.

Features:
  - Rashomon σ per feature (std of SHAP across models) → 11
  - Total Rashomon σ (mean of per-feature σ) → 1
  → 12 features total

Usage:
    python shap_uncertainty_classifier.py
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
RESULTS_DIR = BASE_DIR / "results" / "anomaly_clf_uncertainty"

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


def build_uncertainty_features(shap_data):
    """Only Rashomon uncertainty features."""
    models = list(shap_data.keys())
    idx = shap_data[models[0]].index
    ts_feats = shap_data[models[0]].columns.tolist()

    # Stack: (n_models, n_timesteps, n_features)
    stacked = np.stack([shap_data[m].values for m in models])

    # Per-feature Rashomon σ
    per_feat_std = pd.DataFrame(
        np.std(stacked, axis=0), index=idx,
        columns=[f"rashomon_σ_{c}" for c in ts_feats]
    )

    # Total Rashomon σ
    total_std = pd.Series(
        np.std(stacked, axis=0).mean(axis=1),
        index=idx, name="total_rashomon_σ"
    )

    X = pd.concat([per_feat_std, total_std.to_frame()], axis=1)
    return X


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHAP Uncertainty Anomaly Classifier")
    print("(Only Rashomon σ features)")
    print("=" * 60)

    # Load
    shap_data = load_shap()
    models = list(shap_data.keys())
    print(f"Models: {models}")

    labels = load_labels()

    # Build features — ONLY uncertainty
    X = build_uncertainty_features(shap_data)
    print(f"Feature matrix: {X.shape}")
    print(f"Features: {list(X.columns)}")

    # Labels
    y = labels[TARGET].reindex(X.index).fillna(0).astype(int)
    y = (y > 0).astype(int)
    print(f"\nClass distribution: {y.value_counts().to_dict()}")

    # Temporal split (60/40)
    n = len(X)
    split_idx = int(n * 0.6)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train)} (anom={y_train.sum()})")
    print(f"Test:  {len(X_test)} (anom={y_test.sum()})")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / n_pos if n_pos > 0 else 1

    # Train
    print("\nTraining LightGBM...")
    ds_train = lgb.Dataset(X_train_s, label=y_train.values)
    ds_val = lgb.Dataset(X_test_s, label=y_test.values, reference=ds_train)

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

    # Predict
    y_prob = model.predict(X_test_s)

    # Best threshold by F1
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_thr = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)
    y_pred = (y_prob >= best_thr).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nBest threshold: {best_thr:.2f} (F1={best_f1:.4f})")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"\nClassification Report (threshold={best_thr:.2f}):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    print("Confusion Matrix:")
    print(cm)

    # Feature importance
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print("\nFeature Importance:")
    print(imp.to_string(index=False))

    # ===== Plots =====

    # 1. ROC + PR curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"ROC-AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], ls="--", color="gray")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve (Uncertainty Only)", fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2)

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(rec, prec, lw=2, color="#d62728", label=f"AP = {ap:.3f}")
    ax2.axhline(y_test.mean(), ls="--", color="gray", label=f"Baseline = {y_test.mean():.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (Uncertainty Only)", fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Feature importance bar
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#ff7f0e" if "total" not in f else "#9467bd" for f in imp.feature]
    ax.barh(range(len(imp)), imp.importance.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.feature.values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance — Rashomon σ Only", fontweight="bold")
    legend_elements = [
        Patch(facecolor="#ff7f0e", label="Per-feature Rashomon σ"),
        Patch(facecolor="#9467bd", label="Total Rashomon σ"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Temporal prediction plot
    fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
    time_test = X_test.index

    ax = axes[0]
    ax.plot(time_test, y_prob, lw=0.8, color="#1f77b4", alpha=0.7, label="P(anomaly)")
    ax.axhline(best_thr, ls="--", color="red", lw=1, label=f"Threshold={best_thr:.2f}")
    anom_mask = y_test.values.astype(bool)
    if anom_mask.any():
        ch = np.diff(anom_mask.astype(int))
        starts = np.where(ch == 1)[0] + 1
        ends = np.where(ch == -1)[0] + 1
        if anom_mask[0]: starts = np.insert(starts, 0, 0)
        if anom_mask[-1]: ends = np.append(ends, len(anom_mask))
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(time_test[s], time_test[min(e, len(time_test)-1)],
                       alpha=0.18, color="black",
                       label="True Anomaly" if i == 0 else None)
    ax.set_ylabel("P(anomaly)", fontsize=11)
    ax.set_title("Anomaly Probability — Rashomon Uncertainty Only", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    tp = (y_pred == 1) & (y_test.values == 1)
    fp = (y_pred == 1) & (y_test.values == 0)
    fn = (y_pred == 0) & (y_test.values == 1)
    ax.scatter(time_test[tp], y_prob[tp], c="green", s=3, alpha=0.6, label=f"TP ({tp.sum()})")
    ax.scatter(time_test[fp], y_prob[fp], c="red", s=3, alpha=0.6, label=f"FP ({fp.sum()})")
    ax.scatter(time_test[fn], y_prob[fn], c="orange", s=3, alpha=0.6, label=f"FN ({fn.sum()})")
    ax.axhline(best_thr, ls="--", color="gray", lw=0.5)
    ax.set_ylabel("P(anomaly)", fontsize=11)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_title("TP / FP / FN Distribution", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].tick_params(axis="x", rotation=30, labelsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "temporal_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save metrics
    prec_val = cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
    rec_val = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
    res = pd.DataFrame({
        "metric": ["ROC-AUC", "PR-AUC (AP)", "Best F1", "Best Threshold",
                    "Precision", "Recall", "TP", "FP", "FN", "TN"],
        "value": [auc, ap, best_f1, best_thr, prec_val, rec_val,
                  cm[1,1], cm[0,1], cm[1,0], cm[0,0]]
    })
    res.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
