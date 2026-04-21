"""
Combined Rashomon Analysis — 16 Models (sklearn + H2O + AutoGluon)
==================================================================
Loads pre-computed SHAP CSVs from diverse/, h2o/, and autogluon results.
16 models total:
  sklearn  : LightGBM, XGBoost, RandomForest, Ridge, DecisionTree
  H2O      : GBM, DRF, XRT, DeepLearning, GLM
  AutoGluon: LightGBMXT_BAG_L1, RandomForestMSE_BAG_L1, LightGBM_BAG_L1,
             WeightedEnsemble_L2, WeightedEnsemble_L3, LightGBMXT_BAG_L2

Outputs:
  1) 4-panel Rashomon summary plot
  2) Per-feature Rashomon subplot
  3) Per-feature anomaly statistics (t-test, Cohen's d, point-biserial)
  4) Anomaly classification comparison (5 configs)
  5) Effect size / boxplot / ROC-PR / importance plots

Usage:
    python run_combined.py                                      # full
    python run_combined.py --start 2021-06-01 --end 2021-06-25  # zoom
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MULTI_DIR     = BASE_DIR.parent            # automl/multivariate
PROJECT_DIR   = MULTI_DIR.parent.parent    # DC-VAE
DATA_DIR      = PROJECT_DIR / "TELCO_data"
LABELS_DIR    = PROJECT_DIR / "TELCO_labels"
DIVERSE_DIR   = MULTI_DIR / "results" / "diverse"
H2O_DIR       = MULTI_DIR / "results" / "h2o"
AUTOGLUON_DIR = MULTI_DIR / "results"
RESULTS_DIR   = BASE_DIR / "results"

TARGET = "TS1"

# ── model metadata ──────────────────────────────────────────────────
MODEL_COLORS = {
    # sklearn
    "LightGBM":         "#1f77b4",
    "XGBoost":          "#d62728",
    "RandomForest":     "#2ca02c",
    "Ridge":            "#ff7f0e",
    "DecisionTree":     "#9467bd",
    # H2O
    "GBM_H2O":          "#17becf",
    "DRF_H2O":          "#bcbd22",
    "XRT_H2O":          "#e377c2",
    "DeepLearning_H2O": "#7f7f7f",
    "GLM_H2O":          "#8c564b",
    # AutoGluon
    "LightGBMXT_BAG_L1_AG":      "#aec7e8",
    "RandomForestMSE_BAG_L1_AG": "#98df8a",
    "LightGBM_BAG_L1_AG":        "#ff9896",
    "WeightedEnsemble_L2_AG":    "#c5b0d5",
    "WeightedEnsemble_L3_AG":    "#c49c94",
    "LightGBMXT_BAG_L2_AG":      "#dbdb8d",
}

MODEL_SHORT = {
    "LightGBM": "LightGBM", "XGBoost": "XGBoost",
    "RandomForest": "RF", "Ridge": "Ridge",
    "DecisionTree": "DT",
    "GBM_H2O": "GBM(H2O)", "DRF_H2O": "DRF(H2O)",
    "XRT_H2O": "XRT(H2O)", "DeepLearning_H2O": "DL(H2O)",
    "GLM_H2O": "GLM(H2O)",
    "LightGBMXT_BAG_L1_AG": "LGBMXT L1(AG)",
    "RandomForestMSE_BAG_L1_AG": "RF L1(AG)",
    "LightGBM_BAG_L1_AG": "LGB L1(AG)",
    "WeightedEnsemble_L2_AG": "WE L2(AG)",
    "WeightedEnsemble_L3_AG": "WE L3(AG)",
    "LightGBMXT_BAG_L2_AG": "LGBMXT L2(AG)",
}

FEAT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
]

# ── helpers ──────────────────────────────────────────────────────────
def _read_csv(path):
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = df.index.tz_localize(None)
    return df


def load_test():
    return _read_csv(DATA_DIR / "TELCO_data_test.csv")


def load_labels():
    parts = []
    for s in ("train", "val", "test"):
        p = LABELS_DIR / f"TELCO_labels_{s}.csv"
        if p.exists():
            parts.append(_read_csv(p))
    lb = pd.concat(parts)
    return lb[~lb.index.duplicated(keep="first")].sort_index()


def load_all_shap():
    data = {}
    # sklearn diverse
    for p in sorted(DIVERSE_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace("shap_", "").replace(f"_{TARGET}", "")
        data[name] = _read_csv(p)
    # H2O
    for p in sorted(H2O_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace("shap_", "").replace(f"_{TARGET}", "")
        data[name + "_H2O"] = _read_csv(p)
    # AutoGluon
    for p in sorted(AUTOGLUON_DIR.glob(f"shap_*_{TARGET}.csv")):
        name = p.stem.replace("shap_", "").replace(f"_{TARGET}", "")
        if "anomaly" in name:
            continue
        data[name + "_AG"] = _read_csv(p)
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


def smooth(v, w=15):
    if len(v) <= w:
        return v
    return pd.Series(v).rolling(w, min_periods=1, center=True).mean().values


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
    prec = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    rec  = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    return {"name": name, "auc": auc, "ap": ap, "f1": max(f1s),
            "precision": prec, "recall": rec, "threshold": best_thr,
            "cm": cm, "y_prob": y_prob, "y_pred": y_pred, "importance": imp}


# =====================================================================
# MAIN
# =====================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--start", default=None)
    pa.add_argument("--end", default=None)
    args = pa.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Combined Rashomon + Anomaly Analysis — sklearn + H2O + AutoGluon")
    print("=" * 65)

    test = load_test()
    labels = load_labels()
    features = [c for c in test.columns if c != TARGET]

    shap_data = load_all_shap()
    model_names = list(shap_data.keys())
    n_models = len(model_names)
    print(f"Models ({n_models}): {model_names}")

    # ── date filter ──────────────────────────────────────────────────
    is_zoom = args.start is not None and args.end is not None
    if is_zoom:
        ds = pd.Timestamp(args.start)
        de = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        mask = (test.index >= ds) & (test.index <= de)
        time_axis = test.index[mask]
        test_vals = test[TARGET][mask].values
        shap_plot = {m: df.loc[mask] for m, df in shap_data.items()}
        tag = f"zoom_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
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

    # Feature ranking (mean |SHAP| across all models)
    imps = [df.abs().mean() for df in shap_plot.values()]
    combined_imp = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined_imp.sort_values(ascending=False).index.tolist()
    top_feats = [f for f in top_feats if f in features]
    print(f"Feature ranking: {top_feats}")

    title_range = f"{args.start} to {args.end}" if is_zoom else "Full Test"

    # =================================================================
    # PLOT 1: 4-panel Rashomon summary
    # =================================================================
    fig, axes = plt.subplots(4, 1, figsize=(24, 22),
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1, 1]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Panel 1 — Actual + anomalies
    ax = axes[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.set_title(f"{TARGET} — Combined Rashomon: {n_models} Models "
                 f"(sklearn + H2O + AG) — {title_range}",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Panel 2 — Per-feature SHAP mean ± Rashomon σ
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

    # Panel 3 — Per-model total |SHAP|
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

    # Panel 4 — Total Rashomon uncertainty
    ax = axes[3]
    all_vals = []
    for m in model_names:
        fc = [c for c in shap_plot[m].columns if c in features]
        all_vals.append(shap_plot[m][fc].values[:n])
    stacked = np.stack(all_vals)
    total_std = smooth(np.std(stacked, axis=0).mean(axis=1), sw)
    ax.fill_between(time_axis[:len(total_std)], 0, total_std,
                    alpha=0.4, color="#ff7f0e",
                    label=f"Rashomon σ ({n_models} models)")
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

    out1 = RESULTS_DIR / f"rashomon_summary_{TARGET}_{tag}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # =================================================================
    # PLOT 2: Per-feature Rashomon subplots
    # =================================================================
    n_feats = len(top_feats)
    heights = [1.5] + [1] * n_feats
    fig2, axes2 = plt.subplots(1 + n_feats, 1,
                               figsize=(24, 3 + 2.5 * n_feats),
                               gridspec_kw={"height_ratios": heights},
                               sharex=True)
    fig2.subplots_adjust(hspace=0.15)

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
            ax.legend(fontsize=5, loc="upper right", ncol=6, framealpha=0.9)

    axes2[-1].set_xlabel("Time", fontsize=11)
    if is_zoom:
        axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes2[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes2[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes2[-1].tick_params(axis="x", rotation=30, labelsize=9)

    out2 = RESULTS_DIR / f"rashomon_perfeature_{TARGET}_{tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # =================================================================
    # ANOMALY ANALYSIS — statistical tests
    # =================================================================
    print("\n" + "=" * 65)
    print(f"SHAP–Anomaly Statistical Analysis ({n_models} models)")
    print("=" * 65)

    n_full = len(test)
    anom_vec = np.zeros(n_full, dtype=bool)
    if TARGET in labels.columns:
        ts_l = labels[TARGET]
        tset = {t: i for i, t in enumerate(test.index)}
        for t in ts_l[ts_l > 0].index:
            if t in tset:
                anom_vec[tset[t]] = True
    n_anom = anom_vec.sum()
    print(f"Test: {n_full} steps, anomaly={n_anom}, normal={n_full - n_anom}")

    # Stack full-test SHAP  (n_models, n_timesteps, n_features)
    stacked_full = np.stack([shap_data[m][features].values[:n_full] for m in model_names])
    mean_shap = np.mean(stacked_full, axis=0)
    abs_mean_shap = np.abs(mean_shap)
    rashomon_std = np.std(stacked_full, axis=0)

    # Per-feature stats
    print(f"\n{'Feature':>5s}  {'Anom':>8s}  {'Norm':>8s}  {'Ratio':>6s}  "
          f"{'d':>6s}  {'p':>10s}  {'r_pb':>6s}")
    results = []
    for fi, feat in enumerate(features):
        sa = abs_mean_shap[anom_vec, fi]
        sn = abs_mean_shap[~anom_vec, fi]
        t_val, p = stats.ttest_ind(sa, sn, equal_var=False)
        pooled = np.sqrt((sa.var() + sn.var()) / 2)
        d = (sa.mean() - sn.mean()) / pooled if pooled > 0 else 0
        r_pb, p_pb = stats.pointbiserialr(anom_vec, abs_mean_shap[:, fi])
        ratio = sa.mean() / sn.mean() if sn.mean() > 0 else np.inf

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

    # =================================================================
    # ANOMALY CLASSIFICATION
    # =================================================================
    print("\n" + "=" * 65)
    print("Anomaly Classification")
    print("=" * 65)

    # Feature sets
    # A) Per-model |SHAP|
    shap_feats = pd.DataFrame(index=test.index)
    for m in model_names:
        short = MODEL_SHORT.get(m, m)
        for feat in features:
            shap_feats[f"|shap|_{short}_{feat}"] = shap_data[m][feat].abs().values[:n_full]

    # B) Mean |SHAP|
    mean_feats = pd.DataFrame(abs_mean_shap, index=test.index,
                               columns=[f"mean_|shap|_{f}" for f in features])

    # C) Rashomon σ + total
    rash_feats = pd.DataFrame(rashomon_std, index=test.index,
                               columns=[f"rashomon_σ_{f}" for f in features])
    rash_feats["total_rashomon_σ"] = rashomon_std.mean(axis=1)

    n_shap = shap_feats.shape[1]
    n_mean = mean_feats.shape[1]
    n_rash = rash_feats.shape[1]
    configs = {
        f"All SHAP ({n_shap})":               shap_feats,
        f"Mean |SHAP| ({n_mean})":             mean_feats,
        f"Rashomon σ ({n_rash})":              rash_feats,
        f"Mean + Rashomon ({n_mean+n_rash})":  pd.concat([mean_feats, rash_feats], axis=1),
        f"All Combined ({n_shap+n_rash})":     pd.concat([shap_feats, rash_feats], axis=1),
    }

    y = labels[TARGET].reindex(test.index).fillna(0).astype(int)
    y = (y > 0).astype(int)
    split = int(n_full * 0.6)
    y_train, y_test_y = y.iloc[:split], y.iloc[split:]
    print(f"Train: {split} (anom={y_train.sum()}), Test: {n_full-split} (anom={y_test_y.sum()})")

    clf_results = []
    for name, X in configs.items():
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        r = train_clf(Xtr, Xte, y_train, y_test_y, name)
        clf_results.append(r)
        print(f"\n  {name}:")
        print(f"    AUC={r['auc']:.4f}  AP={r['ap']:.4f}  F1={r['f1']:.4f}  "
              f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}")

    comp = pd.DataFrame([{
        "Model": r["name"], "ROC-AUC": r["auc"], "PR-AUC": r["ap"],
        "F1": r["f1"], "Precision": r["precision"], "Recall": r["recall"],
    } for r in clf_results])
    comp.to_csv(RESULTS_DIR / "classification_comparison.csv", index=False)
    print("\n" + comp.to_string(index=False))

    # =================================================================
    # PLOTS — anomaly analysis
    # =================================================================
    print("\nGenerating anomaly plots...")

    # P1: Effect size bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    rs1 = res_df.sort_values("cohens_d")
    c1 = ["#d62728" if p < 0.05 else "#aaa" for p in rs1.p_value]
    ax1.barh(rs1.feature, rs1.cohens_d, color=c1, alpha=0.8)
    ax1.axvline(0, color="black", lw=0.5)
    ax1.axvline(0.2, color="gray", lw=0.5, ls="--")
    ax1.axvline(-0.2, color="gray", lw=0.5, ls="--")
    ax1.set_xlabel("Cohen's d"); ax1.grid(True, alpha=0.15, axis="x")
    ax1.set_title(f"|SHAP| Effect Size ({n_models} models)\n(+) = higher during anomaly",
                  fontweight="bold")

    rs2 = res_df.sort_values("rash_d")
    c2 = ["#ff7f0e" if p < 0.05 else "#aaa" for p in rs2.rash_p]
    ax2.barh(rs2.feature, rs2.rash_d, color=c2, alpha=0.8)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.axvline(0.2, color="gray", lw=0.5, ls="--")
    ax2.axvline(-0.2, color="gray", lw=0.5, ls="--")
    ax2.set_xlabel("Cohen's d"); ax2.grid(True, alpha=0.15, axis="x")
    ax2.set_title(f"Rashomon σ Effect Size ({n_models} models)\n(+) = more disagreement",
                  fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # P2: Box plots — top 5 features
    top5 = res_df.sort_values("p_value").head(5).feature.tolist()
    fig, axes_b = plt.subplots(1, len(top5), figsize=(4 * len(top5), 5))
    for i, feat in enumerate(top5):
        fi = features.index(feat)
        ax = axes_b[i]
        data_a = abs_mean_shap[anom_vec, fi]
        data_n = abs_mean_shap[~anom_vec, fi]
        bp = ax.boxplot([data_a, data_n], labels=["Anomaly", "Normal"],
                        patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("#d62728"); bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#1f77b4"); bp["boxes"][1].set_alpha(0.6)
        d_val = res_df[res_df.feature == feat].cohens_d.values[0]
        p_val = res_df[res_df.feature == feat].p_value.values[0]
        ax.set_title(f"{feat}\nd={d_val:.2f}, p={p_val:.1e}", fontsize=10)
        ax.set_ylabel("|SHAP|" if i == 0 else "")
        ax.grid(True, alpha=0.15)
    plt.suptitle(f"Top 5 Features: |SHAP| Anomaly vs Normal ({n_models} models)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "boxplots_top5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # P3: ROC + PR comparison
    colors_clf = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for i, r in enumerate(clf_results):
        c = colors_clf[i % len(colors_clf)]
        fpr, tpr, _ = roc_curve(y_test_y, r["y_prob"])
        ax1.plot(fpr, tpr, lw=2, color=c, label=f"{r['name']} ({r['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test_y, r["y_prob"])
        ax2.plot(rec, prec, lw=2, color=c, label=f"{r['name']} ({r['ap']:.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color="gray")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curves", fontweight="bold")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.2)
    ax2.axhline(y_test_y.mean(), ls="--", color="gray")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves", fontweight="bold")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_pr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # P4: Metrics bar
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # P5: Best model feature importance
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

    # ── Rashomon σ summary stats ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("Per-Feature Rashomon σ (mean across time)")
    print("=" * 65)
    for feat in top_feats:
        vals = [shap_data[m][feat].values[:n_full] for m in model_names
                if feat in shap_data[m].columns]
        if len(vals) >= 2:
            sigma = np.std(np.array(vals), axis=0).mean()
            print(f"  {feat:>5s}: σ = {sigma:.4f}")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
