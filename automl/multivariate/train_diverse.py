"""
Diverse Model Rashomon Analysis
================================
Trains structurally different models for real Rashomon uncertainty:
  1. LightGBM (gradient boosting)
  2. XGBoost (gradient boosting, different implementation)
  3. Random Forest (bagging)
  4. Ridge Regression (linear)
  5. Decision Tree (single tree, shallow)

Each gives very different SHAP patterns → visible Rashomon bands.

Usage:
    python train_diverse.py
    python train_diverse.py --skip_train
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings, time, pickle

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
MODEL_DIR   = BASE_DIR / "models" / "diverse"
RESULTS_DIR = BASE_DIR / "results" / "diverse"

TARGET = "TS1"


def load_splits():
    dfs = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(DATA_DIR / f"TELCO_data_{s}.csv",
                        parse_dates=["time"], index_col="time")
        d.index = d.index.tz_localize(None)
        dfs[s] = d
    return dfs["train"], dfs["val"], dfs["test"]


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


# ── main ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--skip_train", action="store_true")
    pa.add_argument("--start", default=None)
    pa.add_argument("--end", default=None)
    args = pa.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Diverse Model Rashomon Analysis")
    print("=" * 60)

    # Load data
    train, val, test = load_splits()
    labels = load_labels()
    features = [c for c in train.columns if c != TARGET]
    print(f"Features: {features}")

    X_train = pd.concat([train, val])[features]
    y_train = pd.concat([train, val])[TARGET]
    X_test = test[features]
    y_test = test[TARGET]

    # Scale for Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Define models ────────────────────────────────────────────
    model_configs = {
        "LightGBM": {
            "class": "lgb",
            "params": {"n_estimators": 500, "learning_rate": 0.05,
                       "num_leaves": 63, "verbose": -1},
        },
        "XGBoost": {
            "class": "xgb",
            "params": {"n_estimators": 500, "learning_rate": 0.05,
                       "max_depth": 6, "verbosity": 0},
        },
        "RandomForest": {
            "class": "rf",
            "params": {"n_estimators": 100, "max_depth": 12,
                       "min_samples_leaf": 10, "n_jobs": -1},
        },
        "Ridge": {
            "class": "ridge",
            "params": {"alpha": 1.0},
        },
        "DecisionTree": {
            "class": "dt",
            "params": {"max_depth": 8, "min_samples_leaf": 20},
        },
    }

    # ── Train or load ────────────────────────────────────────────
    models = {}
    preds = {}
    r2_scores = {}

    for name, cfg in model_configs.items():
        model_file = MODEL_DIR / f"{name}.pkl"

        if args.skip_train and model_file.exists():
            print(f"  Loading {name}...")
            with open(model_file, "rb") as f:
                models[name] = pickle.load(f)
        else:
            print(f"  Training {name}...")
            t0 = time.time()
            if cfg["class"] == "lgb":
                m = lgb.LGBMRegressor(**cfg["params"])
                m.fit(X_train, y_train)
            elif cfg["class"] == "xgb":
                m = xgb.XGBRegressor(**cfg["params"])
                m.fit(X_train, y_train)
            elif cfg["class"] == "rf":
                m = RandomForestRegressor(**cfg["params"], random_state=42)
                m.fit(X_train, y_train)
            elif cfg["class"] == "ridge":
                m = Ridge(**cfg["params"])
                m.fit(X_train_s, y_train)
            elif cfg["class"] == "dt":
                m = DecisionTreeRegressor(**cfg["params"], random_state=42)
                m.fit(X_train, y_train)
            models[name] = m
            with open(model_file, "wb") as f:
                pickle.dump(m, f)
            print(f"    Done in {time.time() - t0:.1f}s")

        # Predict
        if name == "Ridge":
            preds[name] = models[name].predict(X_test_s)
        else:
            preds[name] = models[name].predict(X_test)

        ss_res = np.sum((y_test.values - preds[name]) ** 2)
        ss_tot = np.sum((y_test.values - y_test.values.mean()) ** 2)
        r2_scores[name] = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y_test.values - preds[name]) ** 2))
        print(f"    {name}: RMSE={rmse:.4f}, R²={r2_scores[name]:.4f}")

    # ── SHAP ─────────────────────────────────────────────────────
    print("\nComputing SHAP values...")
    shap_dict = {}
    for name, m in models.items():
        print(f"  SHAP for {name}...")
        if name == "Ridge":
            explainer = shap.LinearExplainer(m, X_train_s)
            sv = explainer.shap_values(X_test_s)
        elif name in ("LightGBM", "XGBoost", "RandomForest", "DecisionTree"):
            explainer = shap.TreeExplainer(m)
            if name == "RandomForest":
                sv = explainer.shap_values(X_test)
            else:
                sv = explainer.shap_values(X_test)
        shap_df = pd.DataFrame(sv, columns=features, index=X_test.index)
        shap_dict[name] = shap_df
        shap_df.to_csv(RESULTS_DIR / f"shap_{name}_{TARGET}.csv")
        print(f"    Done.")

    model_names = list(shap_dict.keys())

    # ── Date range ───────────────────────────────────────────────
    is_zoom = args.start is not None and args.end is not None
    if is_zoom:
        ds = pd.Timestamp(args.start)
        de = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        mask = (test.index >= ds) & (test.index <= de)
        time_axis = test.index[mask]
        test_vals = test[TARGET][mask].values
        shap_plot = {m: df.loc[mask] for m, df in shap_dict.items()}
        mask_arr = mask.values if hasattr(mask, 'values') else mask
        preds_plot = {m: preds[m][mask_arr] for m in model_names}
        tag = f"zoom_{args.start.replace('-','')}_{args.end.replace('-','')}"
    else:
        time_axis = test.index
        test_vals = y_test.values
        shap_plot = shap_dict
        preds_plot = preds
        tag = "full"

    n = len(time_axis)
    atimes, astarts, aends = get_anomaly_regions(labels, TARGET, time_axis)
    print(f"\nTimesteps: {n}, Anomalies: {len(atimes)}")

    # Feature ranking
    imps = []
    for df in shap_plot.values():
        imps.append(df.abs().mean())
    combined = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined.sort_values(ascending=False).index.tolist()
    print(f"Feature ranking: {top_feats}")

    # ── Plot: 4 panels ───────────────────────────────────────────
    COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff"]
    MODEL_COLORS = {
        "LightGBM": "#1f77b4", "XGBoost": "#d62728", "RandomForest": "#2ca02c",
        "Ridge": "#ff7f0e", "DecisionTree": "#9467bd",
    }
    sw = 10 if is_zoom else 15

    fig, axes = plt.subplots(4, 1, figsize=(24, 20),
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1, 1]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Panel 1: Actual + predictions
    ax = axes[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    for m in model_names:
        ax.plot(time_axis, preds_plot[m][:n], lw=0.7, alpha=0.7, ls="--",
                color=MODEL_COLORS[m],
                label=f"{m} (R²={r2_scores[m]:.3f})")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    title_range = f"{args.start} to {args.end}" if is_zoom else "Full Test"
    ax.set_title(f"{TARGET} — Diverse Models ({title_range})",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.2)

    # Panel 2: Per-feature SHAP + Rashomon band
    ax = axes[1]
    for fi, feat in enumerate(top_feats):
        vals_per_model = []
        for m in model_names:
            if feat in shap_plot[m].columns:
                vals_per_model.append(shap_plot[m][feat].values[:n])
        if len(vals_per_model) >= 2:
            stacked = np.array(vals_per_model)
            mn = smooth(np.mean(stacked, axis=0), sw)
            sd = smooth(np.std(stacked, axis=0), sw)
            c = COLORS[fi % len(COLORS)]
            ax.plot(time_axis[:len(mn)], mn, lw=1.2, color=c, alpha=0.85, label=feat)
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd, alpha=0.15, color=c)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("SHAP value", fontsize=11)
    ax.set_title("Per-Feature SHAP (mean ± Rashomon σ across 5 model types)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.15)

    # Panel 3: Per-model total |SHAP|
    ax = axes[2]
    for m in model_names:
        total = smooth(shap_plot[m].abs().sum(axis=1).values[:n], sw)
        ax.plot(time_axis[:len(total)], total, lw=1.0,
                color=MODEL_COLORS[m], alpha=0.8, label=m)
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Total |SHAP|", fontsize=11)
    ax.set_title("Per-Model Total Feature Attribution", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)

    # Panel 4: Total Rashomon uncertainty
    ax = axes[3]
    all_shap = np.stack([shap_plot[m].values[:n] for m in model_names])
    total_std = smooth(np.std(all_shap, axis=0).mean(axis=1), sw)
    ax.fill_between(time_axis[:len(total_std)], 0, total_std,
                    alpha=0.4, color="#ff7f0e", label="Rashomon σ")
    ax.plot(time_axis[:len(total_std)], total_std, lw=1.0, color="#ff7f0e")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Rashomon σ", fontsize=11)
    ax.set_title("Total Rashomon Uncertainty (5 diverse models)", fontsize=11)
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

    out = RESULTS_DIR / f"rashomon_diverse_{TARGET}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ── Per-feature subplots ─────────────────────────────────────
    n_feats = len(top_feats)
    heights = [1.5] + [1] * n_feats
    fig2, axes2 = plt.subplots(
        1 + n_feats, 1,
        figsize=(24, 3 + 2.5 * n_feats),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    fig2.subplots_adjust(hspace=0.15)

    # Top: time series
    ax = axes2[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    for m in model_names:
        ax.plot(time_axis, preds_plot[m][:n], lw=0.6, alpha=0.6, ls="--",
                color=MODEL_COLORS[m], label=f"{m}")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.set_title(f"{TARGET} — Per-Feature Rashomon (5 diverse models, {title_range})",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.2)

    # Per-feature panels
    for i, feat in enumerate(top_feats):
        ax = axes2[i + 1]
        c = COLORS[i % len(COLORS)]
        vals_per_model = []
        for m in model_names:
            if feat in shap_plot[m].columns:
                v = shap_plot[m][feat].values[:n]
                vals_per_model.append(v)
                ax.plot(time_axis[:len(v)], smooth(v, sw), lw=0.5, alpha=0.4,
                        ls="--", color=MODEL_COLORS[m])
        if len(vals_per_model) >= 2:
            stacked = np.array(vals_per_model)
            mn = smooth(np.mean(stacked, axis=0), sw)
            sd = smooth(np.std(stacked, axis=0), sw)
            ax.plot(time_axis[:len(mn)], mn, lw=1.5, color=c, alpha=0.9,
                    label=f"{feat} (mean)")
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd,
                            alpha=0.25, color=c, label="±1σ Rashomon")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        shade_anom(ax, time_axis, astarts, aends, label=(i == 0))
        ax.set_ylabel(feat, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

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

    print("Done!")


if __name__ == "__main__":
    main()
