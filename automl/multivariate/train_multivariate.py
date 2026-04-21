"""
Multivariate Tabular Forecasting
=================================
Target: TS1
Features: TS2, TS3, ..., TS12 (no lags, no engineered features)

Uses AutoGluon Tabular regressor to predict TS1 from other series.
Then computes SHAP values via surrogate LightGBM (TreeSHAP — fast).

Usage:
    python train_multivariate.py
    python train_multivariate.py --target TS1 --time_limit 600
    python train_multivariate.py --skip_train          # reuse existing model
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings, time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
import lightgbm as lgb
from autogluon.tabular import TabularPredictor

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
MODEL_DIR   = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ── data ─────────────────────────────────────────────────────────────
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
    if parts:
        lb = pd.concat(parts)
        return lb[~lb.index.duplicated(keep="first")].sort_index()
    return pd.DataFrame()


def make_tabular(df, target):
    features = [c for c in df.columns if c != target]
    return df[features].copy(), df[target].copy()


# ── anomaly helpers ──────────────────────────────────────────────────
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


# ── main ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--target", default="TS1")
    pa.add_argument("--time_limit", type=int, default=300)
    pa.add_argument("--top_n", type=int, default=11)
    pa.add_argument("--skip_train", action="store_true",
                    help="Skip training, load existing model")
    args = pa.parse_args()
    target = args.target

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Multivariate Tabular — target: {target}")
    print("=" * 60)

    # 1. Load data
    train, val, test = load_splits()
    labels = load_labels()
    features = [c for c in train.columns if c != target]
    print(f"Features ({len(features)}): {features}")

    X_train, y_train = make_tabular(train, target)
    X_val,   y_val   = make_tabular(val,   target)
    X_test,  y_test  = make_tabular(test,  target)

    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    df_fit = X_fit.copy()
    df_fit[target] = y_fit

    df_test = X_test.copy()
    df_test[target] = y_test

    print(f"Train+Val: {len(df_fit)}, Test: {len(X_test)}")

    # 2. Train or load AutoGluon Tabular
    model_path = MODEL_DIR / target
    if args.skip_train and model_path.exists():
        print("\nLoading existing model...")
        predictor = TabularPredictor.load(str(model_path))
    else:
        print("\nTraining AutoGluon Tabular...")
        t0 = time.time()
        predictor = TabularPredictor(
            label=target,
            path=str(model_path),
            eval_metric="root_mean_squared_error",
            problem_type="regression",
        )
        predictor.fit(
            df_fit,
            time_limit=args.time_limit,
            presets="best_quality",
            verbosity=1,
        )
        print(f"Training done in {time.time() - t0:.0f}s")

    # 3. Evaluate
    perf = predictor.evaluate(df_test)
    print(f"\nTest performance: {perf}")

    lb = predictor.leaderboard(df_test, silent=True)
    print("\nLeaderboard:")
    print(lb[["model", "score_test", "score_val"]].to_string(index=False))
    lb.to_csv(RESULTS_DIR / f"leaderboard_{target}.csv", index=False)

    # Pick structurally diverse models (not multiple ensembles)
    diverse_order = [
        "LightGBMXT_BAG_L1", "RandomForestMSE_BAG_L1", "LightGBM_BAG_L1",
        "WeightedEnsemble_L2", "LightGBMXT_BAG_L2",
    ]
    top_models = [m for m in diverse_order if m in lb["model"].values][:3]
    if len(top_models) < 3:
        top_models = lb["model"].head(3).tolist()
    print(f"\nSelected diverse models: {top_models}")

    # 4. Predictions per model
    preds = {}
    r2_scores = {}
    for m in top_models:
        preds[m] = predictor.predict(df_test, model=m).values
        rmse = np.sqrt(np.mean((preds[m] - y_test.values) ** 2))
        r2 = 1 - np.sum((y_test.values - preds[m])**2) / np.sum((y_test.values - y_test.values.mean())**2)
        r2_scores[m] = r2
        print(f"  {m}: RMSE={rmse:.4f}, R²={r2:.4f}")

    # 5. SHAP via surrogate LightGBM (TreeSHAP — fast!)
    print("\nComputing SHAP values via surrogate TreeSHAP...")
    shap_dict = {}
    faith = {}
    for m in top_models:
        print(f"  Surrogate for {m}...")
        y_pred_train = predictor.predict(df_fit, model=m).values
        y_pred_test  = preds[m]

        # Train surrogate LightGBM
        ds_train = lgb.Dataset(X_fit, label=y_pred_train)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
        }
        surrogate = lgb.train(
            params, ds_train, num_boost_round=500,
            valid_sets=[lgb.Dataset(X_test, label=y_pred_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Faithfulness
        surr_pred = surrogate.predict(X_test)
        ss_res = np.sum((y_pred_test - surr_pred) ** 2)
        ss_tot = np.sum((y_pred_test - y_pred_test.mean()) ** 2)
        r2_f = 1 - ss_res / ss_tot
        faith[m] = r2_f
        print(f"    Surrogate R² = {r2_f:.4f}")

        # TreeSHAP
        explainer = shap.TreeExplainer(surrogate)
        sv = explainer.shap_values(X_test)
        shap_df = pd.DataFrame(sv, columns=features, index=X_test.index)
        shap_dict[m] = shap_df
        shap_df.to_csv(RESULTS_DIR / f"shap_{m}_{target}.csv")
        print(f"    Saved shap_{m}_{target}.csv")

    # Save faithfulness
    faith_df = pd.DataFrame([
        {"model": m, "surrogate_r2": faith[m], "model_r2": r2_scores[m]}
        for m in top_models
    ])
    faith_df.to_csv(RESULTS_DIR / f"faithfulness_{target}.csv", index=False)

    # 6. Plot: 4 panels
    print("\nPlotting...")
    time_axis = test.index
    test_vals = y_test.values
    n = len(time_axis)
    atimes, astarts, aends = get_anomaly_regions(labels, target, time_axis)
    print(f"Anomalies: {len(atimes)}")

    # Feature importance ranking
    imps = []
    for df in shap_dict.values():
        imps.append(df.abs().mean())
    combined = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined.sort_values(ascending=False).head(args.top_n).index.tolist()
    print(f"Top features: {top_feats}")

    COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff"]
    MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

    sw = 15
    def smooth(v, w=sw):
        if len(v) <= w:
            return v
        return pd.Series(v).rolling(w, min_periods=1, center=True).mean().values

    fig, axes = plt.subplots(4, 1, figsize=(24, 18),
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1, 1]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Panel 1: Actual + Predictions
    ax = axes[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    for i, m in enumerate(top_models):
        ax.plot(time_axis, preds[m], lw=0.7, alpha=0.7, ls="--",
                color=MODEL_COLORS[i],
                label=f"{m} (R²={r2_scores[m]:.3f})")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(target, fontsize=11)
    ax.set_title(f"{target} — Multivariate Tabular (features: TS2–TS12, no lags)",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel 2: Per-feature SHAP + Rashomon band
    ax = axes[1]
    for fi, feat in enumerate(top_feats):
        vals_per_model = []
        for m in top_models:
            if feat in shap_dict[m].columns:
                vals_per_model.append(shap_dict[m][feat].values)
        if len(vals_per_model) >= 2:
            stacked = np.array(vals_per_model)
            mn = smooth(np.mean(stacked, axis=0))
            sd = smooth(np.std(stacked, axis=0))
            c = COLORS[fi % len(COLORS)]
            ax.plot(time_axis[:len(mn)], mn, lw=1.2, color=c, alpha=0.85, label=feat)
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd, alpha=0.10, color=c)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("SHAP value", fontsize=11)
    ax.set_title("Per-Feature SHAP (mean ± Rashomon σ)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.15)

    # Panel 3: Per-model total |SHAP|
    ax = axes[2]
    for i, m in enumerate(top_models):
        total = smooth(shap_dict[m].abs().sum(axis=1).values)
        ax.plot(time_axis[:len(total)], total, lw=1.0, color=MODEL_COLORS[i],
                alpha=0.8, label=f"{m} (surr R²={faith[m]:.3f})")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("Total |SHAP|", fontsize=11)
    ax.set_title("Per-Model Total Feature Attribution", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)

    # Panel 4: Total Rashomon uncertainty
    ax = axes[3]
    all_shap = np.stack([shap_dict[m].values for m in top_models])
    total_std = smooth(np.std(all_shap, axis=0).mean(axis=1))
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
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[-1].tick_params(axis="x", rotation=30, labelsize=9)

    out = RESULTS_DIR / f"multivariate_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    print("Done!")


if __name__ == "__main__":
    main()
