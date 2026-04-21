"""
H2O AutoML Multivariate Forecasting + SHAP
============================================
Target: TS1
Features: TS2–TS12 (no lags)

H2O trains diverse models: GBM, XGBoost, GLM, DRF, DeepLearning, StackedEnsemble
→ Natural Rashomon effect from very different model families.

Usage:
    python train_h2o.py
    python train_h2o.py --max_models 10 --max_runtime 300
    python train_h2o.py --skip_train          # reuse saved models
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings, time, json

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
import h2o
from h2o.automl import H2OAutoML

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABELS_DIR  = PROJECT_DIR / "TELCO_labels"
MODEL_DIR   = BASE_DIR / "models" / "h2o"
RESULTS_DIR = BASE_DIR / "results" / "h2o"

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


def compute_surrogate_shap(model, X_train_pd, X_test_pd, features, model_name):
    """Surrogate LightGBM + TreeSHAP for any H2O model."""
    import lightgbm as lgb

    # Get H2O predictions
    h2o_train = h2o.H2OFrame(X_train_pd)
    h2o_test = h2o.H2OFrame(X_test_pd)
    y_pred_train = model.predict(h2o_train).as_data_frame().values.ravel()
    y_pred_test = model.predict(h2o_test).as_data_frame().values.ravel()

    # Train surrogate
    ds_train = lgb.Dataset(X_train_pd[features], label=y_pred_train)
    ds_val = lgb.Dataset(X_test_pd[features], label=y_pred_test)
    params = {
        "objective": "regression", "metric": "rmse", "verbosity": -1,
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
    }
    surrogate = lgb.train(
        params, ds_train, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Faithfulness
    surr_pred = surrogate.predict(X_test_pd[features])
    ss_res = np.sum((y_pred_test - surr_pred) ** 2)
    ss_tot = np.sum((y_pred_test - y_pred_test.mean()) ** 2)
    r2_surr = 1 - ss_res / ss_tot

    # TreeSHAP
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X_test_pd[features])

    return sv, r2_surr, y_pred_test


# ── main ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--max_models", type=int, default=12)
    pa.add_argument("--max_runtime", type=int, default=300,
                    help="Max runtime in seconds for AutoML")
    pa.add_argument("--skip_train", action="store_true")
    pa.add_argument("--start", default=None)
    pa.add_argument("--end", default=None)
    pa.add_argument("--top_n_models", type=int, default=5,
                    help="Number of diverse models for Rashomon")
    args = pa.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H2O AutoML — Multivariate Rashomon")
    print("=" * 60)

    # Load data
    train, val, test = load_splits()
    labels = load_labels()
    features = [c for c in train.columns if c != TARGET]
    print(f"Features: {features}")

    X_train_pd = pd.concat([train, val]).reset_index(drop=True)
    X_test_pd = test.reset_index(drop=True)

    # Init H2O
    h2o.init(nthreads=-1, max_mem_size="4G")
    h2o.no_progress()

    train_h2o = h2o.H2OFrame(X_train_pd)
    test_h2o = h2o.H2OFrame(X_test_pd)

    if not args.skip_train:
        print(f"\nTraining H2O AutoML (max_models={args.max_models}, "
              f"max_runtime={args.max_runtime}s)...")
        t0 = time.time()
        aml = H2OAutoML(
            max_models=args.max_models,
            max_runtime_secs=args.max_runtime,
            seed=42,
            sort_metric="RMSE",
            project_name="multivariate_ts1",
        )
        aml.train(x=features, y=TARGET, training_frame=train_h2o)
        elapsed = time.time() - t0
        print(f"Training done in {elapsed:.0f}s")

        # Leaderboard
        lb = aml.leaderboard.as_data_frame()
        print("\nLeaderboard:")
        print(lb[["model_id", "rmse", "mae", "mean_residual_deviance"]].to_string(index=False))
        lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)

        # Save models — pick diverse types
        all_model_ids = lb["model_id"].tolist()

        # Group by model family
        families = {}
        for mid in all_model_ids:
            if "StackedEnsemble" in mid:
                fam = "StackedEnsemble"
            elif "XGBoost" in mid:
                fam = "XGBoost"
            elif "GBM" in mid:
                fam = "GBM"
            elif "DRF" in mid:
                fam = "DRF"
            elif "GLM" in mid:
                fam = "GLM"
            elif "DeepLearning" in mid:
                fam = "DeepLearning"
            else:
                fam = mid.split("_")[0]
            if fam not in families:
                families[fam] = mid  # best of each family

        # Pick top N diverse
        diverse_models = list(families.values())[:args.top_n_models]
        print(f"\nDiverse models ({len(diverse_models)}):")
        for mid in diverse_models:
            print(f"  {mid}")

        # Save each model
        saved = {}
        for mid in diverse_models:
            m = h2o.get_model(mid)
            path = h2o.save_model(m, path=str(MODEL_DIR), force=True)
            saved[mid] = path
            print(f"  Saved: {path}")

        # Save model list
        with open(MODEL_DIR / "model_list.json", "w") as f:
            json.dump(saved, f, indent=2)

    else:
        print("\nLoading saved models...")
        with open(MODEL_DIR / "model_list.json") as f:
            saved = json.load(f)
        diverse_models = list(saved.keys())
        for mid, path in saved.items():
            h2o.load_model(path)
        print(f"Loaded {len(diverse_models)} models")
        lb = pd.read_csv(RESULTS_DIR / "leaderboard.csv")

    # Evaluate + SHAP
    print("\nComputing predictions + surrogate SHAP...")
    shap_dict = {}
    preds = {}
    r2_scores = {}
    faith = {}

    y_test = test[TARGET].values

    for mid in diverse_models:
        m = h2o.get_model(mid)

        # Short name
        if "StackedEnsemble" in mid:
            short = "StackedEnsemble"
        elif "XGBoost" in mid:
            short = "XGBoost"
        elif "GBM" in mid:
            short = "GBM"
        elif "DRF" in mid:
            short = "DRF"
        elif "GLM" in mid:
            short = "GLM"
        elif "DeepLearning" in mid:
            short = "DeepLearning"
        else:
            short = mid.split("_")[0]

        print(f"\n  {short} ({mid})...")
        sv, r2_surr, y_pred = compute_surrogate_shap(
            m, X_train_pd, X_test_pd, features, short
        )
        faith[short] = r2_surr
        preds[short] = y_pred

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2_scores[short] = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        print(f"    RMSE={rmse:.4f}, R²={r2_scores[short]:.4f}, Surrogate R²={r2_surr:.4f}")

        shap_df = pd.DataFrame(sv, columns=features, index=test.index)
        shap_dict[short] = shap_df
        shap_df.to_csv(RESULTS_DIR / f"shap_{short}_{TARGET}.csv")

    model_names = list(shap_dict.keys())

    # Save faithfulness
    faith_df = pd.DataFrame([
        {"model": m, "surrogate_r2": faith[m], "model_r2": r2_scores[m]}
        for m in model_names
    ])
    faith_df.to_csv(RESULTS_DIR / f"faithfulness_{TARGET}.csv", index=False)
    print("\n" + faith_df.to_string(index=False))

    # ── Date range ───────────────────────────────────────────────
    is_zoom = args.start is not None and args.end is not None
    if is_zoom:
        ds = pd.Timestamp(args.start)
        de = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        mask = (test.index >= ds) & (test.index <= de)
        time_axis = test.index[mask]
        test_vals = test[TARGET][mask].values
        shap_plot = {m: df.loc[mask] for m, df in shap_dict.items()}
        preds_plot = {m: preds[m][mask.values] for m in model_names}
        tag = f"zoom_{args.start.replace('-','')}_{args.end.replace('-','')}"
    else:
        time_axis = test.index
        test_vals = y_test
        shap_plot = shap_dict
        preds_plot = preds
        tag = "full"

    n = len(time_axis)
    atimes, astarts, aends = get_anomaly_regions(labels, TARGET, time_axis)
    print(f"\nTimesteps: {n}, Anomalies: {len(atimes)}")

    # Feature ranking
    imps = [df.abs().mean() for df in shap_plot.values()]
    combined = pd.concat(imps, axis=1).mean(axis=1)
    top_feats = combined.sort_values(ascending=False).index.tolist()
    print(f"Feature ranking: {top_feats}")

    # ── Plot ─────────────────────────────────────────────────────
    COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff"]
    MODEL_COLORS = {
        "GBM": "#1f77b4", "XGBoost": "#d62728", "DRF": "#2ca02c",
        "GLM": "#ff7f0e", "DeepLearning": "#9467bd",
        "StackedEnsemble": "#8c564b",
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
                color=MODEL_COLORS.get(m, "#999"),
                label=f"{m} (R²={r2_scores[m]:.3f})")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    title_range = f"{args.start} to {args.end}" if is_zoom else "Full Test"
    ax.set_title(f"{TARGET} — H2O AutoML Diverse Models ({title_range})",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
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
            c = COLORS[fi % len(COLORS)]
            ax.plot(time_axis[:len(mn)], mn, lw=1.2, color=c, alpha=0.85, label=feat)
            ax.fill_between(time_axis[:len(mn)], mn - sd, mn + sd, alpha=0.15, color=c)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel("SHAP value", fontsize=11)
    ax.set_title(f"Per-Feature SHAP (mean ± Rashomon σ across {len(model_names)} H2O models)",
                 fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.15)

    # Panel 3: Per-model total |SHAP|
    ax = axes[2]
    for m in model_names:
        total = smooth(shap_plot[m].abs().sum(axis=1).values[:n], sw)
        ax.plot(time_axis[:len(total)], total, lw=1.0,
                color=MODEL_COLORS.get(m, "#999"), alpha=0.8, label=m)
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
    ax.set_title(f"Total Rashomon Uncertainty ({len(model_names)} H2O models)", fontsize=11)
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

    out = RESULTS_DIR / f"h2o_rashomon_{TARGET}_{tag}.png"
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

    ax = axes2[0]
    ax.plot(time_axis, test_vals, color="steelblue", lw=0.9, alpha=0.9, label="Actual")
    for m in model_names:
        ax.plot(time_axis, preds_plot[m][:n], lw=0.6, alpha=0.6, ls="--",
                color=MODEL_COLORS.get(m, "#999"), label=m)
    shade_anom(ax, time_axis, astarts, aends)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.set_title(f"{TARGET} — Per-Feature Rashomon (H2O AutoML, {title_range})",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.2)

    for i, feat in enumerate(top_feats):
        ax = axes2[i + 1]
        c = COLORS[i % len(COLORS)]
        vals = []
        for m in model_names:
            if feat in shap_plot[m].columns:
                v = shap_plot[m][feat].values[:n]
                vals.append(v)
                ax.plot(time_axis[:len(v)], smooth(v, sw), lw=0.5, alpha=0.4,
                        ls="--", color=MODEL_COLORS.get(m, "#999"))
        if len(vals) >= 2:
            stacked = np.array(vals)
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

    out2 = RESULTS_DIR / f"h2o_perfeature_{TARGET}_{tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    h2o.cluster().shutdown(prompt=False)
    print("Done!")


if __name__ == "__main__":
    main()
