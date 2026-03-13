"""
H2O AutoML - TELCO Multivariate Time Series Forecasting + Comprehensive XAI
============================================================================
Supervised regression approach for time series forecasting on TELCO data.
Pipeline:
  1. Feature engineering: lags, rolling stats, time features, exogenous
  2. H2O AutoML training (GBM, XGBoost, DRF, DL, StackedEnsemble) — GLM excluded
  3. Evaluation with standard regression metrics
  4. XAI suite:
     - Variable importance (per model)
     - Partial Dependence Profiles (PDP) + ICE
     - SHAP contributions
     - Rashomon set analysis (models within threshold of best)
     - Cross-model feature importance comparison
     - Residual diagnostics

Usage:
    python h2o_forecast.py                          # default: TS1 target
    python h2o_forecast.py --target TS3             # TS3 target
    python h2o_forecast.py --target TS1 --horizon 288 --max_runtime 300
"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import h2o
from h2o.automl import H2OAutoML

# =====================================================================
# Directory setup
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
LABEL_DIR   = PROJECT_DIR / "TELCO_labels"
RESULTS_DIR = BASE_DIR / "results" / "h2o"
PLOTS_DIR   = RESULTS_DIR / "plots"
XAI_DIR     = RESULTS_DIR / "xai"
MODELS_DIR  = BASE_DIR / "models" / "h2o"

# =====================================================================
# Control Panel
# =====================================================================
CONTROL = {
    "target": "TS1",
    "prediction_length": 288,       # 288 steps = 1 day (5-min intervals)
    "freq": "5min",
    # Target lags (5-min: 12=1h, 72=6h, 144=12h, 288=1d)
    "target_lags": [1, 2, 3, 6, 12, 24, 72, 144, 288],
    "rolling_windows": [12, 72, 288],
    # Exogenous feature params (lightweight to save memory)
    "exog_lags": [1, 12, 288],
    "exog_rolling_windows": [12, 288],
    # H2O AutoML settings
    "max_runtime_secs": 300,
    "max_models": 30,
    "nfolds": 5,
    "sort_metric": "MAE",
    "seed": 42,
    # Diff / EWM
    "diff_lags": [1, 12, 288],
    "ewm_spans": [12, 72],
    # Rashomon factor (models within factor * best_MAE)
    "rashomon_factor": 4.0,
    # PDP: top N features to plot
    "pdp_top_n": 10,
}


# =====================================================================
# Feature Engineering
# =====================================================================
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour"]        = idx.hour
    df["minute"]      = idx.minute
    df["dayofweek"]   = idx.dayofweek
    df["dayofmonth"]  = idx.day
    df["month"]       = idx.month
    df["weekofyear"]  = idx.isocalendar().week.astype(int)
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"]   = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]   = np.cos(2 * np.pi * idx.month / 12)
    minutes_of_day    = idx.hour * 60 + idx.minute
    df["tod_sin"]     = np.sin(2 * np.pi * minutes_of_day / 1440)
    df["tod_cos"]     = np.cos(2 * np.pi * minutes_of_day / 1440)
    return df


def create_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    parts = {f"{target}_lag_{lag}": df[target].shift(lag) for lag in lags}
    return pd.concat([df, pd.DataFrame(parts, index=df.index)], axis=1)


def create_rolling_features(df: pd.DataFrame, target: str, windows: list[int]) -> pd.DataFrame:
    parts = {}
    for w in windows:
        roll = df[target].shift(1).rolling(window=w, min_periods=1)
        parts[f"{target}_rmean_{w}"] = roll.mean()
        parts[f"{target}_rstd_{w}"]  = roll.std()
        parts[f"{target}_rmin_{w}"]  = roll.min()
        parts[f"{target}_rmax_{w}"]  = roll.max()
    return pd.concat([df, pd.DataFrame(parts, index=df.index)], axis=1)


def create_diff_features(df: pd.DataFrame, target: str, diff_lags: list[int]) -> pd.DataFrame:
    parts = {f"{target}_diff_{d}": df[target].diff(d) for d in diff_lags}
    return pd.concat([df, pd.DataFrame(parts, index=df.index)], axis=1)


def create_ewm_features(df: pd.DataFrame, target: str, spans: list[int]) -> pd.DataFrame:
    parts = {f"{target}_ewm_{s}": df[target].shift(1).ewm(span=s, min_periods=1).mean() for s in spans}
    return pd.concat([df, pd.DataFrame(parts, index=df.index)], axis=1)


def create_exog_features(df: pd.DataFrame, target: str, exog_cols: list[str],
                         lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    parts = {}
    for col in exog_cols:
        if col == target:
            continue
        for lag in lags:
            parts[f"{col}_lag_{lag}"] = df[col].shift(lag)
        for w in rolling_windows:
            parts[f"{col}_rmean_{w}"] = df[col].shift(1).rolling(window=w, min_periods=1).mean()
    return pd.concat([df, pd.DataFrame(parts, index=df.index)], axis=1)


def build_features(df: pd.DataFrame, target: str, ctrl: dict) -> pd.DataFrame:
    ts_cols = [c for c in df.columns if c.startswith("TS")]
    df = create_time_features(df)
    df = create_lag_features(df, target, ctrl["target_lags"])
    df = create_rolling_features(df, target, ctrl["rolling_windows"])
    df = create_diff_features(df, target, ctrl["diff_lags"])
    df = create_ewm_features(df, target, ctrl["ewm_spans"])
    df = create_exog_features(df, target, ts_cols, ctrl["exog_lags"], ctrl["exog_rolling_windows"])
    return df


# =====================================================================
# Data loading
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    return train, val, test


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# =====================================================================
# Plot: Forecast vs Actual
# =====================================================================
def plot_forecast(y_train_tail: pd.Series, y_test: pd.Series,
                  y_pred: np.ndarray, target: str, model_name: str, out_dir: Path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(y_train_tail.index, y_train_tail.values, color="steelblue", lw=1.5, label="Train (tail)")
    ax.plot(y_test.index, y_test.values, color="forestgreen", lw=2, label="Actual (test)")
    ax.plot(y_test.index[:len(y_pred)], y_pred, color="crimson", lw=2, ls="--", label=f"Forecast ({model_name})")
    ax.axvspan(y_test.index[0], y_test.index[min(len(y_pred)-1, len(y_test)-1)], alpha=0.08, color="forestgreen")
    ax.set_title(f"TELCO {target} - {model_name} Forecast", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(target)
    ax.set_xlabel("Time")
    plt.tight_layout()
    fig.savefig(out_dir / f"forecast_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot: Multi-model forecast comparison
# =====================================================================
def plot_multi_model_forecast(aml, test_h2o, df_test_feat, target: str,
                              train_tail: pd.Series, horizon: int, out_dir: Path):
    """Overlay predictions from all leaderboard models on the test set."""
    if not HAS_MPL:
        return
    lb = aml.leaderboard.as_data_frame()
    test_idx = df_test_feat.index[:horizon]
    test_vals = df_test_feat[target].values[:horizon]
    tab_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=1.5, label="Train (tail)")
    ax.plot(test_idx, test_vals, color="forestgreen", lw=2.5, label="Actual (test)", zorder=10)
    ax.axvspan(test_idx[0], test_idx[-1], alpha=0.06, color="forestgreen")

    for i, row in lb.head(6).iterrows():
        mid = row["model_id"]
        try:
            model = h2o.get_model(mid)
            preds = model.predict(test_h2o).as_data_frame()["predict"].values[:horizon]
            short_name = mid.split("_")[0]
            mae_val = row["mae"]
            ax.plot(test_idx, preds, ls="--", lw=1.5, color=tab_colors[i % 10],
                    label=f"{short_name} (MAE={mae_val:.4f})", alpha=0.85)
        except Exception:
            pass

    ax.set_title(f"Multi-Model Forecast Comparison - TELCO {target}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(target)
    ax.set_xlabel("Time")
    plt.tight_layout()
    fig.savefig(out_dir / f"multi_model_forecast_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Multi-model forecast comparison saved")


# =====================================================================
# XAI: Variable Importance
# =====================================================================
def get_varimp_df(model) -> pd.DataFrame | None:
    """Get variable importance as DataFrame, handling tree and other model types."""
    try:
        varimp = model.varimp(use_pandas=True)
        if varimp is not None and not varimp.empty:
            # Tree models: columns are variable, relative_importance, scaled_importance, percentage
            if "relative_importance" in varimp.columns:
                return varimp
            # Fallback: use whichever numeric column is available
            num_cols = varimp.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                varimp = varimp.rename(columns={num_cols[0]: "relative_importance"})
                return varimp
    except Exception:
        pass
    # Fallback: extract from coefficients table (for linear models)
    try:
        coef = model.coef()
        if coef:
            coef_abs = {k: abs(v) for k, v in coef.items() if k != "Intercept"}
            if coef_abs:
                varimp = pd.DataFrame(
                    sorted(coef_abs.items(), key=lambda x: x[1], reverse=True),
                    columns=["variable", "relative_importance"],
                )
                return varimp
    except Exception:
        pass
    return None


def plot_variable_importance(model, out_dir: Path, target: str, model_id: str, top_n: int = 30):
    if not HAS_MPL:
        return None
    varimp = get_varimp_df(model)
    if varimp is None or varimp.empty:
        print(f"  [WARN] No variable importance available for {model_id}")
        return None
    varimp = varimp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(varimp["variable"][::-1], varimp["relative_importance"][::-1], color="steelblue")
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"Variable Importance - {target} ({model_id})", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    safe_name = model_id.replace("/", "_")
    fig.savefig(out_dir / f"varimp_{target}_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return varimp


# =====================================================================
# XAI: Partial Dependence Profiles (PDP) - per model
# =====================================================================
def _get_top_features(model, feature_cols, top_n):
    varimp = get_varimp_df(model)
    if varimp is not None and not varimp.empty:
        top_features = varimp["variable"].head(top_n).tolist()
        return [f for f in top_features if f in feature_cols]
    return feature_cols[:top_n]


def _compute_pdp(model, train_h2o, feat, nbins=30):
    pdp_data = model.partial_plot(frame=train_h2o, cols=[feat], plot=False, nbins=nbins)
    if pdp_data and len(pdp_data) > 0:
        return pdp_data[0].as_data_frame()
    return None


def plot_pdp_single_model(model, train_h2o, feature_cols, target,
                          out_dir, model_label, top_n=10):
    """PDP for a single model. Saves individual + grid plots."""
    if not HAS_MPL:
        return
    safe_label = model_label.replace("/", "_").replace(" ", "_")
    top_features = _get_top_features(model, feature_cols, top_n)

    pdp_dir = out_dir / "pdp" / safe_label
    pdp_dir.mkdir(parents=True, exist_ok=True)

    print(f"    [{model_label}] PDP for top {len(top_features)} features...")

    # Individual PDP plots
    for feat in top_features:
        try:
            pdf = _compute_pdp(model, train_h2o, feat)
            if pdf is None:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(pdf.iloc[:, 0], pdf["mean_response"], color="crimson", lw=2.5,
                    label="Mean response (PDP)")
            if "stddev_response" in pdf.columns:
                ax.fill_between(pdf.iloc[:, 0],
                                pdf["mean_response"] - pdf["stddev_response"],
                                pdf["mean_response"] + pdf["stddev_response"],
                                alpha=0.15, color="crimson", label="+/- 1 Std Dev")
            ax.set_xlabel(feat, fontsize=11)
            ax.set_ylabel(f"Partial Dependence ({target})", fontsize=11)
            ax.set_title(f"PDP: {feat} -> {target}  [{model_label}]",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            fig.savefig(pdp_dir / f"pdp_{feat}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"      [WARN] PDP failed for {feat}: {e}")

    # Grid
    n_feats = min(len(top_features), 9)
    if n_feats > 0:
        ncols = 3
        nrows = (n_feats + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for i, feat in enumerate(top_features[:n_feats]):
            ax = axes[i // ncols, i % ncols]
            try:
                pdf = _compute_pdp(model, train_h2o, feat)
                if pdf is not None:
                    ax.plot(pdf.iloc[:, 0], pdf["mean_response"], color="crimson", lw=2)
                    if "stddev_response" in pdf.columns:
                        ax.fill_between(pdf.iloc[:, 0],
                                        pdf["mean_response"] - pdf["stddev_response"],
                                        pdf["mean_response"] + pdf["stddev_response"],
                                        alpha=0.15, color="crimson")
                    ax.set_title(feat, fontsize=10, fontweight="bold")
                    ax.set_ylabel("Partial Dep.")
                    ax.grid(True, alpha=0.25)
            except Exception:
                ax.set_title(f"{feat} (failed)", fontsize=10)
        for j in range(n_feats, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)
        fig.suptitle(f"Partial Dependence Profiles - {target}  [{model_label}]",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(out_dir / f"pdp_grid_{target}_{safe_label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"      Saved to {pdp_dir}")


def plot_pdp_comparison(models_dict, train_h2o, feature_cols, target,
                        out_dir, top_n=8):
    """Overlay PDP from multiple models on same axes for comparison.

    models_dict: {label: h2o_model}
    Shows how different model types learn feature-target relationships
    (e.g. GBM vs XGBoost vs DRF step patterns).
    """
    if not HAS_MPL or len(models_dict) < 2:
        return

    print(f"  Generating PDP model comparison (top {top_n} features)...")

    # Union of top features across models
    all_top = []
    for label, model in models_dict.items():
        all_top.extend(_get_top_features(model, feature_cols, top_n))
    # Deduplicate preserving order
    seen = set()
    top_features = []
    for f in all_top:
        if f not in seen:
            seen.add(f)
            top_features.append(f)
    top_features = top_features[:top_n]

    colors = {"GBM": "steelblue", "DRF": "forestgreen", "XGBoost": "darkorange"}
    default_colors = list(plt.cm.Set1(np.linspace(0, 1, 8)))

    n_feats = min(len(top_features), 9)
    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)

    for i, feat in enumerate(top_features[:n_feats]):
        ax = axes[i // ncols, i % ncols]
        for ci, (label, model) in enumerate(models_dict.items()):
            try:
                pdf = _compute_pdp(model, train_h2o, feat)
                if pdf is None:
                    continue
                c = colors.get(label.split("_")[0], default_colors[ci % len(default_colors)])
                ax.plot(pdf.iloc[:, 0], pdf["mean_response"], lw=2.5, label=label, color=c)
                if "stddev_response" in pdf.columns:
                    ax.fill_between(pdf.iloc[:, 0],
                                    pdf["mean_response"] - pdf["stddev_response"],
                                    pdf["mean_response"] + pdf["stddev_response"],
                                    alpha=0.08, color=c)
            except Exception:
                pass
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Partial Dep.")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n_feats, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"PDP Comparison Across Model Types - {target}\n"
                 f"(Feature-target relationships across different models)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / f"pdp_comparison_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    PDP comparison saved")


# =====================================================================
# XAI: SHAP Contributions
# =====================================================================
def plot_shap(model, test_h2o, target: str, out_dir: Path,
              model_id: str, top_n: int = 20, max_rows: int = 2000):
    if not HAS_MPL:
        return
    try:
        # Sample test data for speed (SHAP on full 26k rows is very slow)
        n_rows = test_h2o.nrows
        if n_rows > max_rows:
            sample_h2o = test_h2o[0:max_rows, :]
        else:
            sample_h2o = test_h2o
        # Direct predict_contributions; fallback with background_frame if needed
        try:
            contribs = model.predict_contributions(sample_h2o).as_data_frame()
        except Exception:
            bg = test_h2o[0:500, :]
            contribs = model.predict_contributions(sample_h2o, background_frame=bg).as_data_frame()

        if "BiasTerm" in contribs.columns:
            contribs = contribs.drop(columns=["BiasTerm"])

        mean_abs = contribs.abs().mean().sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(mean_abs.index[::-1], mean_abs.values[::-1], color="darkorange")
        ax.set_xlabel("Mean |SHAP Contribution|")
        ax.set_title(f"SHAP Feature Importance - {target} ({model_id})", fontweight="bold")
        ax.grid(True, alpha=0.25, axis="x")
        plt.tight_layout()
        safe_name = model_id.replace("/", "_")
        fig.savefig(out_dir / f"shap_{target}_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save CSV
        mean_abs.to_frame("mean_abs_shap").to_csv(out_dir / f"shap_values_{target}.csv")
        print(f"    SHAP plot saved ({model_id})")
    except Exception as e:
        print(f"  [WARN] SHAP failed for {model_id}: {e}")


# =====================================================================
# XAI: Rashomon Set Analysis
# =====================================================================
def rashomon_analysis(aml, train_h2o, test_h2o, target: str,
                     feature_cols: list[str], out_dir: Path, factor: float = 1.5):
    if not HAS_MPL:
        return
    print(f"\n  Rashomon Set Analysis (factor={factor}x)...")
    rashomon_dir = out_dir / "rashomon"
    rashomon_dir.mkdir(exist_ok=True)

    lb = aml.leaderboard.as_data_frame()
    if "mae" not in lb.columns:
        print("    [WARN] MAE not in leaderboard, skipping Rashomon analysis")
        return

    best_mae = lb["mae"].iloc[0]
    threshold = best_mae * factor
    rashomon_models = lb[lb["mae"] <= threshold].copy()
    print(f"    Best MAE: {best_mae:.6f}, threshold: {threshold:.6f}")
    print(f"    Models in Rashomon set: {len(rashomon_models)} / {len(lb)}")

    if len(rashomon_models) < 2:
        print("    [INFO] Not enough models in Rashomon set, skipping")
        return

    rashomon_models.to_csv(rashomon_dir / f"rashomon_set_{target}.csv", index=False)

    # Collect variable importance from each model in Rashomon set
    all_varimp = {}
    for _, row in rashomon_models.iterrows():
        mid = row["model_id"]
        try:
            model = h2o.get_model(mid)
            vimp = get_varimp_df(model)
            if vimp is not None and not vimp.empty:
                vimp_dict = dict(zip(vimp["variable"], vimp["relative_importance"]))
                all_varimp[mid] = vimp_dict
        except Exception:
            pass

    if len(all_varimp) < 2:
        print("    [INFO] Not enough models with varimp, skipping comparison")
        return

    # Create comparison DataFrame
    varimp_df = pd.DataFrame(all_varimp).fillna(0)

    # Top features (union of top-15 from each model)
    top_feats = set()
    for mid, vimp_dict in all_varimp.items():
        sorted_feats = sorted(vimp_dict, key=vimp_dict.get, reverse=True)[:15]
        top_feats.update(sorted_feats)
    top_feats = sorted(top_feats, key=lambda f: varimp_df.loc[f].mean() if f in varimp_df.index else 0, reverse=True)[:20]
    top_feats = [f for f in top_feats if f in varimp_df.index]

    # Save comparison CSV
    varimp_df.loc[top_feats].to_csv(rashomon_dir / f"rashomon_varimp_comparison_{target}.csv")

    # ---- Plot 1: Heatmap of feature importance across Rashomon models ----
    plot_df = varimp_df.loc[top_feats]
    # Normalize per model (column-wise) for comparison
    plot_norm = plot_df.div(plot_df.max(axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(max(10, len(all_varimp) * 1.5), max(8, len(top_feats) * 0.4)))
    # Short model names for readability
    short_names = [m.split("_")[0] + "_" + m.split("_")[1] if "_" in m else m[:20] for m in plot_norm.columns]
    im = ax.imshow(plot_norm.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_title(f"Rashomon Set: Feature Importance Comparison - {target}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    plt.colorbar(im, ax=ax, label="Normalized Importance", shrink=0.8)
    plt.tight_layout()
    fig.savefig(rashomon_dir / f"rashomon_heatmap_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: Grouped bar chart (top 10 features, all models) ----
    top10 = top_feats[:10]
    plot_data = varimp_df.loc[top10]
    n_models = len(plot_data.columns)
    n_feats = len(top10)
    x = np.arange(n_feats)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (mid, color) in enumerate(zip(plot_data.columns, colors)):
        short = mid.split("_")[0] + "_" + mid.split("_")[1] if "_" in mid else mid[:15]
        ax.bar(x + i * width, plot_data[mid].values, width, label=short, color=color, alpha=0.85)
    ax.set_xticks(x + width * n_models / 2)
    ax.set_xticklabels(top10, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative Importance")
    ax.set_title(f"Rashomon Set: Feature Contributions Across Models - {target}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(rashomon_dir / f"rashomon_bars_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: Rank stability (how feature ranks change across models) ----
    rank_df = varimp_df.loc[top_feats].rank(ascending=False, axis=0)
    fig, ax = plt.subplots(figsize=(max(10, len(all_varimp) * 1.5), max(6, len(top_feats) * 0.35)))
    for feat in top_feats[:12]:
        ranks = rank_df.loc[feat].values
        ax.plot(range(len(ranks)), ranks, marker="o", lw=1.5, label=feat, alpha=0.8)
    ax.set_xticks(range(len(rank_df.columns)))
    short_names = [m.split("_")[0] + "_" + m.split("_")[1] if "_" in m else m[:15] for m in rank_df.columns]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Feature Rank (1 = most important)")
    ax.set_title(f"Rashomon Set: Feature Rank Stability - {target}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(rashomon_dir / f"rashomon_rank_stability_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    Rashomon analysis saved to {rashomon_dir}")


# =====================================================================
# XAI: Residual Diagnostics
# =====================================================================
def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, target: str, out_dir: Path):
    if not HAS_MPL:
        return
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Histogram
    axes[0, 0].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(0, color="crimson", ls="--", lw=1.5)
    axes[0, 0].set_title("Residual Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Residual")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.25)

    # 2) Residual vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.2, s=8, color="steelblue")
    axes[0, 1].axhline(0, color="crimson", ls="--", lw=1.5)
    axes[0, 1].set_title("Residual vs Predicted", fontweight="bold")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residual")
    axes[0, 1].grid(True, alpha=0.25)

    # 3) Q-Q plot
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical = np.sort(np.random.RandomState(42).normal(0, np.std(residuals), n))
    axes[1, 0].scatter(theoretical, sorted_res, alpha=0.2, s=8, color="steelblue")
    lims = [min(theoretical.min(), sorted_res.min()), max(theoretical.max(), sorted_res.max())]
    axes[1, 0].plot(lims, lims, color="crimson", ls="--", lw=1.5)
    axes[1, 0].set_title("Q-Q Plot", fontweight="bold")
    axes[1, 0].set_xlabel("Theoretical Quantile")
    axes[1, 0].set_ylabel("Observed Quantile")
    axes[1, 0].grid(True, alpha=0.25)

    # 4) Residual over time (first 2000 points)
    n_show = min(2000, len(residuals))
    axes[1, 1].plot(range(n_show), residuals[:n_show], color="steelblue", lw=0.5, alpha=0.7)
    axes[1, 1].axhline(0, color="crimson", ls="--", lw=1.5)
    axes[1, 1].set_title("Residual Over Time (first 2000)", fontweight="bold")
    axes[1, 1].set_xlabel("Time Index")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"Residual Diagnostics - {target}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"residuals_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Report generation
# =====================================================================
def generate_report(target: str, metrics: dict, leaderboard_df: pd.DataFrame,
                    ctrl: dict, elapsed: float, out_dir: Path):
    lines = [
        f"# TELCO {target} - H2O AutoML Forecast Report",
        f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Training time**: {elapsed:.1f} seconds",
        f"**Forecast horizon**: {ctrl['prediction_length']} steps ({ctrl['prediction_length']*5} minutes)",
        f"\n## Test Set Metrics",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v:.6f} |" if not np.isnan(v) else f"| {k} | N/A |")

    lines.append(f"\n## Leaderboard (Top 10)")
    lines.append(leaderboard_df.head(10).to_markdown(index=False))

    lines.append(f"\n## Control Panel Settings")
    lines.append("```json")
    lines.append(json.dumps(ctrl, indent=2, default=str))
    lines.append("```")

    lines.append(f"\n## Output Files")
    lines.append(f"- `plots/forecast_{target}.png` - Best model forecast")
    lines.append(f"- `plots/multi_model_forecast_{target}.png` - All models comparison")
    lines.append(f"- `xai/varimp_{target}_*.png` - Variable importance (per model)")
    lines.append(f"- `xai/pdp/` - Partial Dependence Profiles per model type")
    lines.append(f"- `xai/pdp_grid_{target}_*.png` - PDP grids per model type")
    lines.append(f"- `xai/pdp_comparison_{target}.png` - Cross-model PDP overlay")
    lines.append(f"- `xai/shap_{target}_*.png` - SHAP contributions (per model)")
    lines.append(f"- `xai/rashomon/` - Rashomon set analysis")
    lines.append(f"- `xai/residuals_{target}.png` - Residual diagnostics")

    (out_dir / f"REPORT_{target}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {out_dir / f'REPORT_{target}.md'}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="H2O AutoML TELCO Forecast")
    parser.add_argument("--target", type=str, default=CONTROL["target"])
    parser.add_argument("--horizon", type=int, default=CONTROL["prediction_length"])
    parser.add_argument("--max_runtime", type=int, default=CONTROL["max_runtime_secs"])
    args = parser.parse_args()

    ctrl = CONTROL.copy()
    ctrl["target"] = args.target
    ctrl["prediction_length"] = args.horizon
    ctrl["max_runtime_secs"] = args.max_runtime
    target = ctrl["target"]

    print("=" * 80)
    print(f"H2O AutoML - TELCO {target} Forecast + XAI")
    print("=" * 80)

    for d in [RESULTS_DIR, PLOTS_DIR, XAI_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ----- Load Data -----
    print("\n[1/7] Loading data...")
    train_raw, val_raw, test_raw = load_telco_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    print(f"  Train+Val: {full_train.shape}, Test: {test_raw.shape}")

    # ----- Feature Engineering -----
    print(f"\n[2/7] Feature engineering ({target})...")
    df_combined = pd.concat([full_train, test_raw])
    df_combined = df_combined[~df_combined.index.duplicated(keep="first")].sort_index()
    df_combined_feat = build_features(df_combined.copy(), target, ctrl)

    max_lag = max(ctrl["target_lags"] + ctrl["rolling_windows"])
    test_start = test_raw.index[0]
    df_train_feat = df_combined_feat.loc[:test_start].iloc[max_lag:-1].dropna(subset=[target])
    df_test_feat  = df_combined_feat.loc[test_start:].dropna(subset=[target])

    exclude = {target, "time"}
    feature_cols = [c for c in df_train_feat.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if df_train_feat[c].notna().sum() > 0]

    df_train_feat[feature_cols] = df_train_feat[feature_cols].ffill().bfill().fillna(0)
    df_test_feat[feature_cols]  = df_test_feat[feature_cols].ffill().bfill().fillna(0)

    print(f"  Features: {len(feature_cols)}, Train rows: {len(df_train_feat)}, Test rows: {len(df_test_feat)}")
    pd.DataFrame({"feature": feature_cols}).to_csv(RESULTS_DIR / f"feature_list_{target}.csv", index=False)

    # ----- H2O Init -----
    print(f"\n[3/7] Starting H2O...")
    h2o.init(nthreads=-1, max_mem_size="4G")
    train_h2o = h2o.H2OFrame(df_train_feat[[target] + feature_cols].reset_index(drop=True))
    test_h2o  = h2o.H2OFrame(df_test_feat[[target] + feature_cols].reset_index(drop=True))

    # ----- AutoML Training -----
    print(f"\n[4/8] AutoML training (max {ctrl['max_runtime_secs']}s, max {ctrl['max_models']} models)...")
    aml = H2OAutoML(
        max_runtime_secs=ctrl["max_runtime_secs"],
        max_models=ctrl["max_models"],
        nfolds=ctrl["nfolds"],
        sort_metric=ctrl["sort_metric"],
        seed=ctrl["seed"],
        exclude_algos=["GLM"],
    )
    start_time = time.time()
    aml.train(x=feature_cols, y=target, training_frame=train_h2o)
    elapsed = time.time() - start_time
    print(f"  Training complete. Time: {elapsed:.1f}s")

    lb = aml.leaderboard.as_data_frame()
    print(f"\n  Leaderboard (Top 10):")
    print(lb.head(10).to_string(index=False))
    lb.to_csv(RESULTS_DIR / f"leaderboard_{target}.csv", index=False)

    # ----- Save Models -----
    print(f"\n[5/8] Saving models...")
    for i, row in lb.iterrows():
        mid = row["model_id"]
        try:
            m = h2o.get_model(mid)
            saved_path = h2o.save_model(model=m, path=str(MODELS_DIR), force=True)
            if i < 3:
                print(f"    Saved: {mid}")
        except Exception as e:
            if i < 3:
                print(f"    [WARN] Could not save {mid}: {e}")
    print(f"    All models saved to {MODELS_DIR}")

    # ----- Prediction & Evaluation -----
    print(f"\n[6/8] Prediction & evaluation...")
    best_model = aml.leader
    best_name = best_model.model_id
    preds_h2o = best_model.predict(test_h2o)
    y_pred = preds_h2o.as_data_frame()["predict"].values
    y_test = df_test_feat[target].values

    metrics = compute_metrics(y_test, y_pred)
    print(f"\n  Test metrics ({best_name}):")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}" if not np.isnan(v) else f"    {k}: N/A")

    forecast_df = pd.DataFrame({"time": df_test_feat.index, "actual": y_test, "predicted": y_pred})
    forecast_df.to_csv(RESULTS_DIR / f"forecasts_{target}.csv", index=False)
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f"metrics_{target}.csv", index=False)

    # ----- Forecast Plots -----
    print(f"\n[7/8] Plots...")
    horizon = min(ctrl["prediction_length"], len(y_pred))
    train_tail = full_train[target].iloc[-500:]
    test_series = pd.Series(y_test[:horizon], index=df_test_feat.index[:horizon])
    plot_forecast(train_tail, test_series, y_pred[:horizon], target, best_name, PLOTS_DIR)

    # Multi-model forecast comparison (all leaderboard models overlaid)
    plot_multi_model_forecast(aml, test_h2o, df_test_feat, target, train_tail, horizon, PLOTS_DIR)

    # ----- XAI Suite -----
    print(f"\n[8/8] XAI Analysis...")

    # Identify second-best model of a different type for comparison
    second_model = None
    second_name = None
    leader_type = best_name.split("_")[0]  # e.g. "GBM", "XGBoost", "StackedEnsemble"
    for _, row in lb.iterrows():
        mid = row["model_id"]
        mtype = mid.split("_")[0]
        if mtype != leader_type and "StackedEnsemble" not in mid:
            second_model = h2o.get_model(mid)
            second_name = mid
            break

    # 8a) Variable importance (leader + second model type)
    print(f"  Variable importance...")
    plot_variable_importance(best_model, XAI_DIR, target, best_name)
    if second_model:
        plot_variable_importance(second_model, XAI_DIR, target, second_name)

    # 8b) Partial Dependence Profiles
    print(f"  Partial Dependence Profiles...")
    leader_label = best_name.split("_")[0]
    plot_pdp_single_model(best_model, train_h2o, feature_cols, target,
                          XAI_DIR, model_label=leader_label, top_n=ctrl["pdp_top_n"])
    if second_model:
        second_label = second_name.split("_")[0]
        plot_pdp_single_model(second_model, train_h2o, feature_cols, target,
                              XAI_DIR, model_label=second_label, top_n=ctrl["pdp_top_n"])
        # Cross-model PDP comparison overlay
        plot_pdp_comparison(
            {leader_label: best_model, second_label: second_model},
            train_h2o, feature_cols, target, XAI_DIR, top_n=ctrl["pdp_top_n"],
        )

    # 8c) SHAP contributions (leader + second model type)
    plot_shap(best_model, test_h2o, target, XAI_DIR, best_name)
    if second_model:
        plot_shap(second_model, test_h2o, target, XAI_DIR, second_name)

    # 8d) Rashomon set analysis
    rashomon_analysis(aml, train_h2o, test_h2o, target, feature_cols, XAI_DIR, factor=ctrl["rashomon_factor"])

    # 8e) Residual diagnostics
    plot_residuals(y_test, y_pred, target, XAI_DIR)

    # ----- Report -----
    generate_report(target, metrics, lb, ctrl, elapsed, RESULTS_DIR)

    # ----- Summary -----
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"  Target: {target}")
    print(f"  Best model: {best_name}")
    print(f"  Test MAE: {metrics['MAE']:.6f}, RMSE: {metrics['RMSE']:.6f}, R2: {metrics['R2']:.6f}")
    print(f"\nOutputs:")
    print(f"  - Results: {RESULTS_DIR}")
    print(f"  - Plots:   {PLOTS_DIR}")
    print(f"  - XAI:     {XAI_DIR}")
    print("=" * 80)

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
