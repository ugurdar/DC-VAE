"""
Surrogate SHAP Analysis for AutoGluon Time Series Models
========================================================
Applies TreeSHAP to ALL AutoGluon models via LightGBM surrogate approach:
  DirectTabular, RecursiveTabular, ETS, Theta, Naive, SeasonalNaive, WeightedEnsemble

Method (inspired by arxiv:2510.08739):
  1. Get each model's predictions on test data
  2. Build tabular features (lags, rolling stats, calendar) from actual data
  3. Train LightGBM surrogate: features → model_prediction
  4. Validate surrogate faithfulness (R²)
  5. Compute TreeSHAP on surrogate
  6. Generate visualizations

Usage:
    python surrogate_shap_analysis.py
    python surrogate_shap_analysis.py --models DirectTabular RecursiveTabular
    python surrogate_shap_analysis.py --top_n 20
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "surrogate_shap"
MODELS_DIR  = BASE_DIR / "models" / "autogluon" / "multi"

PREDICTION_LENGTH = 288
FREQ = "5min"

# Lag / rolling config (matching AutoGluon internals)
TARGET_LAGS = [1, 2, 3, 6, 12, 24, 72, 144, 288]
ROLLING_WINDOWS = [12, 72, 288]


# =====================================================================
# Data loading
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    return train, val, test


def to_multi_series_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    ts_cols = sorted([c for c in df.columns if c.startswith("TS")])
    records = []
    for col in ts_cols:
        temp = df[[col]].copy().rename(columns={col: "target"})
        temp["item_id"] = col
        temp = temp.reset_index().rename(columns={"time": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        records.append(temp)
    long_df = pd.concat(records, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id", timestamp_column="timestamp")


# =====================================================================
# Feature Engineering
# =====================================================================
def build_features_for_series(values: np.ndarray, timestamps: pd.DatetimeIndex,
                              forecast_start_idx: int, prediction_length: int,
                              series_id: str) -> pd.DataFrame:
    """
    Build tabular features at each forecast step for a single series.

    For forecast step h (1..prediction_length):
      - Lags are computed from actual data at (forecast_start_idx + h - lag)
      - Rolling stats are computed from a window ending before the forecast step
      - Calendar features from the timestamp
    """
    rows = []
    for h in range(prediction_length):
        t = forecast_start_idx + h  # absolute index in values array
        feat = {}

        # Forecast horizon step
        feat["step"] = h + 1
        feat["step_norm"] = (h + 1) / prediction_length

        # Lag features (from actual data)
        for lag in TARGET_LAGS:
            idx = t - lag
            feat[f"lag_{lag}"] = values[idx] if idx >= 0 else np.nan

        # Rolling statistics
        for w in ROLLING_WINDOWS:
            start = max(0, t - w)
            window_vals = values[start:t]
            if len(window_vals) > 0:
                feat[f"roll_mean_{w}"] = np.mean(window_vals)
                feat[f"roll_std_{w}"] = np.std(window_vals)
                feat[f"roll_min_{w}"] = np.min(window_vals)
                feat[f"roll_max_{w}"] = np.max(window_vals)
            else:
                feat[f"roll_mean_{w}"] = np.nan
                feat[f"roll_std_{w}"] = np.nan
                feat[f"roll_min_{w}"] = np.nan
                feat[f"roll_max_{w}"] = np.nan

        # Diff features (momentum / trend)
        if t >= 1:
            feat["diff_1"] = values[t-1] - values[t-2] if t >= 2 else 0.0
        else:
            feat["diff_1"] = 0.0
        if t >= 12:
            feat["diff_12"] = values[t-1] - values[t-12]
        else:
            feat["diff_12"] = np.nan
        if t >= 288:
            feat["diff_288"] = values[t-1] - values[t-288]
        else:
            feat["diff_288"] = np.nan

        # Calendar features
        ts = timestamps[t]
        feat["hour"] = ts.hour
        feat["minute"] = ts.minute
        feat["day_of_week"] = ts.dayofweek
        feat["is_weekend"] = int(ts.dayofweek >= 5)
        feat["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)

        # Series-level statistics (contextual)
        train_vals = values[:forecast_start_idx]
        feat["series_mean"] = np.mean(train_vals)
        feat["series_std"] = np.std(train_vals)
        feat["series_last"] = values[forecast_start_idx - 1]

        # Series identity
        feat["series_id"] = series_id

        rows.append(feat)

    df = pd.DataFrame(rows)
    return df


def build_all_features(full_data: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """Build features for ALL series, pooled into a single DataFrame."""
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    all_frames = []
    forecast_start_idx = len(full_data) - prediction_length

    for col in ts_cols:
        values = full_data[col].values
        timestamps = full_data.index
        feat_df = build_features_for_series(
            values, timestamps, forecast_start_idx, prediction_length, col
        )
        feat_df["_series"] = col
        feat_df["_step_idx"] = range(prediction_length)
        all_frames.append(feat_df)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


# =====================================================================
# Surrogate Training
# =====================================================================
def train_surrogate(X: pd.DataFrame, y: np.ndarray, model_name: str) -> lgb.LGBMRegressor:
    """Train a LightGBM surrogate model on features → predictions."""
    surrogate = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    surrogate.fit(X, y)
    return surrogate


def evaluate_surrogate(surrogate, X, y_true) -> dict:
    """Evaluate surrogate faithfulness."""
    y_pred = surrogate.predict(X)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Correlation": corr}


# =====================================================================
# SHAP Computation
# =====================================================================
def compute_shap_values(surrogate, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)
    return shap_values


# =====================================================================
# Plot 1: SHAP Bar (mean |SHAP| per feature)
# =====================================================================
def plot_shap_bar(shap_vals, feature_names, model_name, out_dir, top_n=20):
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    names = [feature_names[i] for i in order]
    vals = mean_abs[order]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(vals)))
    ax.barh(range(len(names))[::-1], vals, color=colors)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance — {model_name}\n(Surrogate TreeSHAP, all series)",
                 fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_bar_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 2: SHAP Summary / Beeswarm
# =====================================================================
def plot_shap_summary(shap_vals, X, feature_names, model_name, out_dir, top_n=20):
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    for rank, feat_idx in enumerate(reversed(order)):
        sv = shap_vals[:, feat_idx]
        fv = X.iloc[:, feat_idx].values

        fmin, fmax = np.nanmin(fv), np.nanmax(fv)
        if fmax - fmin > 1e-10:
            normed = (fv - fmin) / (fmax - fmin)
        else:
            normed = np.zeros_like(fv)

        y_jitter = rank + np.random.uniform(-0.3, 0.3, size=len(sv))
        n_show = min(600, len(sv))
        idx = np.random.choice(len(sv), n_show, replace=False)

        scatter = ax.scatter(
            sv[idx], y_jitter[idx], c=normed[idx], cmap="coolwarm",
            s=8, alpha=0.6, edgecolors="none", vmin=0, vmax=1,
        )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in reversed(order)], fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("SHAP Value", fontsize=11)
    ax.set_title(f"SHAP Summary — {model_name}\n(Red=high feature value, Blue=low)",
                 fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.15, axis="x")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Feature Value (normalized)", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 3: SHAP Dependence (top features)
# =====================================================================
def plot_shap_dependence(shap_vals, X, feature_names, model_name, out_dir, top_n=6):
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, feat_idx in enumerate(order):
        ax = axes[i // ncols, i % ncols]
        x = X.iloc[:, feat_idx].values
        y = shap_vals[:, feat_idx]

        n_show = min(1000, len(x))
        idx = np.random.choice(len(x), n_show, replace=False)
        ax.scatter(x[idx], y[idx], s=6, alpha=0.4, color="steelblue", edgecolors="none")
        ax.axhline(0, color="crimson", lw=0.8, ls="--")
        ax.set_xlabel(feature_names[feat_idx], fontsize=9)
        ax.set_ylabel("SHAP Value", fontsize=9)
        ax.set_title(feature_names[feat_idx], fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2)

    for j in range(top_n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"SHAP Dependence — {model_name}", fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_dependence_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 4: Temporal SHAP Heatmap (per series)
# =====================================================================
def plot_temporal_shap_heatmap(shap_vals, feature_df, feature_names, model_name, out_dir,
                                top_n=15):
    """Heatmap per series: features (y) × forecast step (x), color = SHAP value."""
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]

    series_list = sorted(feature_df["_series"].unique())
    n_series = len(series_list)

    fig, axes = plt.subplots(n_series, 1,
                             figsize=(16, max(3, top_n * 0.3) * n_series),
                             squeeze=False)

    for si, series_id in enumerate(series_list):
        ax = axes[si, 0]
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values][:, order]  # (steps, top_n)
        matrix = series_shap.T  # (top_n, steps)

        vmax = np.percentile(np.abs(matrix), 97)
        if vmax < 1e-10:
            vmax = 1.0
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                       interpolation="nearest", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(top_feat_names)))
        ax.set_yticklabels(top_feat_names, fontsize=7)
        ax.set_title(f"{series_id}", fontsize=10, fontweight="bold")

        # X-axis: forecast steps
        step_ticks = np.arange(0, matrix.shape[1], max(1, matrix.shape[1] // 8))
        ax.set_xticks(step_ticks)
        ax.set_xticklabels([f"h={s+1}" for s in step_ticks], fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01)

    axes[-1, 0].set_xlabel("Forecast Step", fontsize=10)
    fig.suptitle(f"Temporal SHAP Heatmap — {model_name}\n"
                 f"Red = pushes UP, Blue = pushes DOWN",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_shap_{model_name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 5: Per-Series SHAP Bar
# =====================================================================
def plot_per_series_shap(shap_vals, feature_df, feature_names, model_name, out_dir, top_n=10):
    """Side-by-side bar plots: mean |SHAP| per feature for each series."""
    series_list = sorted(feature_df["_series"].unique())
    n = len(series_list)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, series_id in enumerate(series_list):
        ax = axes[i // ncols, i % ncols]
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values]
        mean_abs = np.mean(np.abs(series_shap), axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]

        names = [feature_names[j] for j in order]
        vals = mean_abs[order]
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(vals)))
        ax.barh(range(len(names))[::-1], vals, color=colors)
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{series_id}", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2, axis="x")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"Per-Series SHAP Importance — {model_name}",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_per_series_{model_name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 6: Cross-Model SHAP Comparison
# =====================================================================
def plot_cross_model_comparison(all_shap: dict, feature_names: list, out_dir, top_n=15):
    """Heatmap comparing mean |SHAP| across ALL models."""
    if len(all_shap) < 2:
        return

    records = {}
    for model_name, sv in all_shap.items():
        mean_abs = np.mean(np.abs(sv), axis=0)
        records[model_name] = {feature_names[i]: mean_abs[i] for i in range(len(feature_names))}

    df = pd.DataFrame(records)
    # Top features by average across models
    avg = df.mean(axis=1).sort_values(ascending=False)
    top_feats = avg.head(top_n).index.tolist()
    df = df.loc[top_feats]

    # Normalized heatmap
    normed = df.div(df.max(axis=0), axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(records) * 1.8), max(6, top_n * 0.4)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)

    # Annotate cells
    for i in range(len(top_feats)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normed.iloc[i, j] > 0.6 else "black")

    ax.set_title("Cross-Model SHAP Comparison\n(Mean |SHAP| — Normalized per model)",
                 fontweight="bold", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalized |SHAP|")
    plt.tight_layout()
    fig.savefig(out_dir / "cross_model_shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Agreement score
    std_across = normed.std(axis=1)
    agreement = 1 - std_across

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3)))
    colors = ["#2ca02c" if a > 0.7 else "#ff7f0e" if a > 0.4 else "#d62728"
              for a in agreement]
    ax.barh(agreement.index[::-1], agreement.values[::-1], color=colors[::-1])
    ax.axvline(0.7, color="green", ls="--", lw=1, alpha=0.5, label="High agreement")
    ax.axvline(0.4, color="orange", ls="--", lw=1, alpha=0.5, label="Medium agreement")
    ax.set_xlabel("Explanation Agreement (1 = all models agree)", fontsize=11)
    ax.set_title("Cross-Model Explanation Agreement (Rashomon-like)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / "cross_model_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 7: Surrogate Faithfulness Summary
# =====================================================================
def plot_faithfulness(faithfulness: dict, out_dir):
    """Bar chart of surrogate R² per model."""
    models = list(faithfulness.keys())
    r2_vals = [faithfulness[m]["R2"] for m in models]
    corr_vals = [faithfulness[m]["Correlation"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R²
    ax = axes[0]
    colors = ["#2ca02c" if v > 0.9 else "#ff7f0e" if v > 0.7 else "#d62728" for v in r2_vals]
    ax.barh(models, r2_vals, color=colors, edgecolor="white")
    ax.axvline(0.9, color="green", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("R²", fontsize=11)
    ax.set_title("Surrogate Faithfulness (R²)", fontweight="bold")
    for i, v in enumerate(r2_vals):
        ax.text(max(v + 0.01, 0.02), i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")

    # Correlation
    ax = axes[1]
    colors = ["#2ca02c" if v > 0.95 else "#ff7f0e" if v > 0.85 else "#d62728" for v in corr_vals]
    ax.barh(models, corr_vals, color=colors, edgecolor="white")
    ax.set_xlabel("Pearson Correlation", fontsize=11)
    ax.set_title("Surrogate Faithfulness (Correlation)", fontweight="bold")
    for i, v in enumerate(corr_vals):
        ax.text(max(v + 0.01, 0.02), i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("Surrogate Model Faithfulness\n(How well LightGBM mimics each AutoGluon model)",
                 fontweight="bold", fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(out_dir / "surrogate_faithfulness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 8: SHAP Waterfall (single prediction)
# =====================================================================
def plot_shap_waterfall(shap_vals, X, feature_names, model_name, out_dir,
                        sample_idx=0, top_n=15):
    row_shap = shap_vals[sample_idx]
    abs_vals = np.abs(row_shap)
    order = np.argsort(abs_vals)[::-1][:top_n]

    feats = [feature_names[i] for i in order]
    vals = [row_shap[i] for i in order]
    feat_vals_str = [f"{feature_names[i]} = {X.iloc[sample_idx, i]:.2f}" for i in order]

    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(range(len(feats)), vals, color=colors, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feat_vals_str, fontsize=9)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP Value (contribution to prediction)", fontsize=11)
    ax.set_title(f"SHAP Waterfall — Sample #{sample_idx} — {model_name}",
                 fontweight="bold", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_waterfall_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 9: Temporal SHAP Lines — actual time axis (per series)
# =====================================================================
def plot_temporal_shap_lines(shap_vals, feature_df, feature_names, model_name,
                             test_timestamps, out_dir, top_n=6):
    """
    Line plot: SHAP values over ACTUAL TIME for each series.
    Top panel: SHAP lines per feature, Bottom panel: total |SHAP| magnitude.
    """
    mean_abs_global = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_global)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]

    series_list = sorted(feature_df["_series"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    for series_id in series_list:
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values]
        n_steps = series_shap.shape[0]
        time_axis = test_timestamps[:n_steps]

        fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                                 gridspec_kw={"height_ratios": [2.5, 1]})

        # Top: individual feature SHAP lines
        ax = axes[0]
        for rank, (feat_idx, fname) in enumerate(zip(order, top_feat_names)):
            vals = series_shap[:, feat_idx]
            # Smooth
            if len(vals) > 30:
                w = max(1, len(vals) // 30)
                vals = pd.Series(vals).rolling(w, min_periods=1, center=True).mean().values
            ax.plot(time_axis, vals, lw=1.5, label=fname, color=colors[rank], alpha=0.85)

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("SHAP Value", fontsize=11)
        ax.set_title(f"Temporal SHAP — {series_id} — {model_name}\n"
                     f"(Feature contributions over time)",
                     fontweight="bold", fontsize=12)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=30)

        # Bottom: total absolute SHAP
        ax2 = axes[1]
        total_abs = np.sum(np.abs(series_shap[:, order]), axis=1)
        if len(total_abs) > 30:
            w = max(1, len(total_abs) // 30)
            total_abs = pd.Series(total_abs).rolling(w, min_periods=1, center=True).mean().values
        ax2.fill_between(time_axis, total_abs, alpha=0.4, color="steelblue")
        ax2.plot(time_axis, total_abs, lw=1.5, color="steelblue")
        ax2.set_ylabel("Total |SHAP|", fontsize=11)
        ax2.set_xlabel("Time", fontsize=11)
        ax2.set_title("Total Feature Contribution Magnitude", fontweight="bold", fontsize=10)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        fig.savefig(out_dir / f"temporal_lines_{model_name}_{series_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# Plot 10: Forecast + SHAP Combined (actual vs predicted + SHAP explanation)
# =====================================================================
def plot_forecast_with_shap(shap_vals, feature_df, feature_names, model_name,
                            full_data, predictions_dict, test_timestamps,
                            out_dir, top_n=5, history_steps=500):
    """
    Combined plot per series:
      Top panel: actual (train tail + test) vs model forecast
      Bottom panel: stacked area of SHAP contributions over the same time axis
    """
    mean_abs_global = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_global)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]

    series_list = sorted(feature_df["_series"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    pred_length = PREDICTION_LENGTH

    for series_id in series_list:
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values]
        n_steps = series_shap.shape[0]
        time_forecast = test_timestamps[:n_steps]

        # Actual data
        actual_full = full_data[series_id].values
        actual_idx = full_data.index

        n_hist = min(history_steps, len(actual_full) - pred_length)
        hist_time = actual_idx[-(n_hist + pred_length):-pred_length]
        hist_vals = actual_full[-(n_hist + pred_length):-pred_length]
        test_vals = actual_full[-pred_length:]

        # Model predictions
        pred_vals = predictions_dict.get(series_id, None)

        fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                                 gridspec_kw={"height_ratios": [2, 1.5, 1]})

        # ---- Panel 1: Forecast ----
        ax = axes[0]
        ax.plot(hist_time, hist_vals, color="steelblue", lw=1.2, label="Train (tail)")
        ax.plot(time_forecast[:len(test_vals)], test_vals[:len(time_forecast)],
                color="forestgreen", lw=2, label="Actual (test)")
        if pred_vals is not None:
            ax.plot(time_forecast[:len(pred_vals)], pred_vals[:len(time_forecast)],
                    color="crimson", lw=2, ls="--", label=f"Forecast ({model_name})")
        ax.axvspan(time_forecast[0], time_forecast[-1], alpha=0.05, color="forestgreen")
        ax.set_ylabel(series_id, fontsize=11)
        ax.set_title(f"{series_id} — Forecast + SHAP Explanation — {model_name}",
                     fontweight="bold", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)

        # ---- Panel 2: SHAP Lines ----
        ax2 = axes[1]
        for rank, (feat_idx, fname) in enumerate(zip(order, top_feat_names)):
            vals = series_shap[:, feat_idx]
            if len(vals) > 30:
                w = max(1, len(vals) // 30)
                vals = pd.Series(vals).rolling(w, min_periods=1, center=True).mean().values
            ax2.plot(time_forecast, vals, lw=1.5, label=fname, color=colors[rank], alpha=0.85)
        ax2.axhline(0, color="gray", lw=0.8, ls="--")
        ax2.set_ylabel("SHAP Value", fontsize=11)
        ax2.set_title("Feature SHAP Contributions Over Time", fontweight="bold", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right", ncol=2)
        ax2.grid(True, alpha=0.2)

        # ---- Panel 3: Stacked Area (absolute contributions) ----
        ax3 = axes[2]
        abs_shap = np.abs(series_shap[:, order])
        # Smooth
        smoothed = np.zeros_like(abs_shap)
        if abs_shap.shape[0] > 30:
            w = max(1, abs_shap.shape[0] // 30)
            for j in range(abs_shap.shape[1]):
                smoothed[:, j] = pd.Series(abs_shap[:, j]).rolling(
                    w, min_periods=1, center=True).mean().values
        else:
            smoothed = abs_shap

        ax3.stackplot(time_forecast, smoothed.T, labels=top_feat_names,
                      colors=colors[:top_n], alpha=0.7)
        ax3.set_ylabel("|SHAP| (stacked)", fontsize=11)
        ax3.set_xlabel("Time", fontsize=11)
        ax3.set_title("Stacked Feature Contributions (absolute)", fontweight="bold", fontsize=10)
        ax3.legend(fontsize=7, loc="upper right", ncol=2)
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        fig.savefig(out_dir / f"forecast_shap_{model_name}_{series_id}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# Plot 11: All-Series Temporal SHAP Summary (one compact overview)
# =====================================================================
def plot_all_series_temporal_summary(shap_vals, feature_df, feature_names, model_name,
                                     test_timestamps, out_dir, top_n=5):
    """
    Grid of subplots: one per series, each showing smoothed SHAP lines over time.
    """
    mean_abs_global = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_global)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]

    series_list = sorted(feature_df["_series"].unique())
    n = len(series_list)
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for i, series_id in enumerate(series_list):
        ax = axes[i // ncols, i % ncols]
        mask = feature_df["_series"] == series_id
        series_shap = shap_vals[mask.values]
        n_steps = series_shap.shape[0]
        time_axis = test_timestamps[:n_steps]

        for rank, (feat_idx, fname) in enumerate(zip(order, top_feat_names)):
            vals = series_shap[:, feat_idx]
            if len(vals) > 20:
                w = max(1, len(vals) // 20)
                vals = pd.Series(vals).rolling(w, min_periods=1, center=True).mean().values
            ax.plot(time_axis, vals, lw=1.2, label=fname if i == 0 else "",
                    color=colors[rank], alpha=0.8)

        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_title(f"{series_id}", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=top_n, fontsize=9,
              bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Temporal SHAP Lines — {model_name} — All Series\n"
                 f"(Feature SHAP values over actual time)",
                 fontweight="bold", fontsize=14, y=1.06)
    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_lines_all_{model_name}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Plot 12: SHAP over Forecast Horizon (how importance changes with step)
# =====================================================================
def plot_shap_vs_horizon(shap_vals, feature_df, feature_names, model_name, out_dir, top_n=6):
    """Line plot: mean |SHAP| per feature across forecast steps."""
    mean_abs_global = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs_global)[::-1][:top_n]
    top_feat_names = [feature_names[i] for i in order]

    steps = feature_df["_step_idx"].values
    unique_steps = np.sort(np.unique(steps))

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    for rank, (feat_idx, fname) in enumerate(zip(order, top_feat_names)):
        step_means = []
        for s in unique_steps:
            mask = steps == s
            step_means.append(np.mean(np.abs(shap_vals[mask, feat_idx])))
        step_means = np.array(step_means)

        # Smooth
        if len(step_means) > 20:
            window = max(1, len(step_means) // 30)
            step_means = pd.Series(step_means).rolling(window, min_periods=1, center=True).mean().values

        ax.plot(unique_steps + 1, step_means, lw=1.8, label=fname,
                color=colors[rank], alpha=0.85)

    ax.set_xlabel("Forecast Step (h)", fontsize=11)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Feature Importance vs Forecast Horizon — {model_name}\n"
                 f"(How feature importance changes as we predict further ahead)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_vs_horizon_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Surrogate SHAP for AutoGluon models")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to analyze (default: all)")
    parser.add_argument("--top_n", type=int, default=20, help="Top N features for plots")
    parser.add_argument("--temporal_window", type=int, default=288)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Surrogate SHAP Analysis for AutoGluon Time Series Models")
    print("=" * 70)

    # ----- 1. Load Data -----
    print("\n[1/5] Loading data...")
    train, val, test = load_telco_data()
    full_data = pd.concat([train, val, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  Series: {ts_cols}")
    print(f"  Total timesteps: {len(full_data)}")
    print(f"  Prediction length: {PREDICTION_LENGTH}")

    # ----- 2. Load Predictor -----
    print("\n[2/5] Loading AutoGluon predictor...")
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))
    leaderboard = predictor.leaderboard(silent=True)
    all_models = leaderboard["model"].tolist()
    print(f"  Available models: {all_models}")

    if args.models:
        selected_models = [m for m in args.models if m in all_models]
    else:
        selected_models = all_models
    print(f"  Selected for analysis: {selected_models}")

    # ----- 3. Build Features -----
    print("\n[3/5] Building tabular features...")
    feature_df = build_all_features(full_data, PREDICTION_LENGTH)

    # Encode series_id as numeric
    series_map = {s: i for i, s in enumerate(sorted(feature_df["series_id"].unique()))}
    feature_df["series_id_num"] = feature_df["series_id"].map(series_map)

    # Feature columns (exclude metadata)
    meta_cols = ["_series", "_step_idx", "series_id"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    # Replace series_id with numeric version
    feature_cols = [c for c in feature_cols if c != "series_id"]

    X_all = feature_df[feature_cols].copy()
    X_all = X_all.fillna(X_all.median())
    print(f"  Feature matrix: {X_all.shape}")
    print(f"  Features: {feature_cols}")

    # Test timestamps for temporal plots
    test_timestamps = full_data.index[-PREDICTION_LENGTH:]

    # ----- 4. Prepare TimeSeriesDataFrame for predictions -----
    train_tsdf = to_multi_series_tsdf(pd.concat([train, val]).pipe(
        lambda d: d[~d.index.duplicated(keep="first")].sort_index()
    ))

    # ----- 5. Run SHAP for each model -----
    print("\n[4/5] Computing surrogate SHAP for each model...")
    all_shap_values = {}
    all_faithfulness = {}

    for model_name in selected_models:
        print(f"\n  {'='*50}")
        print(f"  Model: {model_name}")
        print(f"  {'='*50}")

        # Get predictions
        try:
            preds = predictor.predict(train_tsdf, model=model_name)
        except Exception as e:
            print(f"    [ERROR] Prediction failed: {e}")
            continue

        # Extract predicted values for each series
        y_all = []
        predictions_dict = {}
        for series_id in ts_cols:
            try:
                item_preds = preds.loc[series_id]["mean"].values
                if len(item_preds) >= PREDICTION_LENGTH:
                    item_preds = item_preds[:PREDICTION_LENGTH]
                y_all.append(item_preds)
                predictions_dict[series_id] = item_preds
            except Exception:
                y_all.append(np.full(PREDICTION_LENGTH, np.nan))

        y_target = np.concatenate(y_all)
        valid_mask = ~np.isnan(y_target)

        if valid_mask.sum() < 100:
            print(f"    [WARN] Too few valid predictions ({valid_mask.sum()}), skipping")
            continue

        X_valid = X_all[valid_mask].reset_index(drop=True)
        y_valid = y_target[valid_mask]

        print(f"    Predictions: {len(y_valid)} valid samples")

        # Train surrogate
        print(f"    Training LightGBM surrogate...")
        surrogate = train_surrogate(X_valid, y_valid, model_name)

        # Evaluate faithfulness
        metrics = evaluate_surrogate(surrogate, X_valid, y_valid)
        all_faithfulness[model_name] = metrics
        print(f"    Surrogate R² = {metrics['R2']:.4f}, "
              f"Corr = {metrics['Correlation']:.4f}")

        if metrics["R2"] < 0.5:
            print(f"    [WARN] Low surrogate R² — SHAP values may not be faithful")

        # Compute SHAP
        print(f"    Computing TreeSHAP...")
        sv = compute_shap_values(surrogate, X_valid)
        all_shap_values[model_name] = sv

        # Generate per-model plots
        print(f"    Generating plots...")
        fn = feature_cols

        plot_shap_bar(sv, fn, model_name, RESULTS_DIR, top_n=args.top_n)
        print(f"      - SHAP bar")

        plot_shap_summary(sv, X_valid, fn, model_name, RESULTS_DIR, top_n=args.top_n)
        print(f"      - SHAP summary (beeswarm)")

        plot_shap_dependence(sv, X_valid, fn, model_name, RESULTS_DIR, top_n=6)
        print(f"      - SHAP dependence")

        plot_shap_waterfall(sv, X_valid, fn, model_name, RESULTS_DIR, sample_idx=0)
        print(f"      - SHAP waterfall")

        # Temporal heatmap (uses metadata from feature_df)
        valid_feature_df = feature_df[valid_mask].reset_index(drop=True)
        plot_temporal_shap_heatmap(sv, valid_feature_df, fn, model_name, RESULTS_DIR,
                                   top_n=min(15, args.top_n))
        print(f"      - Temporal SHAP heatmap")

        plot_per_series_shap(sv, valid_feature_df, fn, model_name, RESULTS_DIR, top_n=10)
        print(f"      - Per-series SHAP")

        plot_shap_vs_horizon(sv, valid_feature_df, fn, model_name, RESULTS_DIR, top_n=6)
        print(f"      - SHAP vs forecast horizon")

        # Temporal SHAP line plots (actual time axis)
        plot_temporal_shap_lines(sv, valid_feature_df, fn, model_name,
                                test_timestamps, RESULTS_DIR, top_n=6)
        print(f"      - Temporal SHAP lines (per series)")

        plot_all_series_temporal_summary(sv, valid_feature_df, fn, model_name,
                                         test_timestamps, RESULTS_DIR, top_n=5)
        print(f"      - All-series temporal summary")

        # Forecast + SHAP combined
        plot_forecast_with_shap(sv, valid_feature_df, fn, model_name,
                                full_data, predictions_dict, test_timestamps,
                                RESULTS_DIR, top_n=5)
        print(f"      - Forecast + SHAP combined (per series)")

        # Export SHAP values
        shap_df = pd.DataFrame(sv, columns=fn)
        shap_df["_series"] = valid_feature_df["_series"].values
        shap_df["_step"] = valid_feature_df["_step_idx"].values
        shap_df.to_csv(RESULTS_DIR / f"shap_values_{model_name}.csv", index=False)
        print(f"      - SHAP values exported")

    # ----- 6. Cross-Model Comparison -----
    print(f"\n[5/5] Cross-model comparison...")
    if len(all_shap_values) >= 2:
        plot_cross_model_comparison(all_shap_values, feature_cols, RESULTS_DIR,
                                    top_n=min(15, args.top_n))
        print("    Cross-model heatmap + agreement saved")
    else:
        print("    [INFO] Need >= 2 models for cross-model comparison")

    # Faithfulness summary
    if all_faithfulness:
        plot_faithfulness(all_faithfulness, RESULTS_DIR)
        print("    Faithfulness plot saved")

        faith_df = pd.DataFrame(all_faithfulness).T
        faith_df.to_csv(RESULTS_DIR / "surrogate_faithfulness.csv")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"\nOutput directory: {RESULTS_DIR}")
    print(f"\nSurrogate Faithfulness:")
    for m, f in all_faithfulness.items():
        print(f"  {m:25s}  R²={f['R2']:.4f}  Corr={f['Correlation']:.4f}")
    print(f"\nGenerated files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
