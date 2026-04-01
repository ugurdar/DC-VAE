"""
SHAP & Temporal-SHAP Analysis for H2O AutoML Models
=====================================================
Generates comprehensive SHAP visualizations from saved H2O models:
  1. SHAP Summary (beeswarm / bar)
  2. SHAP Dependence plots (top features)
  3. SHAP Waterfall (single prediction)
  4. Temporal SHAP Heatmap (feature importance over time — TimeSHAP-like)
  5. Rashomon SHAP comparison (multiple models)

Usage:
    python shap_analysis.py                         # default TS1
    python shap_analysis.py --target TS3
    python shap_analysis.py --target TS1 --max_models 5
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import h2o

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
RESULTS_DIR = BASE_DIR / "results" / "h2o"
MODELS_DIR  = BASE_DIR / "models" / "h2o"

# Import feature engineering from h2o_forecast
from h2o_forecast import (
    load_telco_data, build_features, CONTROL,
    get_varimp_df,
)

# =====================================================================
# Helpers
# =====================================================================

def get_shap_contributions(model, data_h2o, max_rows=2000):
    """Get SHAP contributions from H2O model as pandas DataFrame."""
    n = data_h2o.nrows
    sample = data_h2o[0:min(n, max_rows), :]
    try:
        contribs = model.predict_contributions(sample).as_data_frame()
    except Exception:
        bg = data_h2o[0:500, :]
        contribs = model.predict_contributions(sample, background_frame=bg).as_data_frame()
    if "BiasTerm" in contribs.columns:
        contribs = contribs.drop(columns=["BiasTerm"])
    return contribs


def get_top_features(contribs, top_n=15):
    """Get top N features by mean absolute SHAP value."""
    mean_abs = contribs.abs().mean().sort_values(ascending=False)
    return mean_abs.head(top_n).index.tolist()


# =====================================================================
# Plot 1: SHAP Summary (Beeswarm-like)
# =====================================================================
def plot_shap_summary(contribs, feature_values, target, model_id, out_dir, top_n=20):
    """Beeswarm-style plot: each dot is a sample, x=SHAP, color=feature value."""
    top_feats = get_top_features(contribs, top_n)
    top_feats_available = [f for f in top_feats if f in feature_values.columns]

    if not top_feats_available:
        print(f"  [WARN] No matching features for summary plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_feats_available) * 0.4)))

    for i, feat in enumerate(reversed(top_feats_available)):
        shap_vals = contribs[feat].values
        feat_vals = feature_values[feat].values[:len(shap_vals)]

        # Normalize feature values for color
        fmin, fmax = np.nanmin(feat_vals), np.nanmax(feat_vals)
        if fmax - fmin > 1e-10:
            normed = (feat_vals - fmin) / (fmax - fmin)
        else:
            normed = np.zeros_like(feat_vals)

        # Jitter y
        y_jitter = i + np.random.uniform(-0.3, 0.3, size=len(shap_vals))

        # Subsample for readability
        n_show = min(500, len(shap_vals))
        idx = np.random.choice(len(shap_vals), n_show, replace=False)

        scatter = ax.scatter(
            shap_vals[idx], y_jitter[idx],
            c=normed[idx], cmap="coolwarm", s=8, alpha=0.6,
            edgecolors="none", vmin=0, vmax=1,
        )

    ax.set_yticks(range(len(top_feats_available)))
    ax.set_yticklabels(list(reversed(top_feats_available)), fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("SHAP Value", fontsize=11)
    ax.set_title(f"SHAP Summary — {target} ({model_id.split('_')[0]})",
                 fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.15, axis="x")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Feature Value (normalized)", fontsize=9)

    plt.tight_layout()
    safe = model_id.replace("/", "_")
    fig.savefig(out_dir / f"shap_summary_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    SHAP summary saved ({model_id.split('_')[0]})")


# =====================================================================
# Plot 2: SHAP Bar (mean |SHAP|)
# =====================================================================
def plot_shap_bar(contribs, target, model_id, out_dir, top_n=20):
    mean_abs = contribs.abs().mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3)))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(mean_abs)))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance — {target} ({model_id.split('_')[0]})",
                 fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    safe = model_id.replace("/", "_")
    fig.savefig(out_dir / f"shap_bar_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    SHAP bar saved ({model_id.split('_')[0]})")


# =====================================================================
# Plot 3: SHAP Dependence (top features)
# =====================================================================
def plot_shap_dependence(contribs, feature_values, target, model_id, out_dir, top_n=6):
    """Scatter: feature value vs SHAP value for top features."""
    top_feats = get_top_features(contribs, top_n)
    top_feats = [f for f in top_feats if f in feature_values.columns]

    if not top_feats:
        return

    ncols = 3
    nrows = (len(top_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, feat in enumerate(top_feats):
        ax = axes[i // ncols, i % ncols]
        x = feature_values[feat].values[:len(contribs)]
        y = contribs[feat].values

        n_show = min(1000, len(x))
        idx = np.random.choice(len(x), n_show, replace=False)

        ax.scatter(x[idx], y[idx], s=6, alpha=0.4, color="steelblue", edgecolors="none")
        ax.axhline(0, color="crimson", lw=0.8, ls="--")
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("SHAP Value", fontsize=9)
        ax.set_title(feat, fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2)

    for j in range(len(top_feats), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"SHAP Dependence — {target} ({model_id.split('_')[0]})",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    safe = model_id.replace("/", "_")
    fig.savefig(out_dir / f"shap_dependence_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    SHAP dependence saved ({model_id.split('_')[0]})")


# =====================================================================
# Plot 4: SHAP Waterfall (single prediction)
# =====================================================================
def plot_shap_waterfall(contribs, feature_values, target, model_id, out_dir,
                        sample_idx=0, top_n=15):
    """Waterfall chart for a single prediction."""
    row = contribs.iloc[sample_idx]
    abs_vals = row.abs().sort_values(ascending=False)
    top = abs_vals.head(top_n)
    feats = top.index.tolist()
    vals = [row[f] for f in feats]

    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(range(len(feats)), vals, color=colors, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f"{f} = {feature_values[f].iloc[sample_idx]:.2f}"
                        if f in feature_values.columns else f
                        for f in feats], fontsize=9)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP Value (contribution to prediction)", fontsize=11)
    ax.set_title(f"SHAP Waterfall — Sample #{sample_idx} — {target} ({model_id.split('_')[0]})",
                 fontweight="bold", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    safe = model_id.replace("/", "_")
    fig.savefig(out_dir / f"shap_waterfall_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    SHAP waterfall saved (sample #{sample_idx})")


# =====================================================================
# Plot 5: Temporal SHAP Heatmap (TimeSHAP-like)
# =====================================================================
def plot_temporal_shap_heatmap(contribs, time_index, target, model_id, out_dir,
                                top_n=15, window=None):
    """
    Heatmap: features (y) x time (x) with SHAP values as color.
    This is the TimeSHAP-like visualization showing how feature
    contributions evolve over time.
    """
    top_feats = get_top_features(contribs, top_n)
    shap_matrix = contribs[top_feats].values.T  # (features, time)

    # Optionally limit time window
    n_time = shap_matrix.shape[1]
    if window and n_time > window:
        shap_matrix = shap_matrix[:, :window]
        time_index = time_index[:window]
        n_time = window

    fig, ax = plt.subplots(figsize=(min(20, max(12, n_time * 0.015)), max(5, top_n * 0.4)))

    vmax = np.percentile(np.abs(shap_matrix), 97)
    im = ax.imshow(shap_matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)

    # X-axis: show time labels at intervals
    n_labels = min(12, n_time)
    step = max(1, n_time // n_labels)
    tick_positions = list(range(0, n_time, step))
    if hasattr(time_index, "strftime"):
        tick_labels = [time_index[i].strftime("%m-%d %H:%M") for i in tick_positions if i < len(time_index)]
    else:
        tick_labels = [str(i) for i in tick_positions]
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(f"Temporal SHAP Heatmap — {target} ({model_id.split('_')[0]})\n"
                 f"Red = pushes prediction UP, Blue = pushes prediction DOWN",
                 fontweight="bold", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("SHAP Value", fontsize=10)

    plt.tight_layout()
    safe = model_id.replace("/", "_")
    fig.savefig(out_dir / f"temporal_shap_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Temporal SHAP heatmap saved ({model_id.split('_')[0]})")


# =====================================================================
# Plot 6: Temporal SHAP Line (top features over time)
# =====================================================================
def plot_temporal_shap_lines(contribs, time_index, target, model_id, out_dir,
                              top_n=6, window=None):
    """Line plot of SHAP values over time for top features + total."""
    top_feats = get_top_features(contribs, top_n)
    n = len(contribs)
    if window and n > window:
        contribs = contribs.iloc[:window]
        time_index = time_index[:window]
        n = window

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top: individual features
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_feats)))
    for feat, color in zip(top_feats, colors):
        # Smooth for readability
        vals = contribs[feat].rolling(window=max(1, n // 200), min_periods=1, center=True).mean()
        axes[0].plot(range(len(vals)), vals, lw=1.2, label=feat, color=color, alpha=0.8)

    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("SHAP Value", fontsize=11)
    axes[0].set_title(f"Feature SHAP Over Time — {target} ({model_id.split('_')[0]})",
                      fontweight="bold", fontsize=12)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].grid(True, alpha=0.2)

    # Bottom: total absolute SHAP (model confidence proxy)
    total_abs = contribs[top_feats].abs().sum(axis=1)
    total_smooth = total_abs.rolling(window=max(1, n // 200), min_periods=1, center=True).mean()
    axes[1].fill_between(range(len(total_smooth)), total_smooth, alpha=0.4, color="steelblue")
    axes[1].plot(range(len(total_smooth)), total_smooth, lw=1.5, color="steelblue")
    axes[1].set_ylabel("Total |SHAP|", fontsize=11)
    axes[1].set_xlabel("Time Index", fontsize=11)
    axes[1].set_title("Total Feature Contribution Magnitude", fontweight="bold", fontsize=11)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    safe = model_id.replace("/", "_")
    fig.savefig(out_dir / f"temporal_shap_lines_{target}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Temporal SHAP lines saved ({model_id.split('_')[0]})")


# =====================================================================
# Plot 7: Rashomon SHAP Comparison
# =====================================================================
def plot_rashomon_shap(all_contribs, target, out_dir, top_n=15):
    """Compare mean |SHAP| across multiple Rashomon set models."""
    if len(all_contribs) < 2:
        return

    # Collect mean |SHAP| per model
    records = {}
    for model_id, contribs in all_contribs.items():
        mean_abs = contribs.abs().mean()
        short = model_id.split("_")[0] + "_" + model_id.split("_")[1] if "_" in model_id else model_id[:15]
        records[short] = mean_abs

    df = pd.DataFrame(records).fillna(0)

    # Top features by average across models
    avg = df.mean(axis=1).sort_values(ascending=False)
    top_feats = avg.head(top_n).index.tolist()
    df = df.loc[top_feats]

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(records) * 1.5), max(6, top_n * 0.4)))
    normed = df.div(df.max(axis=0), axis=1)
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_title(f"Rashomon SHAP Comparison — {target}\n"
                 f"(Normalized mean |SHAP| across near-optimal models)",
                 fontweight="bold", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalized |SHAP|")
    plt.tight_layout()
    fig.savefig(out_dir / f"rashomon_shap_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Agreement score per feature
    std_across = normed.std(axis=1)
    agreement = 1 - std_across
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3)))
    colors = ["#2ca02c" if a > 0.7 else "#ff7f0e" if a > 0.4 else "#d62728" for a in agreement]
    ax.barh(agreement.index[::-1], agreement.values[::-1], color=colors[::-1])
    ax.axvline(0.7, color="green", ls="--", lw=1, alpha=0.5, label="High agreement")
    ax.axvline(0.4, color="orange", ls="--", lw=1, alpha=0.5, label="Medium agreement")
    ax.set_xlabel("Explanation Agreement (1 = all models agree)", fontsize=11)
    ax.set_title(f"Rashomon Explanation Agreement — {target}", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / f"rashomon_agreement_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Rashomon SHAP comparison saved ({len(all_contribs)} models)")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="SHAP & Temporal-SHAP Analysis")
    parser.add_argument("--target", type=str, default="TS1")
    parser.add_argument("--max_models", type=int, default=5,
                        help="Max models from leaderboard to analyze")
    parser.add_argument("--max_rows", type=int, default=2000,
                        help="Max test rows for SHAP computation")
    parser.add_argument("--temporal_window", type=int, default=500,
                        help="Time window for temporal SHAP plots")
    args = parser.parse_args()

    target = args.target
    ctrl = CONTROL.copy()
    ctrl["target"] = target

    # Output directory
    out_dir = RESULTS_DIR / "shap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SHAP & Temporal-SHAP Analysis — {target}")
    print("=" * 70)

    # ----- Load Data -----
    print("\n[1/4] Loading data & features...")
    train_raw, val_raw, test_raw = load_telco_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()

    df_combined = pd.concat([full_train, test_raw])
    df_combined = df_combined[~df_combined.index.duplicated(keep="first")].sort_index()
    df_combined_feat = build_features(df_combined.copy(), target, ctrl)

    max_lag = max(ctrl["target_lags"] + ctrl["rolling_windows"])
    test_start = test_raw.index[0]
    df_test_feat = df_combined_feat.loc[test_start:].dropna(subset=[target])

    exclude = {target, "time"}
    feature_cols = [c for c in df_test_feat.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if df_test_feat[c].notna().sum() > 0]
    df_test_feat[feature_cols] = df_test_feat[feature_cols].ffill().bfill().fillna(0)

    print(f"  Features: {len(feature_cols)}, Test rows: {len(df_test_feat)}")

    # ----- H2O Init -----
    print("\n[2/4] Starting H2O & loading models...")
    h2o.init(nthreads=-1, max_mem_size="4G")

    # Load leaderboard to find models
    lb_path = RESULTS_DIR / f"leaderboard_{target}.csv"
    if not lb_path.exists():
        print(f"  [ERROR] No leaderboard found at {lb_path}")
        print(f"  Run h2o_forecast.py --target {target} first.")
        h2o.cluster().shutdown(prompt=False)
        return

    lb = pd.read_csv(lb_path)
    print(f"  Leaderboard: {len(lb)} models")

    # Load top-k models
    test_h2o = h2o.H2OFrame(df_test_feat[[target] + feature_cols].reset_index(drop=True))
    models = {}
    for _, row in lb.head(args.max_models).iterrows():
        mid = row["model_id"]
        try:
            # Try loading from saved models directory
            model_files = list(MODELS_DIR.glob(f"{mid}*"))
            if model_files:
                model = h2o.load_model(str(model_files[0]))
            else:
                model = h2o.get_model(mid)
            models[mid] = model
            print(f"    Loaded: {mid}")
        except Exception as e:
            print(f"    [WARN] Could not load {mid}: {e}")

    if not models:
        print("  [ERROR] No models loaded. Run h2o_forecast.py first.")
        h2o.cluster().shutdown(prompt=False)
        return

    # ----- SHAP Analysis -----
    print(f"\n[3/4] Computing SHAP contributions...")
    all_contribs = {}
    leader_id = list(models.keys())[0]
    leader_model = models[leader_id]

    for mid, model in models.items():
        print(f"\n  --- {mid} ---")
        try:
            contribs = get_shap_contributions(model, test_h2o, max_rows=args.max_rows)
            all_contribs[mid] = contribs

            # Feature values (aligned with SHAP rows)
            feat_vals = df_test_feat[feature_cols].iloc[:len(contribs)].reset_index(drop=True)
            time_idx = df_test_feat.index[:len(contribs)]

            # Generate plots for this model
            plot_shap_bar(contribs, target, mid, out_dir)
            plot_shap_summary(contribs, feat_vals, target, mid, out_dir)
            plot_shap_dependence(contribs, feat_vals, target, mid, out_dir)

            # Waterfall for first sample
            plot_shap_waterfall(contribs, feat_vals, target, mid, out_dir, sample_idx=0)

            # Temporal SHAP (TimeSHAP-like)
            plot_temporal_shap_heatmap(contribs, time_idx, target, mid, out_dir,
                                       window=args.temporal_window)
            plot_temporal_shap_lines(contribs, time_idx, target, mid, out_dir,
                                     window=args.temporal_window)

            # Save raw SHAP values
            contribs.to_csv(out_dir / f"shap_values_{target}_{mid.split('_')[0]}.csv", index=False)

        except Exception as e:
            print(f"    [ERROR] SHAP failed for {mid}: {e}")

    # ----- Rashomon SHAP Comparison -----
    print(f"\n[4/4] Rashomon SHAP comparison...")
    if len(all_contribs) >= 2:
        plot_rashomon_shap(all_contribs, target, out_dir)
    else:
        print("  [INFO] Need >= 2 models for Rashomon comparison")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"\nOutput directory: {out_dir}")
    print(f"\nGenerated files:")
    for f in sorted(out_dir.glob(f"*{target}*")):
        print(f"  - {f.name}")
    print("=" * 70)

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
