"""
Temporal Variable Importance — AutoGluon Time Series XAI
=========================================================
Produces a (variable x time) importance heatmap by masking each variable
at each time window and measuring the forecast degradation.

Output:
  - temporal_varimp_heatmap.png   : 2D heatmap (variable x time)
  - temporal_varimp_lines.png     : Line plot per variable over time
  - temporal_varimp_stacked.png   : Stacked area (relative contribution)
  - temporal_varimp_matrix.csv    : Raw importance matrix
  - top_variable_per_window.csv   : Most important variable per time window

Usage:
    python temporal_variable_xai.py
    python temporal_variable_xai.py --window 24 --stride 12
    python temporal_variable_xai.py --model_path ../automl/models/autogluon/multi
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import time as timer

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
MODELS_DIR  = BASE_DIR / "models" / "autogluon"
RESULTS_DIR = BASE_DIR / "results" / "autogluon" / "temporal_xai"


# =====================================================================
# Data loading
# =====================================================================
def load_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  parse_dates=["time"], index_col="time")
    return train, val, test


def to_tsdf(df: pd.DataFrame, ts_cols: list[str]) -> TimeSeriesDataFrame:
    """Convert wide DataFrame to AutoGluon TimeSeriesDataFrame (long format)."""
    records = []
    for col in ts_cols:
        chunk = df[[col]].copy()
        chunk.columns = ["target"]
        chunk["item_id"] = col
        chunk["timestamp"] = chunk.index
        records.append(chunk)
    long = pd.concat(records, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(long, id_column="item_id",
                                                timestamp_column="timestamp")


# =====================================================================
# Core: Temporal Variable Importance via Occlusion
# =====================================================================
def compute_temporal_variable_importance(
    predictor: TimeSeriesPredictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ts_cols: list[str],
    window_size: int = 24,
    stride: int = 12,
    prediction_length: int = 288,
    target_series: str = "TS1",
    n_repeats: int = 3,
):
    """
    For each variable, at each time window in the training tail,
    perturb that variable and measure how much the forecast changes.

    Returns: DataFrame (n_windows x n_variables) with importance scores.
    """
    # Baseline: unperturbed forecast
    full_train = train_df.copy()
    tsdf_base = to_tsdf(full_train, ts_cols)
    pred_base = predictor.predict(tsdf_base)

    # Extract baseline forecast for target series
    base_forecast = pred_base.loc[target_series]["mean"].values[:prediction_length]

    # Time windows in the tail of training data (most recent matters most)
    tail_length = min(len(full_train), prediction_length * 2)
    tail_start = len(full_train) - tail_length
    n_windows = (tail_length - window_size) // stride + 1

    print(f"  Tail length: {tail_length}, Windows: {n_windows}, Variables: {len(ts_cols)}")

    importance_matrix = np.zeros((n_windows, len(ts_cols)))
    window_centers = []

    for w_idx in range(n_windows):
        w_start = tail_start + w_idx * stride
        w_end = w_start + window_size
        center_time = full_train.index[min(w_start + window_size // 2, len(full_train) - 1)]
        window_centers.append(center_time)

        for v_idx, var in enumerate(ts_cols):
            deltas = []
            for rep in range(n_repeats):
                perturbed = full_train.copy()

                # Perturbation: shuffle values within the window for this variable
                original_vals = perturbed[var].iloc[w_start:w_end].values.copy()
                np.random.shuffle(original_vals)
                perturbed.iloc[w_start:w_end, perturbed.columns.get_loc(var)] = original_vals

                tsdf_pert = to_tsdf(perturbed, ts_cols)
                try:
                    pred_pert = predictor.predict(tsdf_pert)
                    pert_forecast = pred_pert.loc[target_series]["mean"].values[:prediction_length]
                    delta = np.mean((base_forecast - pert_forecast) ** 2)
                    deltas.append(delta)
                except Exception:
                    deltas.append(0.0)

            importance_matrix[w_idx, v_idx] = np.mean(deltas)

        pct = (w_idx + 1) / n_windows * 100
        if (w_idx + 1) % max(1, n_windows // 10) == 0 or w_idx == n_windows - 1:
            print(f"    Window {w_idx+1}/{n_windows} ({pct:.0f}%)")

    imp_df = pd.DataFrame(importance_matrix, columns=ts_cols,
                           index=pd.DatetimeIndex(window_centers))
    return imp_df


# =====================================================================
# Plot 1: Heatmap (variable x time)
# =====================================================================
def plot_heatmap(imp_df, target, out_dir):
    fig, ax = plt.subplots(figsize=(min(20, max(12, len(imp_df) * 0.06)),
                                     max(4, len(imp_df.columns) * 0.5)))

    data = imp_df.values.T  # (variables x time)
    vmax = np.percentile(data[data > 0], 95) if (data > 0).any() else 1.0

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="bilinear",
                   vmin=0, vmax=vmax)

    ax.set_yticks(range(len(imp_df.columns)))
    ax.set_yticklabels(imp_df.columns, fontsize=10)

    n_labels = min(15, len(imp_df))
    step = max(1, len(imp_df) // n_labels)
    tick_pos = list(range(0, len(imp_df), step))
    tick_labels = [imp_df.index[i].strftime("%m-%d %H:%M") for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Time Window Center", fontsize=11)
    ax.set_ylabel("Variable", fontsize=11)
    ax.set_title(f"Temporal Variable Importance — Forecast Target: {target}\n"
                 f"(Brighter = more important for prediction at that time)",
                 fontweight="bold", fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Importance (MSE change on forecast)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_varimp_heatmap_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Heatmap saved")


# =====================================================================
# Plot 2: Line plot per variable
# =====================================================================
def plot_lines(imp_df, target, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    colors = plt.cm.tab20(np.linspace(0, 1, len(imp_df.columns)))

    # Top: individual lines (smoothed)
    for i, col in enumerate(imp_df.columns):
        vals = imp_df[col].rolling(window=max(1, len(imp_df) // 50),
                                    min_periods=1, center=True).mean()
        axes[0].plot(range(len(vals)), vals, lw=1.5, label=col,
                     color=colors[i], alpha=0.85)

    axes[0].set_ylabel("Importance", fontsize=11)
    axes[0].set_title(f"Variable Importance Over Time — {target}", fontweight="bold", fontsize=13)
    axes[0].legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.8)
    axes[0].grid(True, alpha=0.2)

    # Bottom: dominant variable at each time
    dominant = imp_df.idxmax(axis=1)
    dom_encoded = dominant.map({col: i for i, col in enumerate(imp_df.columns)})
    dom_colors = [colors[int(v)] for v in dom_encoded.values]
    axes[1].scatter(range(len(dominant)), dom_encoded.values, c=dom_colors, s=15, alpha=0.8)
    axes[1].set_yticks(range(len(imp_df.columns)))
    axes[1].set_yticklabels(imp_df.columns, fontsize=8)
    axes[1].set_xlabel("Time Window Index", fontsize=11)
    axes[1].set_ylabel("Dominant Variable", fontsize=10)
    axes[1].set_title("Most Important Variable per Time Window", fontweight="bold", fontsize=11)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_varimp_lines_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Lines saved")


# =====================================================================
# Plot 3: Stacked area (relative contribution)
# =====================================================================
def plot_stacked(imp_df, target, out_dir):
    # Normalize to relative percentages
    row_sums = imp_df.sum(axis=1).replace(0, 1)
    pct_df = imp_df.div(row_sums, axis=0) * 100

    # Smooth
    w = max(1, len(pct_df) // 50)
    pct_smooth = pct_df.rolling(window=w, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(pct_smooth.columns)))

    ax.stackplot(range(len(pct_smooth)), *[pct_smooth[c].values for c in pct_smooth.columns],
                 labels=pct_smooth.columns, colors=colors, alpha=0.8)

    ax.set_ylabel("Relative Importance (%)", fontsize=11)
    ax.set_xlabel("Time Window Index", fontsize=11)
    ax.set_title(f"Relative Variable Contribution Over Time — {target}",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_varimp_stacked_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Stacked area saved")


# =====================================================================
# Plot 4: Summary bar (time-averaged importance per variable)
# =====================================================================
def plot_summary_bar(imp_df, target, out_dir):
    mean_imp = imp_df.mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(mean_imp) * 0.4)))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(mean_imp)))
    ax.barh(mean_imp.index, mean_imp.values, color=colors)
    ax.set_xlabel("Mean Importance (averaged over time)", fontsize=11)
    ax.set_title(f"Overall Variable Importance — {target}", fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    fig.savefig(out_dir / f"temporal_varimp_summary_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Summary bar saved")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Temporal Variable Importance — AutoGluon XAI")
    parser.add_argument("--model_path", type=str,
                        default=str(MODELS_DIR / "multi"))
    parser.add_argument("--target_series", type=str, default="TS1",
                        help="Which series' forecast to explain")
    parser.add_argument("--window", type=int, default=24,
                        help="Occlusion window size (timesteps)")
    parser.add_argument("--stride", type=int, default=12,
                        help="Stride between windows")
    parser.add_argument("--prediction_length", type=int, default=288)
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="Repeats per perturbation (reduces noise)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Temporal Variable Importance — Target: {args.target_series}")
    print("=" * 70)

    # ----- Load Data -----
    print("\n[1/4] Loading data...")
    train_raw, val_raw, test_raw = load_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    ts_cols = [c for c in full_train.columns if c.startswith("TS")]
    print(f"  Train: {full_train.shape}, Variables: {ts_cols}")

    # ----- Load Model -----
    print("\n[2/4] Loading AutoGluon predictor...")
    predictor = TimeSeriesPredictor.load(args.model_path)
    print(f"  Model loaded from {args.model_path}")

    # ----- Compute Temporal Importance -----
    print(f"\n[3/4] Computing temporal variable importance...")
    print(f"  Window: {args.window}, Stride: {args.stride}, Repeats: {args.n_repeats}")
    t0 = timer.time()

    imp_df = compute_temporal_variable_importance(
        predictor=predictor,
        train_df=full_train,
        test_df=test_raw,
        ts_cols=ts_cols,
        window_size=args.window,
        stride=args.stride,
        prediction_length=args.prediction_length,
        target_series=args.target_series,
        n_repeats=args.n_repeats,
    )

    elapsed = timer.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Save raw matrix
    imp_df.to_csv(RESULTS_DIR / f"temporal_varimp_matrix_{args.target_series}.csv")

    # Top variable per window
    top_per_window = imp_df.idxmax(axis=1).to_frame("top_variable")
    top_per_window["importance"] = imp_df.max(axis=1)
    top_per_window.to_csv(RESULTS_DIR / f"top_variable_per_window_{args.target_series}.csv")

    # ----- Plots -----
    print(f"\n[4/4] Generating plots...")
    plot_heatmap(imp_df, args.target_series, RESULTS_DIR)
    plot_lines(imp_df, args.target_series, RESULTS_DIR)
    plot_stacked(imp_df, args.target_series, RESULTS_DIR)
    plot_summary_bar(imp_df, args.target_series, RESULTS_DIR)

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"\nOutput: {RESULTS_DIR}")
    print(f"\nFiles:")
    for f in sorted(RESULTS_DIR.glob(f"*{args.target_series}*")):
        print(f"  - {f.name}")

    # Quick insight
    mean_imp = imp_df.mean().sort_values(ascending=False)
    print(f"\nTop 5 variables (time-averaged):")
    for i, (var, val) in enumerate(mean_imp.head(5).items()):
        print(f"  {i+1}. {var}: {val:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
