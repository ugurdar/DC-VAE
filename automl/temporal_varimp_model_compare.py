"""
DirectTabular vs RecursiveTabular — Temporal Variable Importance Comparison
===========================================================================
Her hedef seri (TS1, TS2, ...) için her iki modelin değişkenlere verdiği
önemi zamana bağlı olarak aynı line graph üzerinde karşılaştırır.

Usage:
    python temporal_varimp_model_compare.py
    python temporal_varimp_model_compare.py --target TS1
    python temporal_varimp_model_compare.py --target all --window 24 --stride 12
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

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
MODELS_DIR  = BASE_DIR / "models" / "autogluon" / "multi"
OUT_DIR     = BASE_DIR / "results" / "autogluon" / "model_compare_varimp"

MODELS = ["DirectTabular", "RecursiveTabular"]


# ── Data helpers ─────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   parse_dates=["time"], index_col="time")
    return pd.concat([train, val]).pipe(lambda d: d[~d.index.duplicated(keep="first")].sort_index())


def to_tsdf(df: pd.DataFrame, ts_cols: list[str]) -> TimeSeriesDataFrame:
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


# ── Temporal importance (occlusion-based) ────────────────────────────
def compute_importance(predictor, model_name, train_df, ts_cols,
                       target, window_size, stride, pred_len, n_repeats):
    """Compute (n_windows x n_variables) importance matrix for one model."""
    tsdf_base = to_tsdf(train_df, ts_cols)
    pred_base = predictor.predict(tsdf_base, model=model_name)
    base_fc   = pred_base.loc[target]["mean"].values[:pred_len]

    tail_len  = min(len(train_df), pred_len * 2)
    tail_start = len(train_df) - tail_len
    n_windows = (tail_len - window_size) // stride + 1

    imp = np.zeros((n_windows, len(ts_cols)))
    centers = []

    for w in range(n_windows):
        ws = tail_start + w * stride
        we = ws + window_size
        centers.append(train_df.index[min(ws + window_size // 2, len(train_df) - 1)])

        for v, var in enumerate(ts_cols):
            deltas = []
            for _ in range(n_repeats):
                pert = train_df.copy()
                vals = pert[var].iloc[ws:we].values.copy()
                np.random.shuffle(vals)
                pert.iloc[ws:we, pert.columns.get_loc(var)] = vals
                try:
                    pf = predictor.predict(to_tsdf(pert, ts_cols), model=model_name)
                    deltas.append(np.mean((base_fc - pf.loc[target]["mean"].values[:pred_len]) ** 2))
                except Exception:
                    deltas.append(0.0)
            imp[w, v] = np.mean(deltas)

        if (w + 1) % max(1, n_windows // 5) == 0 or w == n_windows - 1:
            print(f"      [{model_name}] Window {w+1}/{n_windows}")

    return pd.DataFrame(imp, columns=ts_cols, index=pd.DatetimeIndex(centers))


# ── Plot: two models on same graph, per variable ─────────────────────
def plot_comparison(imp_direct, imp_recursive, target, out_dir):
    """Her değişken için DirectTabular vs RecursiveTabular line graph."""
    variables = imp_direct.columns.tolist()
    n_vars = len(variables)

    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = axes.flatten()

    smooth_w = max(1, len(imp_direct) // 30)

    for i, var in enumerate(variables):
        ax = axes[i]
        d_vals = imp_direct[var].rolling(smooth_w, min_periods=1, center=True).mean()
        r_vals = imp_recursive[var].rolling(smooth_w, min_periods=1, center=True).mean()

        ax.plot(range(len(d_vals)), d_vals, lw=2, label="DirectTabular", color="#2196F3")
        ax.plot(range(len(r_vals)), r_vals, lw=2, label="RecursiveTabular", color="#F44336",
                linestyle="--")
        ax.fill_between(range(len(d_vals)), d_vals, alpha=0.10, color="#2196F3")
        ax.fill_between(range(len(r_vals)), r_vals, alpha=0.10, color="#F44336")

        ax.set_title(f"{var}", fontweight="bold", fontsize=11)
        ax.set_ylabel("Importance")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=9)

    # Hide unused axes
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Temporal Variable Importance — Target: {target}\n"
                 f"DirectTabular vs RecursiveTabular",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.supxlabel("Time Window Index", fontsize=12)
    plt.tight_layout()
    path = out_dir / f"varimp_compare_{target}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="TS1", help="Target series or 'all'")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=288)
    parser.add_argument("--n_repeats", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data & model...")
    train_df = load_data()
    ts_cols = [c for c in train_df.columns if c.startswith("TS")]
    predictor = TimeSeriesPredictor.load(str(MODELS_DIR))

    targets = ts_cols if args.target == "all" else [args.target]

    for target in targets:
        print(f"\n{'='*60}")
        print(f"  Target: {target}")
        print(f"{'='*60}")

        results = {}
        for model_name in MODELS:
            print(f"    Computing importance for {model_name}...")
            t0 = timer.time()
            results[model_name] = compute_importance(
                predictor, model_name, train_df, ts_cols,
                target, args.window, args.stride, args.pred_len, args.n_repeats,
            )
            print(f"    {model_name} done in {timer.time()-t0:.1f}s")

        # Save CSVs
        for m, df in results.items():
            df.to_csv(OUT_DIR / f"varimp_{m}_{target}.csv")

        # Plot
        plot_comparison(results["DirectTabular"], results["RecursiveTabular"], target, OUT_DIR)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
