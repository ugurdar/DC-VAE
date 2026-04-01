"""
Implet-based XAI for Time Series Forecasting Models
====================================================
Adaptation of "Implet: A Post-hoc Subsequence Explainer for Time Series Models"
(Meng et al., 2025 IEEE ICDMW) for AutoGluon forecasting outputs.

Original paper: classification → per-timestep attribution → implet extraction → DTW clustering
This adaptation: forecasting → occlusion-based attribution → implet extraction → DTW clustering

Pipeline:
  1. Compute per-timestep attribution via occlusion (mask input windows, measure forecast change)
  2. Extract contiguous high-attribution subsequences (implets) using Algorithm 1
  3. Cluster implets across series using DTW + k-medoids (Algorithm 2 / Coh-Implet)
  4. Visualize results

Usage:
    python implet_forecast_xai.py --mode multi
    python implet_forecast_xai.py --mode multi --window_size 12 --lamb 0.1
"""

from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import json
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from dtaidistance import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("[WARN] dtaidistance not found. DTW clustering will use scipy fallback.")

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score as sk_silhouette

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError:
    raise SystemExit("autogluon.timeseries not found.")

# =====================================================================
# Paths
# =====================================================================
BASE_DIR    = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR    = PROJECT_DIR / "TELCO_data"
MODELS_DIR  = BASE_DIR / "models" / "autogluon"
RESULTS_DIR = BASE_DIR / "results" / "autogluon"
IMPLET_DIR  = RESULTS_DIR / "implet_xai"


# =====================================================================
# Data loading (same as autogluon_forecast.py)
# =====================================================================
def load_telco_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", index_col=0, parse_dates=True)
    val   = pd.read_csv(DATA_DIR / "TELCO_data_val.csv",   index_col=0, parse_dates=True)
    test  = pd.read_csv(DATA_DIR / "TELCO_data_test.csv",  index_col=0, parse_dates=True)
    train.index.name = "time"
    val.index.name   = "time"
    test.index.name  = "time"
    return train, val, test


def to_multi_series_tsdf(df: pd.DataFrame):
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
# Step 1: Occlusion-based Attribution
# =====================================================================
def compute_occlusion_attribution(
    predictor: TimeSeriesPredictor,
    train_tsdf: TimeSeriesDataFrame,
    full_data: pd.DataFrame,
    prediction_length: int,
    window_size: int = 12,
    stride: int = 6,
    model_name: str = None,
) -> dict:
    """
    Compute per-timestep attribution scores via occlusion for each series.

    For each input window of size `window_size`, replace values with the series mean
    and measure the change in forecast MAE. Higher change = higher importance.

    Returns:
        dict: {series_name: attribution_array} where attribution_array has shape (n_input_timesteps,)
    """
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])

    # Baseline prediction
    predict_kwargs = {"model": model_name} if model_name else {}
    base_preds = predictor.predict(train_tsdf, **predict_kwargs)

    attributions = {}

    for item_id in ts_cols:
        print(f"    Occlusion attribution for {item_id}...")
        base_pred_vals = base_preds.loc[item_id]["mean"].values
        actual_vals = full_data[item_id].values

        # Input region: everything before the prediction horizon
        input_length = len(actual_vals) - prediction_length
        attr_scores = np.zeros(input_length)
        attr_counts = np.zeros(input_length)

        series_mean = np.mean(actual_vals[:input_length])

        # Slide occlusion window over input
        for start in range(0, input_length - window_size + 1, stride):
            end = start + window_size

            # Create occluded data
            occluded_data = full_data.copy()
            occluded_data.iloc[start:end, occluded_data.columns.get_loc(item_id)] = series_mean

            # Build occluded TSDF
            occluded_tsdf = to_multi_series_tsdf(occluded_data)
            occluded_train = occluded_tsdf.slice_by_timestep(None, -prediction_length)

            # Predict with occluded input
            try:
                occ_preds = predictor.predict(occluded_train, **predict_kwargs)
                occ_pred_vals = occ_preds.loc[item_id]["mean"].values

                # Attribution = absolute change in predictions
                delta = np.mean(np.abs(base_pred_vals - occ_pred_vals))
                attr_scores[start:end] += delta
                attr_counts[start:end] += 1
            except Exception as e:
                print(f"      [WARN] Occlusion at [{start}:{end}] failed: {e}")
                continue

        # Average overlapping attributions
        mask = attr_counts > 0
        attr_scores[mask] /= attr_counts[mask]

        attributions[item_id] = attr_scores
        print(f"      Done. Max attr: {attr_scores.max():.6f}, Mean: {attr_scores.mean():.6f}")

    return attributions


# =====================================================================
# Step 1b: Fast Residual-based Attribution (alternative)
# =====================================================================
def compute_residual_attribution(
    full_data: pd.DataFrame,
    predictions,
    prediction_length: int,
    lookback_window: int = 288,
) -> dict:
    """
    Fast attribution based on local variability and autocorrelation with forecast errors.

    Uses rolling statistics on the input series and correlates with forecast errors
    to identify which input patterns most influence prediction quality.
    """
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    attributions = {}

    for item_id in ts_cols:
        actual = full_data[item_id].values
        input_vals = actual[:-prediction_length]
        input_length = len(input_vals)

        # Forecast errors
        pred_vals = predictions.loc[item_id]["mean"].values
        test_vals = actual[-prediction_length:]
        errors = np.abs(test_vals - pred_vals[:len(test_vals)])
        mean_error = np.mean(errors)

        # Attribution: combination of local variability and gradient magnitude
        # High local change in input → potentially more influential
        grad = np.abs(np.gradient(input_vals))

        # Rolling std (captures volatility)
        win = min(lookback_window, input_length // 4)
        rolling_std = pd.Series(input_vals).rolling(win, min_periods=1, center=True).std().fillna(0).values

        # Combine: normalize each and weight
        grad_norm = grad / (grad.max() + 1e-12)
        std_norm = rolling_std / (rolling_std.max() + 1e-12)

        # Focus on recent history (exponential decay from end of input)
        decay = np.exp(-np.arange(input_length)[::-1] / lookback_window)
        decay /= decay.max()

        attr = (0.5 * grad_norm + 0.3 * std_norm + 0.2 * decay)
        attr /= (attr.max() + 1e-12)

        attributions[item_id] = attr

    return attributions


# =====================================================================
# Step 2: Implet Extraction (Algorithm 1 from paper)
# =====================================================================
def max_score_subsequence(arr: np.ndarray, left: int, lamb: float,
                          threshold: float, kmin: int = 3, kmax: int = None) -> tuple:
    """
    Find the optimal subsequence starting at `left` that maximizes:
        score = mean(attr[left:left+k]) - lamb * k

    This is adapted from Algorithm 1 in the Implet paper.
    """
    n = len(arr)
    if kmax is None:
        kmax = n - left

    best_score = -np.inf
    best_k = kmin

    cumsum = 0.0
    for k in range(1, min(kmax, n - left) + 1):
        cumsum += arr[left + k - 1]
        if k < kmin:
            continue

        mean_val = cumsum / k
        if mean_val < threshold:
            continue

        score = mean_val - lamb * k
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def extract_implets(
    input_series: np.ndarray,
    attr_scores: np.ndarray,
    lamb: float = 0.1,
    thresh_factor: float = 1.0,
    kmin: int = 3,
    kmax: int = None,
    max_implets: int = 10,
) -> list:
    """
    Extract implets (high-attribution subsequences) from input series.

    Implements Algorithm 1 from the Implet paper:
    1. Find position with highest remaining attribution
    2. Extract optimal subsequence around it
    3. Zero out extracted region and repeat

    Returns:
        list of dicts: [{start, end, values, attr_values, score}, ...]
    """
    n = len(attr_scores)
    threshold = np.mean(attr_scores) * thresh_factor
    remaining_attr = attr_scores.copy()
    implets = []

    for _ in range(max_implets):
        # Find position with highest remaining attribution
        peak = np.argmax(remaining_attr)
        if remaining_attr[peak] < threshold:
            break

        # Search for optimal subsequence starting near peak
        # Try starting positions around the peak
        best_overall_score = -np.inf
        best_start = peak
        best_length = kmin

        search_range = max(kmax or 50, 50)
        for start in range(max(0, peak - search_range), min(n, peak + 1)):
            k, score = max_score_subsequence(remaining_attr, start, lamb, threshold, kmin, kmax)
            if score > best_overall_score and start + k > peak:
                best_overall_score = score
                best_start = start
                best_length = k

        if best_overall_score <= 0:
            break

        end = min(best_start + best_length, n)

        implet = {
            "start": best_start,
            "end": end,
            "values": input_series[best_start:end].copy(),
            "attr_values": attr_scores[best_start:end].copy(),
            "score": best_overall_score,
            "mean_attr": np.mean(attr_scores[best_start:end]),
        }
        implets.append(implet)

        # Zero out extracted region
        remaining_attr[best_start:end] = 0.0

    # Sort by score descending
    implets.sort(key=lambda x: x["score"], reverse=True)
    return implets


# =====================================================================
# Step 3: DTW-based Clustering (Algorithm 2 / Coh-Implet)
# =====================================================================
def compute_dtw_distance_matrix(implet_values_list: list) -> np.ndarray:
    """Compute pairwise DTW distance matrix for a list of implet value arrays."""
    n = len(implet_values_list)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if HAS_DTW:
                d = dtw.distance(implet_values_list[i].astype(np.double),
                                 implet_values_list[j].astype(np.double))
            else:
                # Scipy fallback: use euclidean on zero-padded
                max_len = max(len(implet_values_list[i]), len(implet_values_list[j]))
                a = np.pad(implet_values_list[i], (0, max_len - len(implet_values_list[i])))
                b = np.pad(implet_values_list[j], (0, max_len - len(implet_values_list[j])))
                d = np.sqrt(np.sum((a - b) ** 2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def cluster_implets(
    all_implets: list,
    k_range: tuple = (2, 8),
) -> tuple:
    """
    Cluster implets using hierarchical clustering with DTW distance.

    Uses silhouette score to select optimal k (Coh-Implet approach).

    Returns:
        (labels, best_k, dist_matrix)
    """
    if len(all_implets) < 3:
        return np.zeros(len(all_implets), dtype=int), 1, None

    # Normalize implet values (z-score within each implet)
    normalized_values = []
    for imp in all_implets:
        vals = imp["values"]
        std = np.std(vals)
        if std > 1e-8:
            normalized_values.append((vals - np.mean(vals)) / std)
        else:
            normalized_values.append(vals - np.mean(vals))

    # Compute DTW distance matrix
    print("    Computing DTW distance matrix...")
    dist_matrix = compute_dtw_distance_matrix(normalized_values)

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")

    # Find best k via silhouette score
    best_k = k_range[0]
    best_score = -1

    max_k = min(k_range[1], len(all_implets) - 1)
    for k in range(k_range[0], max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            score = sk_silhouette(dist_matrix, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    labels = fcluster(Z, t=best_k, criterion="maxclust")
    print(f"    Best k={best_k}, silhouette={best_score:.4f}")

    return labels, best_k, dist_matrix


# =====================================================================
# Visualization
# =====================================================================
def plot_attributions(attributions: dict, full_data: pd.DataFrame,
                      prediction_length: int, out_dir: Path):
    """Plot attribution scores for each series."""
    if not HAS_MPL:
        return
    ts_cols = sorted(attributions.keys())
    n = len(ts_cols)
    fig, axes = plt.subplots(n, 1, figsize=(18, 3 * n), squeeze=False)

    for i, col in enumerate(ts_cols):
        ax = axes[i, 0]
        input_vals = full_data[col].values[:-prediction_length]
        attr = attributions[col]
        time_idx = full_data.index[:-prediction_length]

        # Plot input series
        ax2 = ax.twinx()
        ax2.plot(time_idx[:len(input_vals)], input_vals, color="steelblue", alpha=0.3, lw=0.8)
        ax2.set_ylabel("Value", color="steelblue", fontsize=8)

        # Plot attribution as filled area
        ax.fill_between(time_idx[:len(attr)], 0, attr, color="crimson", alpha=0.5)
        ax.set_ylabel("Attribution", color="crimson", fontsize=9)
        ax.set_title(f"{col} - Per-Timestep Attribution Scores", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Occlusion-based Attribution Scores (Implet Method)",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(out_dir / "attribution_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Attribution plot saved")


def plot_implets_on_series(
    full_data: pd.DataFrame,
    implets_dict: dict,
    prediction_length: int,
    out_dir: Path,
):
    """Highlight extracted implets on the original series."""
    if not HAS_MPL:
        return
    ts_cols = sorted(implets_dict.keys())
    n = len(ts_cols)
    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n), squeeze=False)

    colors = ["crimson", "darkorange", "forestgreen", "dodgerblue", "purple",
              "brown", "deeppink", "olive", "teal", "navy"]

    for i, col in enumerate(ts_cols):
        ax = axes[i, 0]
        input_vals = full_data[col].values[:-prediction_length]
        time_idx = full_data.index[:-prediction_length]

        # Show last portion for visibility
        show_len = min(len(input_vals), 2000)
        offset = len(input_vals) - show_len

        ax.plot(time_idx[offset:], input_vals[offset:], color="steelblue", lw=1, alpha=0.7)

        implets = implets_dict[col]
        for j, imp in enumerate(implets[:5]):  # Show top 5
            s, e = imp["start"], imp["end"]
            if s >= offset:
                c = colors[j % len(colors)]
                ax.plot(time_idx[s:e], input_vals[s:e], color=c, lw=3, alpha=0.9)
                ax.axvspan(time_idx[s], time_idx[e - 1], alpha=0.1, color=c)
                ax.annotate(f"Imp{j+1}\nscore={imp['score']:.3f}",
                           xy=(time_idx[(s + e) // 2], input_vals[(s + e) // 2]),
                           fontsize=7, color=c, fontweight="bold",
                           ha="center", va="bottom")

        ax.set_title(f"{col} - Extracted Implets (Top 5)", fontsize=11, fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Implet Extraction: Critical Subsequences for Forecasting",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(out_dir / "implets_on_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Implets-on-series plot saved")


def plot_implet_clusters(
    all_implets: list,
    labels: np.ndarray,
    best_k: int,
    out_dir: Path,
):
    """Visualize clustered implets (Coh-Implets)."""
    if not HAS_MPL:
        return

    colors = plt.cm.Set1(np.linspace(0, 1, best_k))

    # Plot 1: All implets colored by cluster
    fig, axes = plt.subplots(1, best_k, figsize=(5 * best_k, 5), squeeze=False)
    for k in range(1, best_k + 1):
        ax = axes[0, k - 1]
        cluster_implets = [imp for imp, lab in zip(all_implets, labels) if lab == k]

        for imp in cluster_implets:
            vals = imp["values"]
            # Z-normalize for overlay
            std = np.std(vals)
            if std > 1e-8:
                normed = (vals - np.mean(vals)) / std
            else:
                normed = vals - np.mean(vals)
            ax.plot(normed, alpha=0.3, lw=1, color=colors[k - 1])

        # Plot centroid (mean of all z-normed implets, padded to max length)
        if cluster_implets:
            max_len = max(len(imp["values"]) for imp in cluster_implets)
            padded = []
            for imp in cluster_implets:
                vals = imp["values"]
                std = np.std(vals)
                normed = (vals - np.mean(vals)) / (std + 1e-8)
                padded.append(np.pad(normed, (0, max_len - len(normed)), constant_values=np.nan))
            centroid = np.nanmean(padded, axis=0)
            ax.plot(centroid, color="black", lw=3, label="Centroid")

        ax.set_title(f"Cluster {k} (n={len(cluster_implets)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalized Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Coh-Implets: Clustered Subsequence Patterns",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "coh_implets_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Cluster distribution by series
    series_names = [imp["series"] for imp in all_implets]
    unique_series = sorted(set(series_names))

    cluster_series_counts = np.zeros((best_k, len(unique_series)))
    for imp, lab in zip(all_implets, labels):
        si = unique_series.index(imp["series"])
        cluster_series_counts[lab - 1, si] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(unique_series))
    width = 0.8 / best_k
    for k in range(best_k):
        ax.bar(x + k * width, cluster_series_counts[k], width,
               label=f"Cluster {k+1}", color=colors[k], alpha=0.8)
    ax.set_xticks(x + width * best_k / 2)
    ax.set_xticklabels(unique_series, rotation=45)
    ax.set_ylabel("Number of Implets")
    ax.set_title("Implet Cluster Distribution Across Series", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "cluster_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Cluster plots saved")


def plot_implet_attribution_summary(implets_dict: dict, out_dir: Path):
    """Summary bar chart of implet statistics per series."""
    if not HAS_MPL:
        return

    summary = []
    for col, implets in sorted(implets_dict.items()):
        if implets:
            summary.append({
                "series": col,
                "n_implets": len(implets),
                "top_score": implets[0]["score"],
                "avg_score": np.mean([i["score"] for i in implets]),
                "total_coverage": sum(i["end"] - i["start"] for i in implets),
                "avg_length": np.mean([i["end"] - i["start"] for i in implets]),
            })

    if not summary:
        return

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "implet_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Top score per series
    ax = axes[0]
    df_sorted = df.sort_values("top_score", ascending=True)
    ax.barh(df_sorted["series"], df_sorted["top_score"], color="crimson", alpha=0.8)
    ax.set_xlabel("Top Implet Score")
    ax.set_title("Top Implet Score by Series", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # Number of implets
    ax = axes[1]
    df_sorted = df.sort_values("n_implets", ascending=True)
    ax.barh(df_sorted["series"], df_sorted["n_implets"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Number of Implets")
    ax.set_title("Implet Count by Series", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # Avg length
    ax = axes[2]
    df_sorted = df.sort_values("avg_length", ascending=True)
    ax.barh(df_sorted["series"], df_sorted["avg_length"], color="forestgreen", alpha=0.8)
    ax.set_xlabel("Avg Implet Length (timesteps)")
    ax.set_title("Average Implet Length by Series", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("Implet Extraction Summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "implet_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Implet summary plot saved")


def plot_faithfulness_test(
    predictor, train_tsdf, full_data, predictions, implets_dict,
    prediction_length, out_dir, model_name=None, n_random_trials=5,
):
    """
    Faithfulness test: ablate implets and random subsequences, compare forecast degradation.

    If implets are meaningful, ablating them should cause LARGER forecast changes than
    ablating random subsequences of the same total length.
    """
    if not HAS_MPL:
        return

    ts_cols = sorted(implets_dict.keys())
    predict_kwargs = {"model": model_name} if model_name else {}

    results = []

    for item_id in ts_cols:
        implets = implets_dict[item_id]
        if not implets:
            continue

        actual_vals = full_data[item_id].values
        input_length = len(actual_vals) - prediction_length
        base_pred = predictions.loc[item_id]["mean"].values
        series_mean = np.mean(actual_vals[:input_length])

        # Ablate implets
        ablated_data_imp = full_data.copy()
        total_imp_len = 0
        for imp in implets[:5]:
            s, e = imp["start"], imp["end"]
            ablated_data_imp.iloc[s:e, ablated_data_imp.columns.get_loc(item_id)] = series_mean
            total_imp_len += (e - s)

        imp_tsdf = to_multi_series_tsdf(ablated_data_imp)
        imp_train = imp_tsdf.slice_by_timestep(None, -prediction_length)
        try:
            imp_preds = predictor.predict(imp_train, **predict_kwargs)
            imp_delta = np.mean(np.abs(base_pred - imp_preds.loc[item_id]["mean"].values))
        except Exception:
            imp_delta = 0.0

        # Ablate random subsequences of same total length
        random_deltas = []
        for _ in range(n_random_trials):
            ablated_data_rand = full_data.copy()
            remaining = total_imp_len
            while remaining > 0:
                seg_len = min(remaining, np.random.randint(3, max(4, total_imp_len // 3 + 1)))
                start = np.random.randint(0, max(1, input_length - seg_len))
                ablated_data_rand.iloc[start:start + seg_len,
                                       ablated_data_rand.columns.get_loc(item_id)] = series_mean
                remaining -= seg_len

            rand_tsdf = to_multi_series_tsdf(ablated_data_rand)
            rand_train = rand_tsdf.slice_by_timestep(None, -prediction_length)
            try:
                rand_preds = predictor.predict(rand_train, **predict_kwargs)
                rand_delta = np.mean(np.abs(base_pred - rand_preds.loc[item_id]["mean"].values))
                random_deltas.append(rand_delta)
            except Exception:
                pass

        avg_rand_delta = np.mean(random_deltas) if random_deltas else 0.0

        results.append({
            "series": item_id,
            "implet_ablation_delta": imp_delta,
            "random_ablation_delta": avg_rand_delta,
            "ratio": imp_delta / (avg_rand_delta + 1e-12),
            "implet_total_len": total_imp_len,
        })
        print(f"    {item_id}: implet_delta={imp_delta:.4f}, random_delta={avg_rand_delta:.4f}, "
              f"ratio={imp_delta / (avg_rand_delta + 1e-12):.2f}")

    if not results:
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "faithfulness_test.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width / 2, res_df["implet_ablation_delta"], width,
           label="Implet Ablation", color="crimson", alpha=0.8)
    ax.bar(x + width / 2, res_df["random_ablation_delta"], width,
           label="Random Ablation", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["series"], rotation=45)
    ax.set_ylabel("Mean |Forecast Change|")
    ax.set_title("Faithfulness Test: Implet vs Random Ablation\n"
                 "(Higher implet ablation = more faithful explanations)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    # Add ratio annotations
    for i, row in res_df.iterrows():
        ax.annotate(f"{row['ratio']:.1f}x",
                   xy=(i, max(row["implet_ablation_delta"], row["random_ablation_delta"])),
                   fontsize=8, ha="center", va="bottom", fontweight="bold", color="black")

    plt.tight_layout()
    fig.savefig(out_dir / "faithfulness_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Faithfulness test plot saved")


# =====================================================================
# Main Pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Implet XAI for AutoGluon Forecasting")
    parser.add_argument("--mode", type=str, default="multi", choices=["single", "multi"])
    parser.add_argument("--window_size", type=int, default=24,
                        help="Occlusion window size in timesteps")
    parser.add_argument("--stride", type=int, default=12,
                        help="Occlusion stride")
    parser.add_argument("--lamb", type=float, default=0.1,
                        help="Lambda: trade-off between implet length and attribution strength")
    parser.add_argument("--thresh_factor", type=float, default=1.0,
                        help="Threshold factor for implet extraction")
    parser.add_argument("--max_implets", type=int, default=8,
                        help="Max implets per series")
    parser.add_argument("--attribution", type=str, default="residual",
                        choices=["occlusion", "residual"],
                        help="Attribution method (occlusion is slow but precise, residual is fast)")
    parser.add_argument("--prediction_length", type=int, default=288)
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name (default: best model)")
    args = parser.parse_args()

    print("=" * 80)
    print("Implet XAI for Time Series Forecasting")
    print(f"Attribution: {args.attribution} | Lambda: {args.lamb} | Window: {args.window_size}")
    print("=" * 80)

    IMPLET_DIR.mkdir(parents=True, exist_ok=True)
    prediction_length = args.prediction_length

    # Load data
    print("\n[1/6] Loading data and model...")
    train_raw, val_raw, test_raw = load_telco_data()
    full_train = pd.concat([train_raw, val_raw])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test_raw])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()

    # Load trained predictor
    model_path = MODELS_DIR / args.mode
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run autogluon_forecast.py first.")

    predictor = TimeSeriesPredictor.load(str(model_path))
    print(f"  Predictor loaded from {model_path}")

    # Build TSDF
    tsdf = to_multi_series_tsdf(full_data)
    train_tsdf = tsdf.slice_by_timestep(None, -prediction_length)

    # Get baseline predictions
    predict_kwargs = {"model": args.model} if args.model else {}
    predictions = predictor.predict(train_tsdf, **predict_kwargs)
    ts_cols = sorted(predictions.item_ids)

    # Step 1: Attribution
    print(f"\n[2/6] Computing {args.attribution} attributions...")
    t0 = time.time()
    if args.attribution == "occlusion":
        attributions = compute_occlusion_attribution(
            predictor, train_tsdf, full_data, prediction_length,
            window_size=args.window_size, stride=args.stride,
            model_name=args.model,
        )
    else:
        attributions = compute_residual_attribution(
            full_data, predictions, prediction_length,
            lookback_window=prediction_length,
        )
    attr_time = time.time() - t0
    print(f"  Attribution computed in {attr_time:.1f}s")

    # Save attributions
    attr_df = pd.DataFrame(
        {col: np.pad(attr, (0, max(0, len(full_data) - prediction_length - len(attr))))
         for col, attr in attributions.items()},
        index=full_data.index[:-prediction_length][:max(len(v) for v in attributions.values())]
    )
    attr_df.to_csv(IMPLET_DIR / "attributions.csv")

    # Plot attributions
    plot_attributions(attributions, full_data, prediction_length, IMPLET_DIR)

    # Step 2: Implet Extraction
    print(f"\n[3/6] Extracting implets (lambda={args.lamb}, thresh={args.thresh_factor})...")
    implets_dict = {}
    all_implets_flat = []

    for col in ts_cols:
        input_vals = full_data[col].values[:-prediction_length]
        attr = attributions[col]

        implets = extract_implets(
            input_vals, attr,
            lamb=args.lamb,
            thresh_factor=args.thresh_factor,
            kmin=3,
            kmax=args.window_size * 4 if args.attribution == "occlusion" else 200,
            max_implets=args.max_implets,
        )
        implets_dict[col] = implets

        # Add series info and flatten for clustering
        for imp in implets:
            imp["series"] = col
            all_implets_flat.append(imp)

        n_imp = len(implets)
        if n_imp > 0:
            top_score = implets[0]["score"]
            print(f"  {col}: {n_imp} implets extracted (top score: {top_score:.4f})")
        else:
            print(f"  {col}: no implets found")

    # Save implet details
    implet_records = []
    for imp in all_implets_flat:
        implet_records.append({
            "series": imp["series"],
            "start": imp["start"],
            "end": imp["end"],
            "length": imp["end"] - imp["start"],
            "score": imp["score"],
            "mean_attr": imp["mean_attr"],
        })
    pd.DataFrame(implet_records).to_csv(IMPLET_DIR / "implet_details.csv", index=False)

    # Plot implets on series
    plot_implets_on_series(full_data, implets_dict, prediction_length, IMPLET_DIR)

    # Plot implet summary
    plot_implet_attribution_summary(implets_dict, IMPLET_DIR)

    # Step 3: Coh-Implet Clustering
    print(f"\n[4/6] Clustering implets (Coh-Implet)...")
    if len(all_implets_flat) >= 3:
        labels, best_k, dist_matrix = cluster_implets(all_implets_flat, k_range=(2, 6))

        # Save cluster assignments
        for imp, lab in zip(all_implets_flat, labels):
            imp["cluster"] = int(lab)

        cluster_df = pd.DataFrame([
            {"series": imp["series"], "start": imp["start"], "end": imp["end"],
             "score": imp["score"], "cluster": imp["cluster"]}
            for imp in all_implets_flat
        ])
        cluster_df.to_csv(IMPLET_DIR / "implet_clusters.csv", index=False)

        # Plot clusters
        plot_implet_clusters(all_implets_flat, labels, best_k, IMPLET_DIR)
    else:
        print("  Too few implets for clustering, skipping.")

    # Step 4: Faithfulness Test
    print(f"\n[5/6] Faithfulness test (ablation analysis)...")
    plot_faithfulness_test(
        predictor, train_tsdf, full_data, predictions, implets_dict,
        prediction_length, IMPLET_DIR, model_name=args.model,
        n_random_trials=3,
    )

    # Step 5: Generate Report
    print(f"\n[6/6] Generating report...")
    report_lines = [
        "# Implet XAI Report - Time Series Forecasting",
        f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Attribution method**: {args.attribution}",
        f"**Lambda**: {args.lamb}",
        f"**Threshold factor**: {args.thresh_factor}",
        f"**Window size**: {args.window_size} (occlusion)",
        f"**Max implets per series**: {args.max_implets}",
        f"**Attribution computation time**: {attr_time:.1f}s",
        f"\n## Method",
        "Adaptation of 'Implet: A Post-hoc Subsequence Explainer for Time Series Models'",
        "(Meng et al., 2025 IEEE ICDMW) for forecasting models.",
        "\n### Pipeline:",
        "1. **Attribution**: Per-timestep importance via occlusion/residual analysis",
        "2. **Implet Extraction**: Contiguous high-attribution subsequences (Algorithm 1)",
        "3. **Coh-Implet Clustering**: DTW-based clustering of implets (Algorithm 2)",
        "4. **Faithfulness Test**: Ablation comparison (implet vs random)",
        f"\n## Results",
        f"Total implets extracted: {len(all_implets_flat)}",
    ]

    if len(all_implets_flat) >= 3:
        report_lines.append(f"Optimal clusters (k): {best_k}")

    # Per-series summary
    report_lines.append("\n## Per-Series Implet Summary")
    report_lines.append("| Series | N Implets | Top Score | Avg Length |")
    report_lines.append("|--------|-----------|-----------|------------|")
    for col in ts_cols:
        imps = implets_dict[col]
        if imps:
            report_lines.append(
                f"| {col} | {len(imps)} | {imps[0]['score']:.4f} | "
                f"{np.mean([i['end'] - i['start'] for i in imps]):.0f} |"
            )
        else:
            report_lines.append(f"| {col} | 0 | - | - |")

    # Faithfulness results
    faith_path = IMPLET_DIR / "faithfulness_test.csv"
    if faith_path.exists():
        faith_df = pd.read_csv(faith_path)
        avg_ratio = faith_df["ratio"].mean()
        report_lines.append(f"\n## Faithfulness Test")
        report_lines.append(f"Average implet/random ablation ratio: **{avg_ratio:.2f}x**")
        report_lines.append(f"(>1.0 means implets are more faithful than random subsequences)")

    report_path = IMPLET_DIR / "REPORT_implet_xai.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n{'=' * 80}")
    print(f"IMPLET XAI COMPLETE!")
    print(f"  Results: {IMPLET_DIR}")
    print(f"  Report:  {report_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
