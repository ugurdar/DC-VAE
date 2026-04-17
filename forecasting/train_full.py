"""
Train AutoGluon TimeSeries: DL + Statistical + Tree models.
Uses AutoGluon's default configs (hyperparameters=None) with excluded_model_types
to skip Chronos (user doesn't want) and TemporalFusionTransformer (macOS ARM64 deadlock).
"""
from pathlib import Path
import time
import warnings
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "TELCO_data"
MODELS_DIR = Path(__file__).parent / "models" / "multi"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 288
FREQ = "5min"
# Per-series history length (last N points used for training).
# DL context is ~288, so ~5000 gives DL plenty of history without deadlocking.
MAX_POINTS_PER_SERIES = 5000

# Models to EXCLUDE (Chronos user-rejected, TFT deadlocks on macOS ARM64)
EXCLUDED = [
    "Chronos2",
    "Chronos2SmallFineTuned",
    "ChronosWithRegressor",
    "Chronos",
    "ChronosBolt",
    "Toto",
    "TemporalFusionTransformer",  # deadlocks
]


def load_data():
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val = pd.read_csv(DATA_DIR / "TELCO_data_val.csv", parse_dates=["time"], index_col="time")
    test = pd.read_csv(DATA_DIR / "TELCO_data_test.csv", parse_dates=["time"], index_col="time")
    return train, val, test


def to_tsdf(df):
    ts_cols = sorted([c for c in df.columns if c.startswith("TS")])
    records = []
    for col in ts_cols:
        temp = df[[col]].copy().rename(columns={col: "target"})
        temp["item_id"] = col
        temp = temp.reset_index().rename(columns={"time": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        records.append(temp)
    long_df = pd.concat(records, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(
        long_df, id_column="item_id", timestamp_column="timestamp"
    )


def main():
    print("=" * 60, flush=True)
    print("AutoGluon Training: DL + Statistical + Tree", flush=True)
    print("(Chronos, Toto, TFT excluded)", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/3] Loading data...", flush=True)
    train, val, test = load_data()
    full_train = pd.concat([train, val])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  {len(full_data)} rows, {len(ts_cols)} series", flush=True)

    # Truncate: keep last MAX_POINTS_PER_SERIES per series
    if MAX_POINTS_PER_SERIES and len(full_data) > MAX_POINTS_PER_SERIES:
        full_data = full_data.iloc[-MAX_POINTS_PER_SERIES:]
        print(f"  Truncated to last {MAX_POINTS_PER_SERIES} points per series: {len(full_data)} rows", flush=True)

    print("\n[2/3] Building TSDF...", flush=True)
    tsdf = to_tsdf(full_data)
    train_tsdf = tsdf.slice_by_timestep(None, -PREDICTION_LENGTH)
    print(f"  Shape: {tsdf.shape} (per-series: {MAX_POINTS_PER_SERIES})", flush=True)

    print(f"\n[3/3] Training (excluded: {EXCLUDED})...", flush=True)

    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=PREDICTION_LENGTH,
        freq=FREQ,
        eval_metric="MASE",
        path=str(MODELS_DIR),
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        verbosity=2,
    )

    start = time.time()
    predictor.fit(
        train_tsdf,
        time_limit=7200,  # 2 hours
        presets="medium_quality",
        num_val_windows=2,
        enable_ensemble=True,
        random_seed=42,
        hyperparameters=None,  # Let AutoGluon choose defaults
        excluded_model_types=EXCLUDED,
    )
    elapsed = time.time() - start
    print(f"\nTraining complete: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    print("\nLeaderboard:", flush=True)
    lb = predictor.leaderboard(tsdf, silent=True)
    print(lb[["model", "score_val", "score_test", "fit_time_marginal"]].to_string(), flush=True)
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)

    print(f"\nModels: {MODELS_DIR}", flush=True)
    print(f"Results: {RESULTS_DIR}", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
