"""
Train DeepAR on TS1 only (single series).
Smaller data -> should avoid macOS ARM64 fork deadlock.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "TS1_data"
MODELS_DIR = Path(__file__).parent / "ts1_forecast"
RESULTS_DIR = Path(__file__).parent / "results" / "ts1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 288
FREQ = "5min"
MAX_POINTS = 5000  # Last N points only (avoid macOS deadlock)


def load_ts1_data():
    train = pd.read_csv(DATA_DIR / "TS1_data_train.csv", parse_dates=["time"], index_col="time")
    val = pd.read_csv(DATA_DIR / "TS1_data_val.csv", parse_dates=["time"], index_col="time")
    test = pd.read_csv(DATA_DIR / "TS1_data_test.csv", parse_dates=["time"], index_col="time")
    return train, val, test


def to_tsdf(df, item_id="TS1"):
    temp = df.copy().rename(columns={"TS1": "target"})
    temp["item_id"] = item_id
    temp = temp.reset_index().rename(columns={"time": "timestamp"})
    temp["timestamp"] = pd.to_datetime(temp["timestamp"]).dt.tz_localize(None)
    return TimeSeriesDataFrame.from_data_frame(
        temp, id_column="item_id", timestamp_column="timestamp"
    )


def compute_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mask = np.abs(actual) > 1e-10
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else np.nan
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def main():
    print("=" * 60, flush=True)
    print("TS1-Only DeepAR Training", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/4] Loading TS1 data...", flush=True)
    train, val, test = load_ts1_data()
    full_train = pd.concat([train, val])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    print(f"  train+val: {len(full_train)} rows, test: {len(test)} rows, total: {len(full_data)} rows", flush=True)

    if MAX_POINTS and len(full_data) > MAX_POINTS:
        full_data = full_data.iloc[-MAX_POINTS:]
        print(f"  Truncated to last {MAX_POINTS} points: {len(full_data)} rows", flush=True)

    print("\n[2/4] Building TSDF (single series)...", flush=True)
    tsdf = to_tsdf(full_data)
    train_tsdf = tsdf.slice_by_timestep(None, -PREDICTION_LENGTH)
    print(f"  Full shape: {tsdf.shape}, Train shape: {train_tsdf.shape}", flush=True)

    print("\n[3/4] Training DeepAR...", flush=True)
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
        time_limit=1800,  # 30 min
        num_val_windows=1,
        enable_ensemble=False,
        random_seed=42,
        hyperparameters={
            "DeepAR": {
                "max_epochs": 30,
                "num_batches_per_epoch": 50,
                "trainer_kwargs": {"accelerator": "cpu", "devices": 1},
            }
        },
    )
    elapsed = time.time() - start
    print(f"\n  Trained in {elapsed:.0f}s", flush=True)

    print("\n[4/4] Evaluation:", flush=True)
    lb = predictor.leaderboard(silent=True)
    print(lb.to_string(), flush=True)
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)

    preds = predictor.predict(train_tsdf)
    actual = full_data["TS1"].iloc[-PREDICTION_LENGTH:].values
    pred_mean = preds.loc["TS1"]["mean"].values[:PREDICTION_LENGTH]
    metrics = compute_metrics(actual, pred_mean)
    print(f"\n  Test metrics (last {PREDICTION_LENGTH} steps):", flush=True)
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}", flush=True)

    # Save forecast
    preds.to_csv(RESULTS_DIR / "deepar_forecast.csv")
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "deepar_metrics.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
