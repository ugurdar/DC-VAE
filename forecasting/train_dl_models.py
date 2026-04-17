"""
Train Deep Learning forecasting models with AutoGluon.
Workaround for MPS/threading deadlock on macOS ARM64.
Forces CPU, single-threaded dataloader, explicit max_epochs.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "TELCO_data"
MODELS_DIR = Path(__file__).parent / "models" / "dl_models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 288
FREQ = "5min"

# DL models with explicit CPU settings
DL_HYPERPARAMETERS = {
    "TemporalFusionTransformer": {
        "max_epochs": 50,
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
        },
        "num_batches_per_epoch": 100,
    },
    "DeepAR": {
        "max_epochs": 50,
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
        },
        "num_batches_per_epoch": 100,
    },
    "PatchTST": {
        "max_epochs": 50,
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
        },
        "num_batches_per_epoch": 100,
    },
    "SimpleFeedForward": {
        "max_epochs": 50,
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
        },
        "num_batches_per_epoch": 100,
    },
    "WaveNet": {
        "max_epochs": 50,
        "trainer_kwargs": {
            "accelerator": "cpu",
            "devices": 1,
        },
        "num_batches_per_epoch": 100,
    },
}


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
    print("=" * 60)
    print("DL Model Training (CPU, single-threaded)")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    train, val, test = load_data()
    full_train = pd.concat([train, val])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  {len(full_data)} rows, {len(ts_cols)} series")

    # Build TSDF
    print("\n[2/5] Building TSDF...")
    tsdf = to_tsdf(full_data)
    train_tsdf = tsdf.slice_by_timestep(None, -PREDICTION_LENGTH)
    print(f"  Shape: {tsdf.shape}")

    # Train each DL model separately to avoid one failure killing all
    print("\n[3/5] Training DL models one by one...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, hp in DL_HYPERPARAMETERS.items():
        model_path = MODELS_DIR / model_name
        print(f"\n  --- {model_name} ---")

        predictor = TimeSeriesPredictor(
            target="target",
            prediction_length=PREDICTION_LENGTH,
            freq=FREQ,
            eval_metric="MASE",
            path=str(model_path),
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            verbosity=2,
        )

        start = time.time()
        try:
            predictor.fit(
                train_tsdf,
                time_limit=1200,  # 20 min per model
                presets="high_quality",
                num_val_windows=1,
                enable_ensemble=False,
                random_seed=42,
                hyperparameters={model_name: hp},
            )
            elapsed = time.time() - start
            print(f"  {model_name} trained in {elapsed:.0f}s")

            # Get leaderboard
            lb = predictor.leaderboard(silent=True)
            print(f"  Score: {lb['score_val'].iloc[0]:.4f}")

            # Predict
            preds = predictor.predict(train_tsdf)

            # Metrics
            metrics = {}
            for sid in ts_cols:
                actual = full_data[sid].iloc[-PREDICTION_LENGTH:].values
                pred_mean = preds.loc[sid]["mean"].values[:PREDICTION_LENGTH]
                metrics[sid] = compute_metrics(actual, pred_mean)

            metrics_df = pd.DataFrame(metrics).T
            avg = metrics_df.mean()
            print(f"  Avg MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, R2={avg['R2']:.4f}")

            # Save
            metrics_df.to_csv(RESULTS_DIR / f"dl_metrics_{model_name}.csv")
            preds.to_csv(RESULTS_DIR / f"dl_forecasts_{model_name}.csv")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  {model_name} FAILED after {elapsed:.0f}s: {e}")
            continue

    # Summary
    print("\n[4/5] Summary of DL models:")
    for model_name in DL_HYPERPARAMETERS:
        metrics_path = RESULTS_DIR / f"dl_metrics_{model_name}.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path, index_col=0)
            avg = df.mean()
            print(f"  {model_name:35s} MAE={avg['MAE']:.4f} RMSE={avg['RMSE']:.4f} R2={avg['R2']:.4f}")
        else:
            print(f"  {model_name:35s} NOT TRAINED")

    print("\n[5/5] Done!")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Models: {MODELS_DIR}")


if __name__ == "__main__":
    main()
