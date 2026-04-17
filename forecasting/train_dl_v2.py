"""
Train DL forecasting models with AutoGluon - macOS ARM64 fix.
Uses spawn multiprocessing + OBJC fork safety + monkey-patched DataLoader.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch
# Monkey-patch DataLoader to always use num_workers=0
_orig_dataloader_init = torch.utils.data.DataLoader.__init__
def _patched_dataloader_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    _orig_dataloader_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _patched_dataloader_init

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
MODELS_DIR = Path(__file__).parent / "models" / "dl_models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 288
FREQ = "5min"

DL_MODELS = {
    "DeepAR": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1},
    },
    "SimpleFeedForward": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1},
    },
    "TemporalFusionTransformer": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1},
    },
    "PatchTST": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1},
    },
    "WaveNet": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1},
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
    print("=" * 60, flush=True)
    print("DL Training v2 (spawn + patched DataLoader)", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/4] Loading data...", flush=True)
    train, val, test = load_data()
    full_train = pd.concat([train, val])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  {len(full_data)} rows, {len(ts_cols)} series", flush=True)

    print("\n[2/4] Building TSDF...", flush=True)
    tsdf = to_tsdf(full_data)
    train_tsdf = tsdf.slice_by_timestep(None, -PREDICTION_LENGTH)
    print(f"  Shape: {tsdf.shape}", flush=True)

    print("\n[3/4] Training DL models...", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name, hp in DL_MODELS.items():
        model_path = MODELS_DIR / model_name
        print(f"\n  === {model_name} ===", flush=True)
        sys.stdout.flush()

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
                time_limit=900,
                presets="medium_quality",
                num_val_windows=1,
                enable_ensemble=False,
                random_seed=42,
                hyperparameters={model_name: hp},
            )
            elapsed = time.time() - start
            print(f"  Trained in {elapsed:.0f}s", flush=True)

            lb = predictor.leaderboard(silent=True)
            score = lb["score_val"].iloc[0]
            print(f"  Val score: {score:.4f}", flush=True)

            preds = predictor.predict(train_tsdf)

            metrics = {}
            for sid in ts_cols:
                actual = full_data[sid].iloc[-PREDICTION_LENGTH:].values
                pred_mean = preds.loc[sid]["mean"].values[:PREDICTION_LENGTH]
                metrics[sid] = compute_metrics(actual, pred_mean)

            metrics_df = pd.DataFrame(metrics).T
            avg = metrics_df.mean()
            print(f"  Avg MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, R2={avg['R2']:.4f}", flush=True)

            metrics_df.to_csv(RESULTS_DIR / f"dl_metrics_{model_name}.csv")
            preds.to_csv(RESULTS_DIR / f"dl_forecasts_{model_name}.csv")
            results[model_name] = {"status": "OK", "score": score, **avg.to_dict()}

        except Exception as e:
            elapsed = time.time() - start
            print(f"  FAILED after {elapsed:.0f}s: {e}", flush=True)
            results[model_name] = {"status": f"FAILED: {e}"}
            import traceback
            traceback.print_exc()
            continue

    print("\n[4/4] Summary:", flush=True)
    print("-" * 60, flush=True)
    for model_name, r in results.items():
        if r["status"] == "OK":
            print(f"  {model_name:35s} MAE={r['MAE']:.4f} R2={r['R2']:.4f}", flush=True)
        else:
            print(f"  {model_name:35s} {r['status']}", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
