"""
Train Deep Learning models one-by-one in SEPARATE predictors.
Skips TemporalFusionTransformer (known deadlock on macOS ARM64).
Each model in own predictor so one failure doesn't kill others.
Uses CPU-forced trainer_kwargs + limited epochs.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

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
DL_DIR = Path(__file__).parent / "models" / "dl_models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DL_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 288
FREQ = "5min"

# DL models to try (TFT SKIPPED - known deadlock)
DL_MODELS = {
    "DeepAR": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    },
    "SimpleFeedForward": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    },
    "PatchTST": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    },
    "DLinear": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    },
    "TiDE": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    },
    "WaveNet": {
        "max_epochs": 30,
        "num_batches_per_epoch": 50,
        "trainer_kwargs": {"accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
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


def main():
    print("=" * 60, flush=True)
    print("DL-Only Training (one model per predictor, TFT skipped)", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/3] Loading data...", flush=True)
    train, val, test = load_data()
    full_train = pd.concat([train, val])
    full_train = full_train[~full_train.index.duplicated(keep="first")].sort_index()
    full_data = pd.concat([full_train, test])
    full_data = full_data[~full_data.index.duplicated(keep="first")].sort_index()
    ts_cols = sorted([c for c in full_data.columns if c.startswith("TS")])
    print(f"  {len(full_data)} rows, {len(ts_cols)} series", flush=True)

    print("\n[2/3] Building TSDF...", flush=True)
    tsdf = to_tsdf(full_data)
    train_tsdf = tsdf.slice_by_timestep(None, -PREDICTION_LENGTH)
    print(f"  Shape: {tsdf.shape}", flush=True)

    print(f"\n[3/3] Training {len(DL_MODELS)} DL models...", flush=True)

    results = {}
    for model_name, hp in DL_MODELS.items():
        model_path = DL_DIR / model_name
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
                time_limit=900,  # 15 min per model
                num_val_windows=1,
                enable_ensemble=False,
                random_seed=42,
                hyperparameters={model_name: hp},
            )
            elapsed = time.time() - start
            print(f"  Trained in {elapsed:.0f}s", flush=True)

            lb = predictor.leaderboard(silent=True)
            if len(lb) > 0:
                score = lb["score_val"].iloc[0]
                print(f"  Val score: {score:.4f}", flush=True)
                results[model_name] = {"status": "OK", "score": score, "time": elapsed}
            else:
                results[model_name] = {"status": "No models trained", "time": elapsed}

        except Exception as e:
            elapsed = time.time() - start
            print(f"  FAILED after {elapsed:.0f}s: {e}", flush=True)
            results[model_name] = {"status": f"FAILED: {e}", "time": elapsed}

    print("\n" + "=" * 60, flush=True)
    print("DL Summary:", flush=True)
    print("=" * 60, flush=True)
    for m, r in results.items():
        if r["status"] == "OK":
            print(f"  {m:30s} score={r['score']:.4f}  ({r['time']:.0f}s)", flush=True)
        else:
            print(f"  {m:30s} {r['status'][:60]} ({r['time']:.0f}s)", flush=True)

    # Save summary
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(RESULTS_DIR / "dl_only_summary.csv")
    print(f"\nSaved: {RESULTS_DIR / 'dl_only_summary.csv'}", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
