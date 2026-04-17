"""
Test DeepAR on TS1 with SHORT horizon (48) to match working M4 example.
Hypothesis: longer prediction_length increases context_length which triggers GluonTS deadlock.
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
MODELS_DIR = Path(__file__).parent / "ts1_forecast_short"
RESULTS_DIR = Path(__file__).parent / "results" / "ts1_short"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 48  # SHORT horizon (same as working M4 example)
FREQ = "5min"
MAX_POINTS = 400  # Same as M4 example


def main():
    print(f"TS1 Short Test: PL={PREDICTION_LENGTH}, MAX_POINTS={MAX_POINTS}", flush=True)

    train = pd.read_csv(DATA_DIR / "TS1_data_train.csv", parse_dates=["time"], index_col="time")
    val = pd.read_csv(DATA_DIR / "TS1_data_val.csv", parse_dates=["time"], index_col="time")
    test = pd.read_csv(DATA_DIR / "TS1_data_test.csv", parse_dates=["time"], index_col="time")
    full = pd.concat([train, val, test])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    full = full.iloc[-MAX_POINTS:]
    print(f"Data: {len(full)} rows", flush=True)

    # Build TSDF
    df = full.copy().rename(columns={"TS1": "target"})
    df["item_id"] = "TS1"
    df = df.reset_index().rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    tsdf = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")
    print(f"TSDF: {tsdf.shape}", flush=True)

    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=PREDICTION_LENGTH,
        freq=FREQ,
        eval_metric="MASE",
        path=str(MODELS_DIR),
        verbosity=2,
    )

    start = time.time()
    predictor.fit(
        tsdf.slice_by_timestep(None, -PREDICTION_LENGTH),
        time_limit=180,
        num_val_windows=2,
        enable_ensemble=False,
        random_seed=42,
        hyperparameters={"DeepAR": {}},
    )
    print(f"Trained in {time.time()-start:.0f}s", flush=True)

    lb = predictor.leaderboard(silent=True)
    print(lb.to_string(), flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
