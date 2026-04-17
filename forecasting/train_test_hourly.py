"""
HYPOTHESIS TEST: Deadlock root cause is freq='5min' -> huge lags_seq.
Resample TELCO data to hourly (freq='H', matching M4 exactly).
If this works -> confirmed 5min is the problem.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "TELCO_data"
MODELS_DIR = Path(__file__).parent / "models" / "test_hourly"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 48
FREQ = "H"           # <-- HOURLY (M4-like)
MAX_POINTS = 400


def main():
    print("Loading TELCO and resampling to hourly...", flush=True)
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val = pd.read_csv(DATA_DIR / "TELCO_data_val.csv", parse_dates=["time"], index_col="time")
    test = pd.read_csv(DATA_DIR / "TELCO_data_test.csv", parse_dates=["time"], index_col="time")
    full = pd.concat([train, val, test])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    # Resample to hourly (mean)
    full = full.resample("1h").mean().dropna()
    full = full.iloc[-MAX_POINTS:]
    print(f"Hourly data: {full.shape}", flush=True)

    ts_cols = sorted([c for c in full.columns if c.startswith("TS")])
    records = []
    for col in ts_cols:
        temp = full[[col]].copy().rename(columns={col: "target"})
        temp["item_id"] = col
        temp = temp.reset_index().rename(columns={"time": "timestamp"})
        temp["timestamp"] = pd.to_datetime(temp["timestamp"]).dt.tz_localize(None)
        records.append(temp)
    long_df = pd.concat(records, ignore_index=True)
    tsdf = TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id", timestamp_column="timestamp")
    print(f"TSDF: {tsdf.shape}", flush=True)

    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=PREDICTION_LENGTH,
        freq=FREQ,
        eval_metric="MASE",
        path=str(MODELS_DIR),
        quantile_levels=[0.1, 0.5, 0.9],
        verbosity=2,
    )

    start = time.time()
    predictor.fit(
        tsdf,
        time_limit=300,
        num_val_windows=2,
        enable_ensemble=False,
        random_seed=42,
        hyperparameters={"DeepAR": {}},
    )
    print(f"Done in {time.time()-start:.0f}s", flush=True)
    lb = predictor.leaderboard(silent=True)
    print(lb.to_string(), flush=True)
    print("HOURLY TEST: SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
