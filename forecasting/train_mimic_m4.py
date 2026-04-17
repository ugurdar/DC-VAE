"""
Mimic M4 working example exactly: 12 series x 400 points, PL=48, hyperparameters=None.
Test if deadlock is from our hyperparameters dict or data format.
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
MODELS_DIR = Path(__file__).parent / "models" / "mimic_m4"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 48  # M4 same
FREQ = "5min"
MAX_POINTS = 400  # M4 same


def main():
    print("Loading TELCO...", flush=True)
    train = pd.read_csv(DATA_DIR / "TELCO_data_train.csv", parse_dates=["time"], index_col="time")
    val = pd.read_csv(DATA_DIR / "TELCO_data_val.csv", parse_dates=["time"], index_col="time")
    test = pd.read_csv(DATA_DIR / "TELCO_data_test.csv", parse_dates=["time"], index_col="time")
    full = pd.concat([train, val, test])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    full = full.iloc[-MAX_POINTS:]  # Last 400 points
    print(f"Wide: {full.shape}", flush=True)

    # To long format
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
    print(f"TSDF: {tsdf.shape} (12 series x {MAX_POINTS})", flush=True)

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
        time_limit=600,
        num_val_windows=2,
        enable_ensemble=False,
        random_seed=42,
        hyperparameters={"DeepAR": {}},  # ONLY DeepAR, empty = use defaults
    )
    print(f"Done in {time.time()-start:.0f}s", flush=True)
    lb = predictor.leaderboard(silent=True)
    print(lb.to_string(), flush=True)


if __name__ == "__main__":
    main()
