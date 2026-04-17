"""
Anomaly Utilities
=================
Shared helpers for loading TELCO anomaly labels and overlaying
anomaly regions on matplotlib plots.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_DIR = Path(__file__).parent.parent / "TELCO_labels"


def load_anomaly_labels() -> pd.DataFrame:
    """Load and concatenate anomaly labels from train/val/test CSVs.

    Handles tz-aware timestamps in label files by converting to tz-naive
    to match the TELCO data timestamps.
    """
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(LABEL_DIR / f"TELCO_labels_{split}.csv",
                         parse_dates=["time"], index_col="time")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        frames.append(df)
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return full


def get_anomaly_mask(labels_df: pd.DataFrame, series_id: str,
                     time_index: pd.DatetimeIndex) -> np.ndarray:
    """Return boolean anomaly mask aligned to the given time index.

    Uses nearest-match reindexing with 5-min tolerance to handle
    potential timestamp misalignment.
    """
    if series_id not in labels_df.columns:
        return np.zeros(len(time_index), dtype=bool)

    aligned = labels_df[[series_id]].reindex(
        time_index, method="nearest",
        tolerance=pd.Timedelta("5min")
    )
    return aligned[series_id].fillna(0).values.astype(bool)


def overlay_anomaly_regions(ax, time_index, anomaly_mask,
                            alpha: float = 0.12, color: str = "red",
                            label: str = "Anomaly"):
    """Draw shaded red regions on a matplotlib Axes for anomalous periods.

    Finds contiguous anomaly blocks and draws axvspan for each.
    Only the first block gets the legend label.
    """
    if anomaly_mask is None or not np.any(anomaly_mask):
        return

    changes = np.diff(anomaly_mask.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for i, (s, e) in enumerate(zip(starts, ends)):
        s_idx = min(s, len(time_index) - 1)
        e_idx = min(e, len(time_index) - 1)
        ax.axvspan(time_index[s_idx], time_index[e_idx],
                   alpha=alpha, color=color, zorder=0,
                   label=label if i == 0 else None)
