"""
Load 15-minute Pecan Street data and metadata by region; join and validate.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import REGIONS, METADATA_KEY_COLUMNS, PROJECT_ROOT, ALL_CIRCUIT_COLUMNS, NON_POWER_COLUMNS


def load_15min_data(
    region: str,
    dataid_list: Optional[list] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load 15-minute CSV for a region. Parse local_15min to datetime.
    Optionally filter by dataid and date range.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}. Use one of {list(REGIONS.keys())}")
    path = REGIONS[region]["data_path"]
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path, low_memory=False)
    # Keep only known columns (id, time, voltage, and all circuit columns from config)
    known = set(NON_POWER_COLUMNS) | set(ALL_CIRCUIT_COLUMNS)
    cols = [c for c in df.columns if c in known]
    if cols:
        df = df[cols]
    df["local_15min"] = pd.to_datetime(df["local_15min"], utc=True)

    if dataid_list is not None:
        df = df[df["dataid"].isin(dataid_list)]
    if date_start is not None:
        df = df[df["local_15min"] >= date_start]
    if date_end is not None:
        df = df[df["local_15min"] <= date_end]

    return df


def load_metadata(region: str) -> pd.DataFrame:
    """
    Load metadata for a region. Austin uses root metadata; CA/NY use folder metadata.
    Folder metadata files have a description row (row 2) that we skip.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")
    path = REGIONS[region]["metadata_path"]
    if not path.exists():
        raise FileNotFoundError(str(path))

    if region == "austin":
        meta = pd.read_csv(path, low_memory=False)
    else:
        meta = pd.read_csv(path, skiprows=[1], low_memory=False)

    meta["dataid"] = pd.to_numeric(meta["dataid"], errors="coerce")
    meta = meta.dropna(subset=["dataid"])
    meta["dataid"] = meta["dataid"].astype(int)

    return meta


def normalize_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize to common schema; ensure key columns exist."""
    out = pd.DataFrame()
    for col in METADATA_KEY_COLUMNS:
        if col in meta.columns:
            out[col] = meta[col]
        else:
            out[col] = None
    return out


def merge_data_with_metadata(
    df: pd.DataFrame, meta: pd.DataFrame, normalized: bool = True
) -> pd.DataFrame:
    """Left-join 15-min data with metadata on dataid."""
    if normalized:
        meta = normalize_metadata(meta)
    merged = df.merge(meta, on="dataid", how="left")
    return merged


def validate_coverage(df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """
    Return summary: unique dataids, date range, and for each dataid
    count of rows and (if meta has egauge windows) whether within window.
    """
    dataids = df["dataid"].unique()
    date_min = df["local_15min"].min()
    date_max = df["local_15min"].max()

    rows_per_home = df.groupby("dataid").size()
    out = {
        "n_homes": len(dataids),
        "date_min": date_min,
        "date_max": date_max,
        "total_rows": len(df),
        "rows_per_home": rows_per_home.to_dict(),
    }
    if "egauge_1min_min_time" in meta.columns and "egauge_1min_max_time" in meta.columns:
        meta_sub = meta[meta["dataid"].isin(dataids)].set_index("dataid")
        out["meta_has_window"] = True
    else:
        out["meta_has_window"] = False
    return out
