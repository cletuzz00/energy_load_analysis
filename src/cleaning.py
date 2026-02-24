"""
Cleaning and preprocessing of 15-minute load data.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .config import (
    ALL_CIRCUIT_COLUMNS,
    CIRCUITS_ALLOW_NEGATIVE,
    DATA_PROCESSED_DIR,
    WHOLE_HOME_COLUMN,
)

# Season labels: 1=winter, 2=spring, 3=summer, 4=fall (Northern Hemisphere)
_SEASON_LABELS = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}


def get_season(dates, label: bool = False):
    """
    Return season (1-4 or 'winter'/'spring'/'summer'/'fall') from datetime(s).
    Accepts a Series or a single timestamp. Same convention as add_calendar_features.
    """
    scalar = not hasattr(dates, "dt")
    if scalar:
        dates = pd.Series([pd.Timestamp(dates)])
    t = pd.to_datetime(dates)
    month = t.dt.month
    season_num = ((month % 12) + 3) // 3
    if label:
        out = season_num.map(_SEASON_LABELS)
    else:
        out = season_num
    if scalar:
        return out.iloc[0]
    return out


def is_weekend(dates):
    """Return True for Saturday/Sunday, False otherwise. Accepts Series or scalar datetime."""
    scalar = not hasattr(dates, "dt")
    if scalar:
        dates = pd.Series([pd.Timestamp(dates)])
    t = pd.to_datetime(dates)
    out = t.dt.dayofweek.isin([5, 6])
    if scalar:
        return out.iloc[0]
    return out


def get_day_type(dates):
    """Return 'weekend' or 'weekday'. Accepts Series or scalar datetime."""
    scalar = not hasattr(dates, "dt")
    if scalar:
        dates = pd.Series([pd.Timestamp(dates)])
    t = pd.to_datetime(dates)
    out = np.where(t.dt.dayofweek.isin([5, 6]), "weekend", "weekday")
    out = pd.Series(out, index=dates.index)
    if scalar:
        return out.iloc[0]
    return out


def get_power_columns(df: pd.DataFrame) -> list:
    """Return list of circuit columns that hold real power (kW), in config order."""
    return [c for c in ALL_CIRCUIT_COLUMNS if c in df.columns]


def clean_circuit_values(
    df: pd.DataFrame,
    power_cols: Optional[list] = None,
    clip_consumption_at_zero: bool = True,
    cap_percentile: Optional[float] = 99.0,
) -> pd.DataFrame:
    """
    Apply cleaning rules to power columns:
    - Consumption-only circuits: clip negative to 0 (optional).
    - Optional cap at percentile per home/circuit to limit outliers.
    """
    df = df.copy()
    if power_cols is None:
        power_cols = get_power_columns(df)

    for col in power_cols:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        if clip_consumption_at_zero and col not in CIRCUITS_ALLOW_NEGATIVE:
            ser = ser.clip(lower=0)
        df[col] = ser

    if cap_percentile is not None:
        for dataid, group in df.groupby("dataid"):
            idx = group.index
            for col in power_cols:
                if col not in df.columns:
                    continue
                g = pd.to_numeric(df.loc[idx, col], errors="coerce")
                cap = g.quantile(cap_percentile / 100)
                if np.isfinite(cap) and cap > 0:
                    df.loc[idx, col] = df.loc[idx, col].clip(upper=cap)
    return df


def add_calendar_features(df: pd.DataFrame, time_col: str = "local_15min") -> pd.DataFrame:
    """Add hour, day_of_week, month, season, is_weekend."""
    df = df.copy()
    t = pd.to_datetime(df[time_col])
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["season"] = (df["month"] % 12 + 3) // 3  # 1=winter, 2=spring, 3=summer, 4=fall
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    return df


def align_to_15min_grid(df: pd.DataFrame, time_col: str = "local_15min") -> pd.DataFrame:
    """Floor timestamps to 15-minute grid (:00, :15, :30, :45)."""
    df = df.copy()
    t = pd.to_datetime(df[time_col])
    df[time_col] = t.dt.floor("15min")
    return df


def compute_data_quality_summary(
    df: pd.DataFrame,
    power_cols: Optional[list] = None,
    grid_col: str = WHOLE_HOME_COLUMN,
) -> pd.DataFrame:
    """
    Per-home summary: total intervals, valid grid count, pct valid grid,
    date range. Optionally per month.
    """
    if power_cols is None:
        power_cols = get_power_columns(df)
    if grid_col not in df.columns:
        grid_col = None

    rows_per_home = df.groupby("dataid").size()
    out = pd.DataFrame({"total_intervals": rows_per_home})
    if grid_col:
        valid_count = df.groupby("dataid", group_keys=False).apply(
            lambda g: (pd.to_numeric(g[grid_col], errors="coerce").notna()).sum(),
            include_groups=False,
        )
        out["valid_grid_intervals"] = valid_count
        out["pct_valid_grid"] = (out["valid_grid_intervals"] / out["total_intervals"] * 100).round(2)
    out["date_min"] = df.groupby("dataid")["local_15min"].min()
    out["date_max"] = df.groupby("dataid")["local_15min"].max()
    return out


def run_cleaning_pipeline(
    df: pd.DataFrame,
    add_calendar: bool = True,
    align_time: bool = True,
    clip_negative_consumption: bool = True,
    cap_percentile: Optional[float] = 99.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full cleaning: align time, clean circuits, add calendar features.
    Returns (cleaned_df, quality_summary).
    """
    power_cols = get_power_columns(df)
    if align_time:
        df = align_to_15min_grid(df)
    df = clean_circuit_values(
        df,
        power_cols=power_cols,
        clip_consumption_at_zero=clip_negative_consumption,
        cap_percentile=cap_percentile,
    )
    if add_calendar:
        df = add_calendar_features(df)
    quality = compute_data_quality_summary(df, power_cols=power_cols)
    return df, quality


def write_cleaned_output(
    df: pd.DataFrame,
    quality_summary: pd.DataFrame,
    region: str,
    output_dir: Optional[Path] = None,
) -> None:
    """Write cleaned wide-format CSV and quality summary CSV to data/processed/."""
    if output_dir is None:
        output_dir = DATA_PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"cleaned_15min_{region}.csv"
    df.to_csv(out_path, index=False)
    quality_path = output_dir / f"data_quality_{region}.csv"
    quality_summary.to_csv(quality_path)
    return None
