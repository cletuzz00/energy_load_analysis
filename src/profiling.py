"""
Profile aggregation and segmentation helpers for 15-minute load data.
Expects DataFrame with calendar features (hour, day_of_week, season, is_weekend)
from cleaning.add_calendar_features / run_cleaning_pipeline.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import ALL_CIRCUIT_COLUMNS, WHOLE_HOME_COLUMN


def build_daily_profile(
    df: pd.DataFrame,
    value_col: str = WHOLE_HOME_COLUMN,
    group_cols: Optional[list] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Build daily profile: average load by hour of day (0-23).
    Returns aggregated table with group_cols + hour (if not in group_cols) and value_col aggregated.
    """
    if group_cols is None:
        group_cols = ["dataid"]
    if "hour" not in df.columns:
        raise ValueError("DataFrame must have 'hour' (run add_calendar_features first)")
    if "hour" not in group_cols:
        group_cols = list(group_cols) + ["hour"]
    out = df.groupby(group_cols, dropna=False)[value_col].agg(agg).reset_index()
    return out


def build_seasonal_profile(
    df: pd.DataFrame,
    value_col: str = WHOLE_HOME_COLUMN,
    group_cols: Optional[list] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Build seasonal profile: average load by season and hour of day.
    Expects 'season' (1-4) and 'hour' in df.
    """
    if group_cols is None:
        group_cols = ["dataid"]
    for c in ("season", "hour"):
        if c not in df.columns:
            raise ValueError(f"DataFrame must have '{c}' (run add_calendar_features first)")
        if c not in group_cols:
            group_cols = list(group_cols) + [c]
    out = df.groupby(group_cols, dropna=False)[value_col].agg(agg).reset_index()
    return out


def build_weekly_profile(
    df: pd.DataFrame,
    value_col: str = WHOLE_HOME_COLUMN,
    group_cols: Optional[list] = None,
    use_hour_of_week: bool = True,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Build weekly profile: average load by time of week.
    If use_hour_of_week=True: group by hour_of_week = day_of_week * 24 + hour (0-167).
    Otherwise: group by (is_weekend, hour) for weekday vs weekend by hour.
    """
    if group_cols is None:
        group_cols = ["dataid"]
    df = df.copy()
    if "day_of_week" not in df.columns or "hour" not in df.columns:
        raise ValueError("DataFrame must have 'day_of_week' and 'hour' (run add_calendar_features first)")
    if use_hour_of_week:
        df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]
        time_col = "hour_of_week"
    else:
        if "is_weekend" not in df.columns:
            df["is_weekend"] = df["day_of_week"].isin([5, 6])
        time_col = "is_weekend"
        if time_col not in group_cols:
            group_cols = list(group_cols) + [time_col]
        if "hour" not in group_cols:
            group_cols = list(group_cols) + ["hour"]
    if time_col not in group_cols:
        group_cols = list(group_cols) + [time_col]
    out = df.groupby(group_cols, dropna=False)[value_col].agg(agg).reset_index()
    return out


def household_features_from_grid(
    df: pd.DataFrame,
    value_col: str = WHOLE_HOME_COLUMN,
    include_shape_24: bool = True,
    include_metadata: bool = False,
    meta_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build per-household (dataid) feature matrix from grid 15-min series.
    Features: daily mean load, (optional) 24-dim load shape, peak hour, weekend/weekday ratio,
    seasonality (e.g. std across season means). Optionally merge metadata columns.
    """
    if value_col not in df.columns or "dataid" not in df.columns:
        raise ValueError("DataFrame must have 'dataid' and value_col")
    for c in ("hour", "season", "is_weekend"):
        if c not in df.columns:
            raise ValueError(f"DataFrame must have '{c}' (run add_calendar_features first)")

    gr = df.groupby("dataid")
    # Basic usage
    daily_mean = gr[value_col].mean()
    peak_hour = gr.apply(
        lambda g: g.groupby("hour", group_keys=False)[value_col].mean().idxmax()
        if g[value_col].notna().any() else np.nan,
        include_groups=False,
    )
    peak_hour = peak_hour.reindex(daily_mean.index)

    # Weekend vs weekday ratio (weekend mean / weekday mean)
    weekday_mean = df[~df["is_weekend"]].groupby("dataid")[value_col].mean()
    weekend_mean = df[df["is_weekend"]].groupby("dataid")[value_col].mean()
    weekend_ratio = (weekend_mean / weekday_mean).reindex(daily_mean.index).fillna(0)

    # Seasonality: std of per-season mean loads
    season_means = df.groupby(["dataid", "season"])[value_col].mean().unstack(level="season")
    seasonality_strength = season_means.std(axis=1).reindex(daily_mean.index).fillna(0)

    out = pd.DataFrame({
        "daily_mean_kw": daily_mean,
        "peak_hour": peak_hour,
        "weekend_weekday_ratio": weekend_ratio,
        "seasonality_std": seasonality_strength,
    }, index=daily_mean.index).reset_index()

    if include_shape_24:
        by_hour = df.groupby(["dataid", "hour"])[value_col].mean().reset_index()
        pivot = by_hour.pivot(index="dataid", columns="hour", values=value_col)
        for h in range(24):
            if h in pivot.columns:
                out[f"shape_h{h}"] = out["dataid"].map(pivot[h]).values

    if include_metadata and meta_cols:
        meta = df.drop_duplicates("dataid")[["dataid"] + [c for c in meta_cols if c in df.columns]]
        out = out.merge(meta, on="dataid", how="left")

    return out


def cluster_households(
    features: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: Optional[list] = None,
    use_pca: bool = False,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> pd.Series:
    """
    Cluster households by feature matrix. Returns Series index=dataid, value=segment label (0..n_clusters-1).
    feature_cols: list of column names to use; if None, use numeric columns except dataid.
    """
    if "dataid" not in features.columns:
        raise ValueError("features must have 'dataid'")
    if feature_cols is None:
        feature_cols = [c for c in features.select_dtypes(include=[np.number]).columns if c != "dataid"]
    X = features[feature_cols].copy()
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if use_pca and (n_components or X.shape[1] > 2):
        n_components = n_components or min(10, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=random_state)
        X_scaled = pca.fit_transform(X_scaled)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return pd.Series(
        labels,
        index=pd.Index(features["dataid"].values, name="dataid"),
        name="segment",
    )


def circuit_features_for_home(
    df: pd.DataFrame,
    dataid: int,
    grid_col: str = WHOLE_HOME_COLUMN,
    circuit_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build per-circuit feature matrix for one household: mean, std, peak hour, correlation with grid.
    Returns DataFrame with one row per circuit, columns: circuit, mean_kw, std_kw, peak_hour, corr_with_grid.
    """
    from .cleaning import get_power_columns
    sub = df.loc[df["dataid"] == dataid].copy()
    if sub.empty:
        return pd.DataFrame()
    if "hour" not in sub.columns:
        return pd.DataFrame()
    if circuit_cols is None:
        circuit_cols = [c for c in get_power_columns(sub) if c != grid_col and c in sub.columns]
    grid_ser = pd.to_numeric(sub[grid_col], errors="coerce")
    rows = []
    for col in circuit_cols:
        if col not in sub.columns:
            continue
        ser = pd.to_numeric(sub[col], errors="coerce")
        if ser.isna().all() or ser.eq(0).all():
            continue
        by_hour = sub.groupby("hour")[col].mean()
        peak_hour = by_hour.idxmax() if by_hour.notna().any() else np.nan
        corr = ser.corr(grid_ser) if grid_ser.notna().any() and ser.notna().any() else np.nan
        rows.append({
            "circuit": col,
            "mean_kw": ser.mean(),
            "std_kw": ser.std(),
            "peak_hour": peak_hour,
            "corr_with_grid": corr,
        })
    return pd.DataFrame(rows)


def cluster_circuits(
    circuit_features: pd.DataFrame,
    n_clusters: int = 3,
    feature_cols: Optional[list] = None,
    random_state: int = 42,
) -> pd.Series:
    """
    Cluster circuits by feature matrix. Returns Series index=circuit name, value=cluster label.
    """
    if feature_cols is None:
        feature_cols = [c for c in circuit_features.select_dtypes(include=[np.number]).columns if c != "circuit"]
    if not feature_cols:
        return pd.Series(dtype=int)
    X = circuit_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=min(n_clusters, len(circuit_features)), random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return pd.Series(labels, index=circuit_features["circuit"].values, name="cluster")


def daily_vectors_for_home(
    df: pd.DataFrame,
    dataid: int,
    value_col: str = WHOLE_HOME_COLUMN,
) -> pd.DataFrame:
    """
    For one household, build one row per day: date and 96-interval load vector (15-min intervals).
    Returns DataFrame with columns date, interval_0..interval_95.
    """
    sub = df.loc[df["dataid"] == dataid].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["date"] = pd.to_datetime(sub["local_15min"]).dt.date
    sub = sub.sort_values("local_15min")
    vectors = []
    for date, group in sub.groupby("date"):
        v = pd.to_numeric(group[value_col], errors="coerce").values[:96]
        v = np.resize(np.asarray(v, dtype=float), 96)
        vectors.append((date, v))
    if not vectors:
        return pd.DataFrame()
    out = pd.DataFrame([v for _, v in vectors], columns=[f"interval_{i}" for i in range(96)])
    out.insert(0, "date", [d for d, _ in vectors])
    return out


def cluster_days(
    daily_vectors: pd.DataFrame,
    n_clusters: int = 3,
    vector_cols: Optional[list] = None,
    use_pca: bool = True,
    n_components: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """
    Cluster days by 96-interval vectors (or subset). Returns Series index=date, value=regime label.
    """
    if vector_cols is None:
        vector_cols = [c for c in daily_vectors.columns if c.startswith("interval_")]
    if not vector_cols or "date" not in daily_vectors.columns:
        return pd.Series(dtype=int)
    X = daily_vectors[vector_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if use_pca and n_components < X.shape[1]:
        pca = PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1]), random_state=random_state)
        X_scaled = pca.fit_transform(X_scaled)
    n_clusters = min(n_clusters, len(daily_vectors))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return pd.Series(labels, index=daily_vectors["date"].values, name="regime")
