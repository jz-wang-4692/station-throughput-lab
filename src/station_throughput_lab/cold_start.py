"""
Hierarchical cold-start imputation for new stations.

When a station has no history (or very little), we can't compute lag
or rolling features. This module implements a hierarchical fallback:

    1. Spatial cluster average — stations in the same DBSCAN cluster
    2. Borough average — stations in the same borough proxy
    3. City-wide average — global average across all mature stations

At each level, the imputed value is scaled by a capacity ratio
(station_density_1km / cluster_avg_density) to account for the fact
that a station in a dense area will naturally have higher throughput
than one in a sparse area.

The imputation is applied to lag and rolling features that are NaN
for cold-start stations. The model then learns from both the imputed
features and the "is_new_station" / "station_age_days" indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from station_throughput_lab.config import (
    COLD_START_HOLDOUT_DAYS,
    COLD_START_MIN_MATURE_DAYS,
    TRAIN_END,
)


def identify_cold_start_stations(
    feature_panel: pd.DataFrame,
) -> tuple[set, set]:
    """Split stations into cold-start and mature sets.

    Cold-start: stations with fewer than COLD_START_HOLDOUT_DAYS of history
                at the time of prediction.
    Mature: stations with at least COLD_START_MIN_MATURE_DAYS of history.

    Returns (cold_start_ids, mature_ids).
    """
    station_age = (
        feature_panel.groupby("station_id")["station_age_days"]
        .max()
        .reset_index()
    )
    cold = set(
        station_age.loc[
            station_age["station_age_days"] < COLD_START_HOLDOUT_DAYS,
            "station_id",
        ]
    )
    mature = set(
        station_age.loc[
            station_age["station_age_days"] >= COLD_START_MIN_MATURE_DAYS,
            "station_id",
        ]
    )
    return cold, mature


def compute_hierarchical_baselines(
    feature_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute hierarchical throughput baselines from mature stations.

    Baselines are estimated from training-period station-days that are already
    mature by that date. This keeps calibration/test departures and future
    station maturity status out of features used for later evaluation.

    Returns a DataFrame with one row per (spatial_cluster, borough_proxy, dow)
    containing the average throughput at each hierarchy level.
    """
    train_end = pd.Timestamp(TRAIN_END)
    mature = feature_panel[
        (pd.to_datetime(feature_panel["date"]) <= train_end)
        & feature_panel["station_age_days"].ge(COLD_START_MIN_MATURE_DAYS)
    ].copy()

    # Level 1: spatial cluster × DOW
    cluster_avg = (
        mature.groupby(["spatial_cluster", "dow"], as_index=False)
        .agg(
            cluster_avg_dep=("departures", "mean"),
            cluster_median_dep=("departures", "median"),
            cluster_stations=("station_id", "nunique"),
            cluster_avg_density=("station_density_1km", "mean"),
        )
    )

    # Level 2: borough × DOW
    borough_avg = (
        mature.groupby(["borough_proxy", "dow"], as_index=False)
        .agg(
            borough_avg_dep=("departures", "mean"),
            borough_median_dep=("departures", "median"),
            borough_stations=("station_id", "nunique"),
        )
    )

    # Level 3: city-wide × DOW
    city_avg = (
        mature.groupby("dow", as_index=False)
        .agg(
            city_avg_dep=("departures", "mean"),
            city_median_dep=("departures", "median"),
        )
    )

    return cluster_avg, borough_avg, city_avg


def impute_cold_start_features(
    feature_panel: pd.DataFrame,
    cluster_avg: pd.DataFrame,
    borough_avg: pd.DataFrame,
    city_avg: pd.DataFrame,
) -> pd.DataFrame:
    """Apply hierarchical imputation to cold-start station features.

    For each cold-start row where lag/rolling features are NaN:
    1. Try spatial cluster average (scaled by density ratio)
    2. Fall back to borough average
    3. Fall back to city-wide average

    The imputed values replace NaN in lag/rolling columns. An indicator
    column 'is_imputed' marks which rows received imputation.
    """
    df = feature_panel.copy()

    # Identify rows that need imputation (new stations with NaN lags)
    lag_cols = [
        c for c in df.columns
        if c.startswith("dep_lag_")
        or c.startswith("rolling_mean_")
        or c.startswith("rolling_median_")
    ]
    needs_imputation = df["is_new_station"].eq(1) & df[lag_cols].isna().any(axis=1)
    df["is_imputed"] = needs_imputation.astype(int)

    if not needs_imputation.any():
        return df

    # Join hierarchical baselines
    df = df.merge(
        cluster_avg[["spatial_cluster", "dow", "cluster_avg_dep", "cluster_avg_density"]],
        on=["spatial_cluster", "dow"],
        how="left",
    )
    df = df.merge(
        borough_avg[["borough_proxy", "dow", "borough_avg_dep"]],
        on=["borough_proxy", "dow"],
        how="left",
    )
    df = df.merge(
        city_avg[["dow", "city_avg_dep"]],
        on="dow",
        how="left",
    )

    # Capacity-ratio scaling: adjust cluster average by local density
    density_ratio = (
        df["station_density_1km"] / df["cluster_avg_density"].replace(0, np.nan)
    ).fillna(1.0).clip(0.3, 3.0)

    # Hierarchical fallback
    imputed_value = (
        (df["cluster_avg_dep"] * density_ratio)
        .fillna(df["borough_avg_dep"])
        .fillna(df["city_avg_dep"])
        .fillna(0)
    )

    # Apply imputation to NaN lag/rolling columns
    for col in lag_cols:
        mask = needs_imputation & df[col].isna()
        if mask.any():
            df.loc[mask, col] = imputed_value[mask]

    # Also impute DOW rolling central-tendency features
    for col in ["dow_rolling_mean_4", "dow_rolling_median_4"]:
        if col not in df.columns:
            continue
        mask = needs_imputation & df[col].isna()
        if mask.any():
            df.loc[mask, col] = imputed_value[mask]

    # Clean up temporary columns
    drop_cols = [
        "cluster_avg_dep", "cluster_avg_density",
        "borough_avg_dep", "city_avg_dep",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    n_imputed = needs_imputation.sum()
    n_stations = df.loc[needs_imputation, "station_id"].nunique()
    print(f"Cold-start imputation: {n_imputed:,} rows across "
          f"{n_stations} stations received hierarchical fill.")

    return df
