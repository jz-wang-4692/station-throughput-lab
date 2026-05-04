"""
Temporal and contextual feature engineering for daily station throughput.

Builds the feature panel that feeds AutoGluon, including:
- Day-of-week, month, holiday indicators
- Lag features: same-day-last-week, 2-week, 4-week
- Rolling statistics: mean, median, std over 7/14/28-day windows
- DOW-specific rolling means (e.g., average Monday throughput over last 4 Mondays)
- Weather features (temperature, precipitation, snow)
- Spatial neighbor throughput (KNN average departures yesterday)
- Station maturity and capacity proxy features
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from station_throughput_lab.config import (
    PROCESSED_DIR,
    KNN_NEIGHBORS,
)


# US federal holidays (approximate, covers 2024)
US_HOLIDAYS = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-31",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
}


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month, quarter, holiday, weekend indicators."""
    df["dow"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(US_HOLIDAYS).astype(int)
    df["is_holiday_or_weekend"] = (df["is_weekend"] | df["is_holiday"]).astype(int)

    # Days to/from nearest holiday — captures pre/post-holiday demand ramps
    holiday_dates = sorted(pd.to_datetime(list(US_HOLIDAYS)))
    dates = df["date"].drop_duplicates().sort_values()
    date_holiday_map = {}
    for d in dates:
        days_to = [abs((h - d).days) for h in holiday_dates]
        date_holiday_map[d] = min(days_to) if days_to else 30
    df["days_to_nearest_holiday"] = df["date"].map(date_holiday_map).fillna(30).clip(0, 30)

    # Cyclical encoding for DOW and month
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features: 1d, 7d (same DOW last week), 14d, 28d."""
    group = df.groupby("station_id", group_keys=False)

    for lag in [1, 2, 7, 14, 28]:
        df[f"dep_lag_{lag}"] = group["departures"].shift(lag)

    # Same-DOW lags: average of same weekday over last 2, 4 weeks
    # Use nanmean semantics so that when only one lag is available
    # we use that value instead of averaging with zero.
    _lag7 = df["dep_lag_7"]
    _lag14 = df["dep_lag_14"]
    _stack = pd.concat([_lag7, _lag14], axis=1)
    df["dep_lag_7_14_avg"] = _stack.mean(axis=1, skipna=True)

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, median, std over 7/14/28-day windows."""
    group = df.groupby("station_id", group_keys=False)

    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = group["departures"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).mean()
        )
        df[f"rolling_median_{window}"] = group["departures"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).median()
        )

    # Rolling std and max for variability signal
    df["rolling_std_7"] = group["departures"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).std()
    )
    df["rolling_std_14"] = group["departures"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=3).std()
    )
    df["rolling_max_14"] = group["departures"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=3).max()
    )
    df["rolling_min_14"] = group["departures"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=3).min()
    )

    # Coefficient of variation (normalized volatility)
    df["cv_7"] = df["rolling_std_7"] / df["rolling_mean_7"].replace(0, np.nan)

    # Trend: 7-day mean vs 28-day mean
    df["trend_7_vs_28"] = (
        df["rolling_mean_7"] / df["rolling_mean_28"].replace(0, np.nan)
    )

    # Zero-day rate: fraction of zero-departure days in last 14 days
    df["zero_day_rate_14"] = group["departures"].transform(
        lambda s: (s.shift(1) == 0).rolling(14, min_periods=3).mean()
    )

    return df


def _add_dow_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """DOW-specific rolling statistics: average and median throughput on the
    same weekday over the last 4 occurrences. This captures weekday-specific
    patterns (e.g., commuter stations peak on weekdays, park stations on
    weekends).
    """
    group = df.groupby(["station_id", "dow"], group_keys=False)
    df["dow_rolling_mean_4"] = group["departures"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    df["dow_rolling_median_4"] = group["departures"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).median()
    )

    # Weekend vs weekday ratio per station (from recent history)
    # Compute rolling means within each station, then take the ratio.
    # We use a single groupby and conditional rolling to avoid index
    # alignment issues from filtering + transform on subsets.
    def _weekend_weekday_ratio(sdf: pd.DataFrame) -> pd.Series:
        shifted = sdf["departures"].shift(1)
        is_wkend = sdf["is_weekend"] == 1
        # Expanding mean of weekend and weekday departures separately
        wkend_vals = shifted.where(is_wkend)
        wkday_vals = shifted.where(~is_wkend)
        wkend_mean = wkend_vals.expanding(min_periods=2).mean().ffill()
        wkday_mean = wkday_vals.expanding(min_periods=2).mean().ffill()
        ratio = (wkend_mean / wkday_mean.replace(0, np.nan)).fillna(1.0).clip(0.1, 10.0)
        return ratio

    df["weekend_weekday_ratio"] = (
        df[["departures", "is_weekend"]]
        .groupby(df["station_id"], group_keys=False)
        .apply(_weekend_weekday_ratio)
        .reset_index(level=0, drop=True)
    )

    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather features from bundled or downloaded weather data.

    If weather data isn't available, creates synthetic seasonal proxies
    based on day-of-year (temperature follows a sinusoidal pattern in NYC).
    """
    weather_path = PROCESSED_DIR / "nyc_weather_daily.parquet"

    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        weather["date"] = pd.to_datetime(weather["date"])
        df = df.merge(weather, on="date", how="left")
        for col in ["temp_max_f", "temp_min_f", "precip_in", "snow_in"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)
    else:
        # Synthetic seasonal temperature proxy (NYC pattern)
        doy = df["day_of_year"].values
        df["temp_max_f"] = 55 + 30 * np.sin(2 * np.pi * (doy - 100) / 365)
        df["temp_min_f"] = df["temp_max_f"] - 15
        df["precip_in"] = 0.0
        df["snow_in"] = 0.0

    df["temp_avg_f"] = (df["temp_max_f"] + df["temp_min_f"]) / 2
    df["is_cold"] = (df["temp_avg_f"] < 35).astype(int)
    df["is_hot"] = (df["temp_avg_f"] > 85).astype(int)
    df["is_rainy"] = (df["precip_in"] > 0.1).astype(int)
    df["is_snowy"] = (df["snow_in"] > 0.1).astype(int)

    # Weather interaction features — weather impact varies by context
    df["cold_weekend"] = df["is_cold"] * df["is_weekend"]
    df["rain_weekend"] = df["is_rainy"] * df["is_weekend"]

    daily_temp = (
        df[["date", "temp_avg_f"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    recent_norm = daily_temp["temp_avg_f"].shift(1).rolling(28, min_periods=7).mean()
    daily_temp["temp_deviation"] = daily_temp["temp_avg_f"] - recent_norm
    df = df.merge(daily_temp[["date", "temp_deviation"]], on="date", how="left")
    df["temp_deviation"] = df["temp_deviation"].fillna(0)

    return df


def _add_neighbor_features(
    df: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """KNN neighbor throughput: average departures yesterday at the
    k nearest stations. This captures local demand spillover and
    provides a spatial smoothing signal.
    """
    meta = station_meta[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    coords_rad = np.radians(meta[["lat", "lon"]].values)
    station_ids = meta["station_id"].values

    k = min(KNN_NEIGHBORS + 1, len(meta))
    tree = BallTree(coords_rad, metric="haversine")
    _, indices = tree.query(coords_rad, k=k)

    # Build station_id -> neighbor station_ids mapping (exclude self)
    neighbor_map = {}
    for i, sid in enumerate(station_ids):
        neighbor_map[sid] = [station_ids[j] for j in indices[i, 1:]]

    # For each date, compute average neighbor departures from yesterday
    yesterday = df[["station_id", "date", "departures"]].copy()
    yesterday["date"] = yesterday["date"] + pd.Timedelta(days=1)
    yesterday = yesterday.rename(columns={"departures": "dep_yesterday"})

    # Expand neighbor map
    rows = []
    for sid, neighbors in neighbor_map.items():
        for nsid in neighbors:
            rows.append({"station_id": sid, "neighbor_id": nsid})
    neighbor_df = pd.DataFrame(rows)

    # Join yesterday's departures for each neighbor
    neighbor_dep = neighbor_df.merge(
        yesterday.rename(columns={"station_id": "neighbor_id"}),
        on="neighbor_id",
        how="inner",
    )
    knn_avg = (
        neighbor_dep.groupby(["station_id", "date"], as_index=False)
        ["dep_yesterday"].mean()
        .rename(columns={"dep_yesterday": "knn_avg_dep_yesterday"})
    )

    df = df.merge(knn_avg, on=["station_id", "date"], how="left")
    df["knn_avg_dep_yesterday"] = df["knn_avg_dep_yesterday"].fillna(0)

    return df


def _add_station_maturity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Station age, maturity indicators, and capacity proxy."""
    df["log_station_age"] = np.log1p(df["station_age_days"])
    df["is_new_station"] = (df["station_age_days"] < 60).astype(int)
    df["is_very_new"] = (df["station_age_days"] < 30).astype(int)

    # Capacity proxy: station's historical average throughput from prior days
    # This acts as a "station size" indicator — high-throughput stations tend
    # to stay high-throughput
    station_avg = df.groupby("station_id")["departures"].transform(
        lambda s: s.shift(1).expanding(min_periods=7).mean()
    )
    df["station_historical_avg"] = station_avg.fillna(0)
    df["log_station_historical_avg"] = np.log1p(df["station_historical_avg"])

    return df


def fill_missing_features(df: pd.DataFrame, train_end: str | None = None) -> pd.DataFrame:
    """Fill NaN values in feature columns with sensible defaults.

    When train_end is provided, fill values (medians) are computed only
    from rows up to that date to prevent data leakage from future splits.
    """
    if train_end is not None:
        train_mask = df["date"] <= pd.Timestamp(train_end)
        train_df = df.loc[train_mask]
    else:
        train_df = df

    lag_cols = [c for c in df.columns if c.startswith("dep_lag_") or c.startswith("rolling_")]
    for col in lag_cols:
        # Compute fill values from training data only
        station_median_map = train_df.groupby("station_id")[col].median()
        station_median = df["station_id"].map(station_median_map)
        global_median = train_df[col].median()
        if pd.isna(global_median):
            global_median = 0.0
        df[col] = (
            df[col]
            .fillna(station_median)
            .fillna(global_median)
            .replace([np.inf, -np.inf], global_median)
        )

    fill_zero_cols = [
        "trend_7_vs_28", "dow_rolling_mean_4", "dow_rolling_median_4",
        "knn_avg_dep_yesterday", "cv_7", "zero_day_rate_14",
        "weekend_weekday_ratio", "station_historical_avg",
        "log_station_historical_avg",
        "city_total_yesterday", "city_total_rolling_7", "city_total_rolling_28",
        "log_city_total_yesterday", "log_city_total_rolling_7",
        "log_city_total_rolling_28",
        "cluster_total_yesterday", "log_cluster_total_yesterday",
        "days_to_nearest_holiday",
        "cold_weekend", "rain_weekend", "temp_deviation",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    return df


def build_feature_panel(
    daily_panel: pd.DataFrame,
    station_geo: pd.DataFrame,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build the full feature panel for modeling.

    Joins geo features and adds temporal/lag/rolling/weather/neighbor features.
    Missing lag and rolling values are intentionally left in place so the
    cold-start imputer can fill them before the final generic missing-value pass.
    """
    cache_path = PROCESSED_DIR / "feature_panel.parquet"
    meta_path = PROCESSED_DIR / "feature_panel.metadata.json"
    meta = {
        "rows": len(daily_panel),
        "stations": int(daily_panel["station_id"].nunique()),
        "feature_version": 5,
    }

    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        try:
            if json.loads(meta_path.read_text()) == meta:
                print("Loading cached feature panel ...")
                return pd.read_parquet(cache_path)
        except (json.JSONDecodeError, KeyError):
            pass

    print("Building feature panel ...")
    df = daily_panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Join geo features
    geo_cols = [
        "station_id", "spatial_cluster", "dist_to_center_km",
        "dist_to_manhattan_km", "station_density_1km",
        "knn_avg_dist_km", "borough_proxy",
    ]
    available_geo = [c for c in geo_cols if c in station_geo.columns]
    df = df.merge(station_geo[available_geo], on="station_id", how="left")

    # Station metadata for neighbor features
    station_meta = station_geo[["station_id", "lat", "lon"]].drop_duplicates("station_id")

    df = _add_calendar_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_dow_rolling_features(df)
    df = _add_weather_features(df)
    df = _add_neighbor_features(df, station_meta)
    df = _add_station_maturity_features(df)

    # --- City-wide and cluster-level demand signals ---
    # These capture network-level demand context: "is today a high-demand
    # day across the whole system?" This helps the model handle seasonal
    # shifts and holidays without explicit holiday features.
    city_daily = (
        df.groupby("date", as_index=False)["departures"]
        .sum()
        .rename(columns={"departures": "_city_total"})
    )
    city_daily["city_total_yesterday"] = city_daily["_city_total"].shift(1)
    city_daily["city_total_rolling_7"] = (
        city_daily["_city_total"].shift(1).rolling(7, min_periods=1).mean()
    )
    city_daily["city_total_rolling_28"] = (
        city_daily["_city_total"].shift(1).rolling(28, min_periods=3).mean()
    )
    df = df.merge(
        city_daily[["date", "city_total_yesterday", "city_total_rolling_7",
                     "city_total_rolling_28"]],
        on="date", how="left",
    )
    # Log-transform city totals (large numbers)
    for col in ["city_total_yesterday", "city_total_rolling_7", "city_total_rolling_28"]:
        df[col] = df[col].fillna(0)
        df[f"log_{col}"] = np.log1p(df[col])

    # Cluster-level demand signal
    if "spatial_cluster" in df.columns:
        cluster_daily = (
            df.groupby(["spatial_cluster", "date"], as_index=False)["departures"]
            .sum()
            .rename(columns={"departures": "_cluster_total"})
        )
        cluster_daily["cluster_total_yesterday"] = (
            cluster_daily.groupby("spatial_cluster")["_cluster_total"].shift(1)
        )
        df = df.merge(
            cluster_daily[["spatial_cluster", "date", "cluster_total_yesterday"]],
            on=["spatial_cluster", "date"], how="left",
        )
        df["cluster_total_yesterday"] = df["cluster_total_yesterday"].fillna(0)
        df["log_cluster_total_yesterday"] = np.log1p(df["cluster_total_yesterday"])

    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Feature panel: {len(df):,} rows, {df.shape[1]} columns, "
          f"{df['station_id'].nunique():,} stations.")
    return df
