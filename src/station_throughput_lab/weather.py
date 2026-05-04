"""
Real weather data from NOAA GHCN-Daily for NYC Central Park.

Downloads daily weather observations (temperature, precipitation, snow)
from the NOAA daily summaries service, with a deterministic synthetic
fallback if the download is unavailable. Produces a clean daily weather
parquet that the feature pipeline merges automatically.

This replaces the synthetic sinusoidal temperature proxy with actual
observations, which captures real cold snaps, heat waves, rain events,
and snowstorms that drive throughput variation.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests

from station_throughput_lab.config import (
    NOAA_STATION_ID,
    PROCESSED_DIR,
)

# NOAA daily summaries service.
NOAA_CDO_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"


def _download_ghcn_daily(station_id: str, start: str, end: str) -> pd.DataFrame | None:
    """Download daily weather from NOAA CDO bulk data service (no API key)."""
    url = (
        f"{NOAA_CDO_BASE}"
        f"?dataset=daily-summaries"
        f"&stations={station_id}"
        f"&startDate={start}"
        f"&endDate={end}"
        f"&dataTypes=TMAX,TMIN,PRCP,SNOW,SNWD"
        f"&units=standard"
        f"&format=csv"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  NOAA CDO download failed: {e}")
        return None


def _generate_realistic_synthetic_weather(
    start: str, end: str,
) -> pd.DataFrame:
    """Generate realistic synthetic weather when real data is unavailable.

    Uses NYC climate normals with added daily noise to produce more
    realistic patterns than a pure sinusoidal proxy. Includes:
    - Temperature with day-to-day autocorrelation
    - Precipitation events (~30% of days, more in spring/fall)
    - Snow events in winter months
    """
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.RandomState(42)

    doy = dates.dayofyear.values
    # NYC climate normals: avg high ~62°F, amplitude ~25°F
    base_high = 55 + 28 * np.sin(2 * np.pi * (doy - 105) / 365)
    # Add autocorrelated noise (weather persistence)
    noise = np.zeros(len(dates))
    noise[0] = rng.normal(0, 5)
    for i in range(1, len(noise)):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 3)

    temp_max = base_high + noise
    temp_min = temp_max - 12 - rng.uniform(2, 8, len(dates))

    # Precipitation: ~30% of days, heavier in spring/fall
    precip_prob = 0.25 + 0.1 * np.sin(2 * np.pi * (doy - 60) / 365)
    has_precip = rng.random(len(dates)) < precip_prob
    precip_amount = np.where(
        has_precip,
        rng.exponential(0.3, len(dates)),
        0.0,
    )

    # Snow: only when temp_max < 38°F and there's precipitation
    snow_possible = temp_max < 38
    snow_amount = np.where(
        snow_possible & has_precip,
        precip_amount * rng.uniform(8, 14, len(dates)),  # snow:water ratio
        0.0,
    )

    return pd.DataFrame({
        "date": dates,
        "temp_max_f": np.round(temp_max, 1),
        "temp_min_f": np.round(temp_min, 1),
        "precip_in": np.round(precip_amount, 2),
        "snow_in": np.round(snow_amount, 1),
    })


def build_weather_data(
    start: str = "2024-04-01",
    end: str = "2024-12-31",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build daily weather data for the project period.

    Tries to download real NOAA data first, falls back to realistic
    synthetic data if the download fails.
    """
    cache_path = PROCESSED_DIR / "nyc_weather_daily.parquet"

    if cache_path.exists() and not force_rebuild:
        print("Loading cached weather data ...")
        return pd.read_parquet(cache_path)

    print("Fetching NYC weather data ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Try real NOAA data
    real_data = _download_ghcn_daily(NOAA_STATION_ID, start, end)

    if real_data is not None and len(real_data) > 100:
        print(f"  Downloaded {len(real_data)} days of real NOAA weather data.")
        weather = real_data.rename(columns={
            "DATE": "date",
            "TMAX": "temp_max_f",
            "TMIN": "temp_min_f",
            "PRCP": "precip_in",
            "SNOW": "snow_in",
        })
        weather["date"] = pd.to_datetime(weather["date"])
        for col in ["temp_max_f", "temp_min_f", "precip_in", "snow_in"]:
            if col in weather.columns:
                weather[col] = pd.to_numeric(weather[col], errors="coerce")
        weather = weather[["date", "temp_max_f", "temp_min_f", "precip_in", "snow_in"]].copy()
        # Fill any gaps
        full_dates = pd.date_range(start, end, freq="D")
        weather = (
            weather.set_index("date")
            .reindex(full_dates)
            .interpolate(method="linear")
            .ffill().bfill()
            .reset_index()
            .rename(columns={"index": "date"})
        )
        weather["snow_in"] = weather["snow_in"].fillna(0)
    else:
        print("  Real weather data unavailable — generating realistic synthetic data.")
        weather = _generate_realistic_synthetic_weather(start, end)

    weather.to_parquet(cache_path, index=False)
    print(f"  Weather data: {len(weather)} days saved.")
    return weather
