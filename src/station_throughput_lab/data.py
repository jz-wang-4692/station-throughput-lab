"""
Data acquisition and daily station-level throughput panel construction.

Downloads Citi Bike monthly trip CSVs from the public trip-data bucket,
aggregates to daily station-level departure counts, and joins
station metadata (lat/lon, capacity proxy, borough).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from station_throughput_lab.config import (
    ALL_MONTHS,
    CITIBIKE_BASE_URL,
    MIN_STATION_DAYS,
    PROCESSED_DIR,
    RAW_DIR,
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> Path:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=dest.name, leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def _read_trip_csv(path: Path) -> pd.DataFrame:
    """Read a single Citi Bike trip CSV (handles both old and new schemas).

    Citi Bike monthly zips contain multiple CSV shards (e.g.,
    202410-citibike-tripdata_1.csv through _6.csv). We read and
    concatenate ALL CSVs inside the zip.
    """
    if path.suffix == ".zip":
        frames = []
        with zipfile.ZipFile(path) as zf:
            csv_names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
            if not csv_names:
                raise ValueError(f"No CSV found inside {path}")
            for csv_name in csv_names:
                with zf.open(csv_name) as f:
                    frames.append(pd.read_csv(f, low_memory=False))
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    else:
        df = pd.read_csv(path, low_memory=False)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names across old/new Citi Bike schemas."""
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # New schema (2021+)
    if "started_at" in cols_lower:
        col_map[cols_lower["started_at"]] = "start_time"
        col_map[cols_lower.get("start_station_id", "")] = "station_id"
        col_map[cols_lower.get("start_station_name", "")] = "station_name"
        col_map[cols_lower.get("start_lat", "")] = "lat"
        col_map[cols_lower.get("start_lng", "")] = "lon"
    # Old schema
    elif "starttime" in cols_lower:
        col_map[cols_lower["starttime"]] = "start_time"
        col_map[cols_lower.get("start station id", "")] = "station_id"
        col_map[cols_lower.get("start station name", "")] = "station_name"
        col_map[cols_lower.get("start station latitude", "")] = "lat"
        col_map[cols_lower.get("start station longitude", "")] = "lon"

    col_map = {k: v for k, v in col_map.items() if k}
    df = df.rename(columns=col_map)

    needed = ["start_time", "station_id", "lat", "lon"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")

    return df[["start_time", "station_id", "station_name", "lat", "lon"]].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_trip_data(months: list[str] | None = None) -> list[Path]:
    """Download Citi Bike monthly trip files."""
    if months is None:
        months = ALL_MONTHS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    missing = []
    for month in months:
        # URL pattern: YYYYMM-citibike-tripdata.zip (no .csv in name)
        fname = f"{month}-citibike-tripdata.zip"
        url = f"{CITIBIKE_BASE_URL}/{fname}"
        dest = RAW_DIR / fname
        if not dest.exists():
            print(f"Downloading {fname} ...")
            try:
                _download_file(url, dest)
            except Exception as e:
                print(f"  Warning: failed to download {fname}: {e}")
                missing.append(fname)
                continue
        paths.append(dest)
    if missing:
        raise RuntimeError(
            "Missing required trip files: "
            + ", ".join(missing)
            + ". Re-run when the source is available or place the files in data/raw/."
        )
    return paths


def build_daily_station_panel(
    trip_paths: list[Path],
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build daily station-level departure panel from trip CSVs.

    Returns a DataFrame with columns:
        station_id, date, departures, lat, lon, station_name,
        first_seen, last_seen, station_age_days
    """
    cache_path = PROCESSED_DIR / "daily_station_panel.parquet"
    meta_path = PROCESSED_DIR / "daily_station_panel.metadata.json"
    meta = {
        "panel_version": 5,
        "n_files": len(trip_paths),
        "files": [p.name for p in trip_paths],
    }

    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        try:
            if json.loads(meta_path.read_text()) == meta:
                print("Loading cached daily station panel ...")
                return pd.read_parquet(cache_path)
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"Building daily station panel from {len(trip_paths)} trip files ...")
    frames = []
    failed_files = []
    for path in tqdm(trip_paths, desc="Reading trip files"):
        try:
            raw = _read_trip_csv(path)
            df = _normalize_columns(raw)
            frames.append(df)
        except Exception as e:
            failed_files.append(f"{path.name}: {e}")

    if failed_files:
        raise RuntimeError(
            "Failed to read required trip files: "
            + "; ".join(failed_files)
        )

    if not frames:
        raise RuntimeError("No trip data loaded.")

    trips = pd.concat(frames, ignore_index=True)
    trips["start_time"] = pd.to_datetime(trips["start_time"], errors="coerce")
    trips = trips.dropna(subset=["start_time", "station_id", "lat", "lon"])
    trips["date"] = trips["start_time"].dt.date.astype("datetime64[ns]")

    months = sorted({p.name[:6] for p in trip_paths if p.name[:6].isdigit()})
    if months:
        period_start = pd.to_datetime(f"{months[0]}01", format="%Y%m%d")
        period_end = (
            pd.to_datetime(f"{months[-1]}01", format="%Y%m%d")
            + pd.offsets.MonthEnd(0)
        )
        trips = trips[trips["date"].between(period_start, period_end)].copy()

    # Station metadata: use the most common lat/lon/name per station
    station_meta = (
        trips.groupby("station_id", as_index=False)
        .agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            station_name=("station_name", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
            first_seen=("date", "min"),
            last_seen=("date", "max"),
            observed_days=("date", "nunique"),
            total_trips=("date", "size"),
        )
    )

    # Filter station universe before zero-filling. A station needs enough
    # observed trip days, and we only model dates between its first and last
    # observed trips. Otherwise removed stations become artificial zero-demand
    # rows for the rest of the calendar.
    station_meta = station_meta[
        station_meta["observed_days"].ge(MIN_STATION_DAYS)
        & station_meta["lat"].between(40.5, 41.0)
        & station_meta["lon"].between(-74.3, -73.7)
    ].copy()

    # Daily departures per station
    daily = (
        trips.groupby(["station_id", "date"], as_index=False)
        .size()
        .rename(columns={"size": "departures"})
    )
    daily = daily[daily["station_id"].isin(station_meta["station_id"])].copy()

    # Fill missing days with zero for active stations
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    stations = station_meta["station_id"].unique()
    full_index = pd.MultiIndex.from_product(
        [stations, date_range], names=["station_id", "date"]
    )
    daily = (
        daily.set_index(["station_id", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Join metadata
    daily = daily.merge(
        station_meta[
            [
                "station_id", "lat", "lon", "station_name",
                "first_seen", "last_seen",
            ]
        ],
        on="station_id",
        how="left",
    )

    # Only keep days while the station is observed in the trip feed.
    daily = daily[
        daily["date"].between(daily["first_seen"], daily["last_seen"])
    ].copy()

    # Station age
    daily["station_age_days"] = (daily["date"] - daily["first_seen"]).dt.days

    daily = daily.sort_values(["station_id", "date"]).reset_index(drop=True)

    print(f"Daily panel: {len(daily):,} rows, "
          f"{daily['station_id'].nunique():,} stations, "
          f"{daily['date'].nunique():,} days.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    return daily
