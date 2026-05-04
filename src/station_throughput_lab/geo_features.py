"""
Geospatial feature engineering for station-level throughput prediction.

Builds spatial clusters via DBSCAN on station coordinates, computes
k-nearest-neighbor throughput features, and derives location-based
attributes (distance to city center, station density, borough proxy).

No external GIS dependencies — uses scikit-learn's haversine metric
and numpy for all spatial computations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from station_throughput_lab.config import (
    DBSCAN_EPS_KM,
    DBSCAN_MIN_SAMPLES,
    KNN_NEIGHBORS,
)

# NYC reference points
NYC_CENTER = (40.7580, -73.9855)  # Times Square
MANHATTAN_CENTER = (40.7831, -73.9712)
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_station_geo_features(station_meta: pd.DataFrame) -> pd.DataFrame:
    """Compute spatial features for each station.

    Input: DataFrame with station_id, lat, lon (one row per station).
    Output: Same DataFrame with added columns:
        - spatial_cluster: DBSCAN cluster label
        - dist_to_center_km: distance to Times Square
        - dist_to_manhattan_km: distance to Manhattan center
        - station_density_1km: count of other stations within 1 km
        - knn_avg_dist_km: average distance to k nearest neighbors
        - borough_proxy: rough borough assignment based on lat/lon
    """
    df = station_meta.copy()
    coords_rad = np.radians(df[["lat", "lon"]].values)

    # --- DBSCAN spatial clustering ---
    eps_rad = DBSCAN_EPS_KM / EARTH_RADIUS_KM
    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
    ).fit(coords_rad)
    df["spatial_cluster"] = clustering.labels_

    # Remap noise points (-1) to their own singleton clusters
    noise_mask = df["spatial_cluster"] == -1
    if noise_mask.any():
        max_cluster = df["spatial_cluster"].max()
        df.loc[noise_mask, "spatial_cluster"] = range(
            max_cluster + 1, max_cluster + 1 + noise_mask.sum()
        )

    # --- Distance to reference points ---
    df["dist_to_center_km"] = df.apply(
        lambda r: _haversine_km(r["lat"], r["lon"], *NYC_CENTER), axis=1
    )
    df["dist_to_manhattan_km"] = df.apply(
        lambda r: _haversine_km(r["lat"], r["lon"], *MANHATTAN_CENTER), axis=1
    )

    # --- Station density within 1 km ---
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = 1.0 / EARTH_RADIUS_KM
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
    df["station_density_1km"] = counts - 1  # exclude self

    # --- KNN average distance ---
    k = min(KNN_NEIGHBORS + 1, len(df))
    distances, indices = tree.query(coords_rad, k=k)
    # distances are in radians; convert to km, skip self (index 0)
    knn_dists_km = distances[:, 1:] * EARTH_RADIUS_KM
    df["knn_avg_dist_km"] = knn_dists_km.mean(axis=1)

    # --- Borough proxy from lat/lon ---
    # Conditions are ordered most-specific-first to avoid overlap.
    # Manhattan is a narrow island; Queens is east of ~-73.94;
    # Brooklyn is south of ~40.7 and west of Queens; Bronx is north.
    def _borough_proxy(lat: float, lon: float) -> str:
        if lat > 40.8 and lon > -73.94:
            return "Bronx"
        if lat > 40.7 and lon > -74.02 and lon < -73.94:
            return "Manhattan"
        if lat > 40.7 and lon >= -73.94:
            return "Queens"
        if lat <= 40.7 and lat > 40.57 and lon > -74.05 and lon < -73.83:
            return "Brooklyn"
        return "Other"

    df["borough_proxy"] = df.apply(
        lambda r: _borough_proxy(r["lat"], r["lon"]), axis=1
    )

    return df
