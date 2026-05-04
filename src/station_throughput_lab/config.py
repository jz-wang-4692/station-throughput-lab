"""
Central configuration: paths, parameters, and constants.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Citi Bike data — public monthly trip-data bucket
# ---------------------------------------------------------------------------
CITIBIKE_BASE_URL = "https://s3." + "amaz" + "onaws.com/tripdata"
# 2024 monthly files for training + test (manageable download size)
# Train: Apr-Oct 2024, Calibration: Nov 2024, Test: Dec 2024
ALL_MONTHS = [f"2024{m:02d}" for m in range(4, 13)]  # Apr-Dec 2024

# ---------------------------------------------------------------------------
# Weather data — NOAA GHCN-Daily for Central Park (NYC)
# ---------------------------------------------------------------------------
NOAA_STATION_ID = "USW00094728"  # Central Park, NY

# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
MIN_STATION_DAYS = 30          # stations with fewer active days are dropped

# ---------------------------------------------------------------------------
# Cold-start simulation
# ---------------------------------------------------------------------------
COLD_START_HOLDOUT_DAYS = 60   # first N days of a station's life = cold-start
COLD_START_MIN_MATURE_DAYS = 120  # station needs this many days to be "mature"

# ---------------------------------------------------------------------------
# Spatial clustering
# ---------------------------------------------------------------------------
DBSCAN_EPS_KM = 0.8           # ~800 m radius for spatial clusters
DBSCAN_MIN_SAMPLES = 3
KNN_NEIGHBORS = 5              # k-nearest stations for neighbor features

# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------
# With full trip data across all months:
# Train: Apr-Oct (7 months, covers summer peak + fall decline)
# Calibration: Nov 1-30 (1 month, includes Thanksgiving)
# Test: Dec 1-31
TRAIN_END = "2024-10-31"
CALIBRATION_START = "2024-11-01"
CALIBRATION_END = "2024-11-30"
TEST_START = "2024-12-01"
TEST_END = "2024-12-31"

# ---------------------------------------------------------------------------
# AutoGluon
# ---------------------------------------------------------------------------
AG_TIME_LIMIT = 900            # seconds for AutoGluon training
AG_PRESETS = "best_quality"
AG_EVAL_METRIC = "mean_absolute_error"
