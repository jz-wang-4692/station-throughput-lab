"""
AutoGluon-based throughput modeling with temporal train/calibration/test splits.

Uses AutoGluon TabularPredictor with best_quality presets, letting the
framework handle model selection, ensembling, and hyperparameter tuning.
The intellectual work is in the features, not the model tuning.

The pipeline:
1. Assign temporal splits (train / calibration / test)
2. Define feature columns (exclude leaky columns)
3. Train AutoGluon on the train split
4. Score all splits
5. Compute per-station, per-cluster, and cold-start-vs-mature metrics
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from station_throughput_lab.config import (
    AG_EVAL_METRIC,
    AG_PRESETS,
    AG_TIME_LIMIT,
    CALIBRATION_START,
    MODELS_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
)
from station_throughput_lab.evaluation import nonzero_mape, nonzero_mape_coverage


# Columns to exclude from features (target, identifiers, leaky)
EXCLUDE_COLS = {
    "departures",          # target
    "station_id",          # identifier
    "station_name",        # identifier
    "date",                # will use derived features instead
    "first_seen",          # metadata
    "last_seen",           # metadata
    "observed_days",       # full-period station lifecycle metadata
    "split",               # split label
    "lat",                 # raw coords (use derived geo features)
    "lon",                 # raw coords
    "is_imputed",          # diagnostic only
    "predicted",           # post-model output
    "predicted_raw",       # post-calibration diagnostic
    "residual",            # post-model diagnostic
    "abs_error",           # post-model diagnostic
    "pct_error",           # post-model diagnostic
    "correction_factor",   # post-calibration diagnostic
}


@dataclass
class ModelingResult:
    scored_panel: pd.DataFrame
    leaderboard: pd.DataFrame
    feature_importance: pd.DataFrame
    split_summary: pd.DataFrame
    predictor_path: Path


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign temporal train/calibration/test splits."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["split"] = "train"
    out.loc[out["date"] >= CALIBRATION_START, "split"] = "calibration"
    out.loc[out["date"] >= TEST_START, "split"] = "test"
    out.loc[out["date"] > TEST_END, "split"] = "future"  # drop these
    out = out[out["split"] != "future"].copy()
    return out


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select feature columns: everything except excluded columns."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train_model(
    feature_panel: pd.DataFrame,
    time_limit: int = AG_TIME_LIMIT,
) -> ModelingResult:
    """Train AutoGluon model and score all splits."""
    from autogluon.tabular import TabularPredictor

    df = assign_splits(feature_panel)
    feature_cols = _get_feature_columns(df)
    label = "departures"

    # Train split only
    train = df[df["split"] == "train"].copy()
    train_data = train[feature_cols + [label]]

    print(f"Training AutoGluon on {len(train):,} rows, "
          f"{len(feature_cols)} features, time_limit={time_limit}s ...")

    predictor_path = MODELS_DIR / "ag_throughput"
    if predictor_path.exists():
        shutil.rmtree(predictor_path)

    predictor = TabularPredictor(
        label=label,
        path=str(predictor_path),
        eval_metric=AG_EVAL_METRIC,
        problem_type="regression",
    )
    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=AG_PRESETS,
        dynamic_stacking=False,
        verbosity=1,
    )

    # Score all data
    score_data = df[feature_cols].copy()
    df["predicted"] = predictor.predict(score_data)
    df["predicted"] = df["predicted"].clip(lower=0)
    df["residual"] = df["departures"] - df["predicted"]
    df["abs_error"] = df["residual"].abs()
    df["pct_error"] = (
        df["abs_error"] / df["departures"].replace(0, np.nan)
    ).clip(upper=10)

    # Leaderboard
    try:
        leaderboard = predictor.leaderboard(silent=True)
    except Exception:
        leaderboard = pd.DataFrame()

    # Feature importance: use the held-out calibration split so the reported
    # importances are not measured on the same rows used for fitting.
    try:
        importance_data = df.loc[
            df["split"] == "calibration",
            feature_cols + [label],
        ]
        if importance_data.empty:
            importance_data = train_data[feature_cols + [label]]
        if len(importance_data) > 50_000:
            importance_data = importance_data.sample(50_000, random_state=42)
        importance = predictor.feature_importance(
            importance_data,
            num_shuffle_sets=2,
            silent=True,
        )
        importance = importance.reset_index().rename(
            columns={"index": "feature"}
        )
    except Exception:
        importance = pd.DataFrame(
            columns=["feature", "importance", "stddev", "p_value"]
        )

    # Split summary
    split_summary = (
        df.groupby("split", as_index=False)
        .agg(
            rows=("departures", "size"),
            stations=("station_id", "nunique"),
            days=("date", "nunique"),
            avg_departures=("departures", "mean"),
            avg_predicted=("predicted", "mean"),
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            new_station_share=("is_new_station", "mean"),
        )
    )
    split_metric_rows = []
    for split, group in df.groupby("split"):
        split_metric_rows.append({
            "split": split,
            "nonzero_mape": nonzero_mape(group["departures"], group["predicted"]),
            "nonzero_mape_coverage": nonzero_mape_coverage(group["departures"]),
        })
    split_summary = split_summary.merge(
        pd.DataFrame(split_metric_rows),
        on="split",
        how="left",
    )

    print("\nSplit summary:")
    print(split_summary.to_string(index=False))

    return ModelingResult(
        scored_panel=df,
        leaderboard=leaderboard,
        feature_importance=importance,
        split_summary=split_summary,
        predictor_path=predictor_path,
    )
