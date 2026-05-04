"""
Evaluation metrics and diagnostic breakdowns.

Computes accuracy metrics sliced by:
- Overall test set
- Cold-start vs mature stations
- Spatial cluster
- Borough
- Day-of-week
- Weather conditions

The key question: how much does hierarchical cold-start imputation
close the accuracy gap between new and established stations?
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().mean())


def median_ae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().median())


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(((actual - predicted) ** 2).mean()))


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    denom = actual.abs().sum()
    if denom == 0:
        return np.nan
    return float((actual - predicted).abs().sum() / denom)


def nonzero_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error on rows with nonzero actuals.

    Standard MAPE is undefined when actual demand is zero. Daily station
    throughput has real zero-departure rows, so this metric is reported as a
    diagnostic rather than the primary percentage-error metric.
    """
    mask = actual > 0
    if not mask.any():
        return np.nan
    return float(((actual[mask] - predicted[mask]).abs() / actual[mask]).mean())


def nonzero_mape_coverage(actual: pd.Series) -> float:
    if len(actual) == 0:
        return np.nan
    return float((actual > 0).mean())


def bias(actual: pd.Series, predicted: pd.Series) -> float:
    denom = actual.sum()
    if denom == 0:
        return np.nan
    return float((predicted - actual).sum() / denom)


def _metrics_for_group(group: pd.DataFrame) -> dict:
    """Compute standard metrics for a group of rows."""
    a = group["departures"]
    p = group["predicted"]
    return {
        "rows": len(group),
        "stations": group["station_id"].nunique(),
        "avg_actual": float(a.mean()),
        "avg_predicted": float(p.mean()),
        "mae": mae(a, p),
        "median_ae": median_ae(a, p),
        "rmse": rmse(a, p),
        "wape": wape(a, p),
        "nonzero_mape": nonzero_mape(a, p),
        "nonzero_mape_coverage": nonzero_mape_coverage(a),
        "bias": bias(a, p),
    }


def evaluate_test_set(scored_panel: pd.DataFrame) -> dict:
    """Compute overall test-set metrics."""
    test = scored_panel[scored_panel["split"] == "test"]
    if test.empty:
        # Fall back to calibration if a caller supplies a shorter custom split.
        test = scored_panel[scored_panel["split"] == "calibration"]
    if test.empty:
        return {
            "rows": 0,
            "stations": 0,
            "mae": 0,
            "median_ae": 0,
            "rmse": 0,
            "wape": 0,
            "nonzero_mape": 0,
            "nonzero_mape_coverage": 0,
            "bias": 0,
        }
    return _metrics_for_group(test)


def evaluate_cold_vs_mature(scored_panel: pd.DataFrame) -> pd.DataFrame:
    """Compare accuracy between cold-start and mature stations (test set only)."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        # Fall back to calibration if a caller supplies a shorter custom split.
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for label, mask in [
        ("cold_start", test["is_new_station"] == 1),
        ("mature", test["is_new_station"] == 0),
    ]:
        group = test[mask]
        if group.empty:
            continue
        metrics = _metrics_for_group(group)
        metrics["segment"] = label
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_by_cluster(scored_panel: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Accuracy by spatial cluster (top N by row count)."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for cluster, group in test.groupby("spatial_cluster"):
        metrics = _metrics_for_group(group)
        metrics["spatial_cluster"] = cluster
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("rows", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def evaluate_by_borough(scored_panel: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by borough proxy."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for borough, group in test.groupby("borough_proxy"):
        metrics = _metrics_for_group(group)
        metrics["borough"] = borough
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def evaluate_by_dow(scored_panel: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by day of week."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for dow, group in test.groupby("dow"):
        metrics = _metrics_for_group(group)
        metrics["dow"] = dow
        metrics["dow_name"] = dow_names[int(dow)]
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("dow").reset_index(drop=True)


def evaluate_by_weather(scored_panel: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by weather condition."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for label, mask in [
        ("cold (<35°F)", test["is_cold"] == 1),
        ("mild (35-85°F)", (test["is_cold"] == 0) & (test["is_hot"] == 0)),
        ("hot (>85°F)", test["is_hot"] == 1),
        ("rainy", test["is_rainy"] == 1),
        ("dry", test["is_rainy"] == 0),
    ]:
        group = test[mask]
        if group.empty:
            continue
        metrics = _metrics_for_group(group)
        metrics["condition"] = label
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_evaluation_summary(scored_panel: pd.DataFrame) -> dict:
    """Build all evaluation tables."""
    return {
        "overall": evaluate_test_set(scored_panel),
        "cold_vs_mature": evaluate_cold_vs_mature(scored_panel),
        "by_cluster": evaluate_by_cluster(scored_panel),
        "by_borough": evaluate_by_borough(scored_panel),
        "by_dow": evaluate_by_dow(scored_panel),
        "by_weather": evaluate_by_weather(scored_panel),
    }
