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


def bias(actual: pd.Series, predicted: pd.Series) -> float:
    denom = actual.sum()
    if denom == 0:
        return np.nan
    return float((predicted - actual).sum() / denom)


def _metrics_for_group(group: pd.DataFrame) -> dict:
    """Compute standard metrics for a group of rows."""
    p = group["predicted"]
    return _metrics_for_predictions(group, p)


def _metrics_for_predictions(group: pd.DataFrame, predicted: pd.Series) -> dict:
    """Compute standard metrics for a group and arbitrary prediction series."""
    a = group["departures"]
    p = predicted.clip(lower=0)
    return {
        "rows": len(group),
        "stations": group["station_id"].nunique(),
        "avg_actual": float(a.mean()),
        "avg_predicted": float(p.mean()),
        "mae": mae(a, p),
        "median_ae": median_ae(a, p),
        "rmse": rmse(a, p),
        "wape": wape(a, p),
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
            "bias": 0,
        }
    return _metrics_for_group(test)


def evaluate_baseline_comparison(scored_panel: pd.DataFrame) -> pd.DataFrame:
    """Compare the model against simple operational baselines on the test set."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()

    methods: list[tuple[str, str, str]] = [
        ("Yesterday", "baseline", "dep_lag_1"),
        ("Same weekday last week", "baseline", "dep_lag_7"),
        ("Rolling 7-day mean", "baseline", "rolling_mean_7"),
        ("Rolling 28-day mean", "baseline", "rolling_mean_28"),
        ("Same-DOW rolling mean", "baseline", "dow_rolling_mean_4"),
    ]
    if "predicted_raw" in test.columns:
        methods.append(("Raw ML model", "model", "predicted_raw"))
    methods.append(("Calibrated ML model", "model", "predicted"))

    rows = []
    for method, kind, col in methods:
        if col not in test.columns:
            continue
        valid = test[test[col].notna()].copy()
        if valid.empty:
            continue
        metrics = _metrics_for_predictions(valid, valid[col])
        metrics["method"] = method
        metrics["kind"] = kind
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    baseline = out[out["kind"] == "baseline"]
    best_wape = baseline["wape"].min() if not baseline.empty else np.nan
    if pd.notna(best_wape) and best_wape > 0:
        out["wape_skill_vs_best_baseline"] = (1 - out["wape"] / best_wape) * 100
    else:
        out["wape_skill_vs_best_baseline"] = np.nan
    return out[
        [
            "method", "kind", "rows", "stations", "avg_actual", "avg_predicted",
            "mae", "median_ae", "rmse", "wape", "bias",
            "wape_skill_vs_best_baseline",
        ]
    ]


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


def evaluate_by_station_volume(scored_panel: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by train-period station volume tier."""
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if test.empty:
        test = scored_panel[scored_panel["split"] == "calibration"].copy()
    if test.empty:
        return pd.DataFrame()

    train_avg = (
        scored_panel[scored_panel["split"] == "train"]
        .groupby("station_id")["departures"]
        .mean()
    )
    test["train_avg_departures"] = test["station_id"].map(train_avg)
    known = test.dropna(subset=["train_avg_departures"]).copy()
    rows = []

    if not known.empty:
        station_volume = (
            known[["station_id", "train_avg_departures"]]
            .drop_duplicates("station_id")
        )
        q33, q67 = station_volume["train_avg_departures"].quantile([1 / 3, 2 / 3])

        def _volume_tier(value: float) -> str:
            if value <= q33:
                return "low_volume"
            if value <= q67:
                return "medium_volume"
            return "high_volume"

        known["volume_segment"] = known["train_avg_departures"].map(_volume_tier)
        for segment, group in known.groupby("volume_segment"):
            metrics = _metrics_for_group(group)
            metrics["volume_segment"] = segment
            metrics["avg_train_departures"] = float(group["train_avg_departures"].mean())
            rows.append(metrics)

    no_train = test[test["train_avg_departures"].isna()].copy()
    if not no_train.empty:
        metrics = _metrics_for_group(no_train)
        metrics["volume_segment"] = "no_train_history"
        metrics["avg_train_departures"] = np.nan
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    order = {
        "low_volume": 0,
        "medium_volume": 1,
        "high_volume": 2,
        "no_train_history": 3,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["volume_segment"].map(order).fillna(99)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


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
        "baseline_comparison": evaluate_baseline_comparison(scored_panel),
        "cold_vs_mature": evaluate_cold_vs_mature(scored_panel),
        "by_station_volume": evaluate_by_station_volume(scored_panel),
        "by_cluster": evaluate_by_cluster(scored_panel),
        "by_borough": evaluate_by_borough(scored_panel),
        "by_dow": evaluate_by_dow(scored_panel),
        "by_weather": evaluate_by_weather(scored_panel),
    }
