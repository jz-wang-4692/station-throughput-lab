"""
Post-model optimization: calibration-based bias correction.

The ML model produces point predictions. This module applies a
calibration layer that corrects systematic bias using the held-out
calibration set — the same pattern used in demand planning to
separate "learning the signal" (model) from "correcting the level"
(calibration).

The approach:
1. Compute residuals on the calibration set
2. Estimate multiplicative bias factors by segment
   (borough × DOW, cluster × DOW, station-level)
3. Apply hierarchical correction: station → cluster → borough → global
4. Clip corrections to avoid extreme adjustments

This is a lightweight, non-parametric correction that doesn't require
retraining. It's the standard "forecast reconciliation" step in
operational demand planning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    """Result of calibration-based bias correction."""
    corrected_panel: pd.DataFrame
    correction_factors: pd.DataFrame
    calibration_diagnostics: dict


def _nonzero_mape(df: pd.DataFrame) -> float:
    mask = df["departures"] > 0
    if not mask.any():
        return np.nan
    pct_error = (
        (df.loc[mask, "departures"] - df.loc[mask, "predicted"]).abs()
        / df.loc[mask, "departures"]
    )
    return float(pct_error.mean())


def _compute_multiplicative_factors(
    cal: pd.DataFrame,
    group_cols: list[str],
    min_rows: int = 30,
) -> pd.DataFrame:
    """Compute multiplicative correction factors from calibration residuals.

    factor = mean(actual) / mean(predicted) for each group.
    Groups with fewer than min_rows observations are excluded.
    """
    grouped = cal.groupby(group_cols, as_index=False).agg(
        actual_mean=("departures", "mean"),
        pred_mean=("predicted", "mean"),
        n_rows=("departures", "size"),
    )
    grouped = grouped[grouped["n_rows"] >= min_rows].copy()
    grouped["factor"] = (
        grouped["actual_mean"] / grouped["pred_mean"].replace(0, np.nan)
    ).clip(0.2, 5.0).fillna(1.0)
    return grouped


def compute_calibration_corrections(
    scored_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute hierarchical correction factors from the calibration set.

    Returns (station_factors, cluster_dow_factors, borough_dow_factors, global_factor).
    """
    cal = scored_panel[scored_panel["split"] == "calibration"].copy()
    if cal.empty:
        raise ValueError("No calibration data available.")

    # Level 1: Station-level (most specific, needs enough data)
    station_factors = _compute_multiplicative_factors(
        cal, ["station_id"], min_rows=14,  # ~2 weeks of data
    )

    # Level 2: Cluster × DOW
    cluster_dow_factors = _compute_multiplicative_factors(
        cal, ["spatial_cluster", "dow"], min_rows=30,
    )

    # Level 3: Borough × DOW
    borough_dow_factors = _compute_multiplicative_factors(
        cal, ["borough_proxy", "dow"], min_rows=50,
    )

    # Level 4: Global
    global_actual = cal["departures"].mean()
    global_pred = cal["predicted"].mean()
    global_factor = pd.DataFrame([{
        "factor": np.clip(global_actual / max(global_pred, 0.01), 0.2, 5.0),
        "actual_mean": global_actual,
        "pred_mean": global_pred,
        "n_rows": len(cal),
    }])

    return station_factors, cluster_dow_factors, borough_dow_factors, global_factor


def apply_calibration_corrections(
    scored_panel: pd.DataFrame,
    station_factors: pd.DataFrame,
    cluster_dow_factors: pd.DataFrame,
    borough_dow_factors: pd.DataFrame,
    global_factor: pd.DataFrame,
) -> pd.DataFrame:
    """Apply hierarchical bias correction to predictions.

    For each row, the correction factor is chosen from the most
    specific level available:
    1. Station-level factor (if the station has enough calibration data)
    2. Cluster × DOW factor (spatial + temporal pattern)
    3. Borough × DOW factor (broader spatial + temporal)
    4. Global factor (last resort)

    The corrected prediction = raw_prediction × correction_factor.
    """
    df = scored_panel.copy()
    df["correction_factor"] = np.nan

    # Level 4: Global (baseline for everything)
    gf = float(global_factor["factor"].iloc[0])
    df["correction_factor"] = gf

    # Level 3: Borough × DOW (overrides global where available)
    if not borough_dow_factors.empty:
        borough_map = borough_dow_factors.set_index(
            ["borough_proxy", "dow"]
        )["factor"].to_dict()
        borough_keys = list(zip(df["borough_proxy"], df["dow"]))
        borough_corrections = pd.Series(
            [borough_map.get(k, np.nan) for k in borough_keys],
            index=df.index,
        )
        mask = borough_corrections.notna()
        df.loc[mask, "correction_factor"] = borough_corrections[mask]

    # Level 2: Cluster × DOW (overrides borough where available)
    if not cluster_dow_factors.empty:
        cluster_map = cluster_dow_factors.set_index(
            ["spatial_cluster", "dow"]
        )["factor"].to_dict()
        cluster_keys = list(zip(df["spatial_cluster"], df["dow"]))
        cluster_corrections = pd.Series(
            [cluster_map.get(k, np.nan) for k in cluster_keys],
            index=df.index,
        )
        mask = cluster_corrections.notna()
        df.loc[mask, "correction_factor"] = cluster_corrections[mask]

    # Level 1: Station-level (most specific, overrides everything)
    if not station_factors.empty:
        station_map = station_factors.set_index("station_id")["factor"].to_dict()
        station_corrections = df["station_id"].map(station_map)
        mask = station_corrections.notna()
        df.loc[mask, "correction_factor"] = station_corrections[mask]

    # Apply correction
    df["predicted_raw"] = df["predicted"]
    df["predicted"] = (df["predicted_raw"] * df["correction_factor"]).clip(lower=0)
    df["residual"] = df["departures"] - df["predicted"]
    df["abs_error"] = df["residual"].abs()
    df["pct_error"] = (
        df["abs_error"] / df["departures"].replace(0, np.nan)
    ).clip(upper=10)

    return df


def run_calibration_optimization(
    scored_panel: pd.DataFrame,
) -> CalibrationResult:
    """Full calibration pipeline: compute factors → apply → diagnose."""
    station_f, cluster_f, borough_f, global_f = compute_calibration_corrections(
        scored_panel,
    )

    corrected = apply_calibration_corrections(
        scored_panel, station_f, cluster_f, borough_f, global_f,
    )

    # Diagnostics: compare before/after on test set
    test_before = scored_panel[scored_panel["split"] == "test"]
    test_after = corrected[corrected["split"] == "test"]

    diagnostics = {
        "before": {
            "mae": float(test_before["abs_error"].mean()),
            "nonzero_mape": _nonzero_mape(test_before),
            "bias": float(
                (test_before["predicted"].sum() - test_before["departures"].sum())
                / max(test_before["departures"].sum(), 1)
            ),
            "wape": float(
                test_before["abs_error"].sum()
                / max(test_before["departures"].abs().sum(), 1)
            ),
        },
        "after": {
            "mae": float(test_after["abs_error"].mean()),
            "nonzero_mape": _nonzero_mape(test_after),
            "bias": float(
                (test_after["predicted"].sum() - test_after["departures"].sum())
                / max(test_after["departures"].sum(), 1)
            ),
            "wape": float(
                test_after["abs_error"].sum()
                / max(test_after["departures"].abs().sum(), 1)
            ),
        },
        "global_factor": float(global_f["factor"].iloc[0]),
        "n_station_factors": len(station_f),
        "n_cluster_dow_factors": len(cluster_f),
        "n_borough_dow_factors": len(borough_f),
    }

    # Build correction factors summary
    factors_summary = []
    for _, row in station_f.iterrows():
        factors_summary.append({
            "level": "station",
            "key": str(row["station_id"]),
            "factor": row["factor"],
            "n_rows": row["n_rows"],
        })
    for _, row in cluster_f.iterrows():
        factors_summary.append({
            "level": "cluster_dow",
            "key": f"cluster={row['spatial_cluster']},dow={int(row['dow'])}",
            "factor": row["factor"],
            "n_rows": row["n_rows"],
        })
    for _, row in borough_f.iterrows():
        factors_summary.append({
            "level": "borough_dow",
            "key": f"{row['borough_proxy']},dow={int(row['dow'])}",
            "factor": row["factor"],
            "n_rows": row["n_rows"],
        })
    factors_summary.append({
        "level": "global",
        "key": "all",
        "factor": float(global_f["factor"].iloc[0]),
        "n_rows": int(global_f["n_rows"].iloc[0]),
    })

    print(f"\nCalibration optimization:")
    print(f"  Global correction factor: {diagnostics['global_factor']:.3f}")
    print(f"  Station-level factors: {diagnostics['n_station_factors']}")
    print(f"  Cluster×DOW factors: {diagnostics['n_cluster_dow_factors']}")
    print(f"  Borough×DOW factors: {diagnostics['n_borough_dow_factors']}")
    print(f"\n  Test MAE:  {diagnostics['before']['mae']:.2f} → {diagnostics['after']['mae']:.2f}")
    print(f"  Test WAPE: {diagnostics['before']['wape']:.3f} → {diagnostics['after']['wape']:.3f}")
    print(
        "  Test nonzero MAPE: "
        f"{diagnostics['before']['nonzero_mape']:.3f} → "
        f"{diagnostics['after']['nonzero_mape']:.3f}"
    )
    print(f"  Test Bias: {diagnostics['before']['bias']:.3f} → {diagnostics['after']['bias']:.3f}")

    return CalibrationResult(
        corrected_panel=corrected,
        correction_factors=pd.DataFrame(factors_summary),
        calibration_diagnostics=diagnostics,
    )
