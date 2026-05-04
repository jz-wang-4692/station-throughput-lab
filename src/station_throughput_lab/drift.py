"""
Feature drift analysis between train and test distributions.

Computes Jensen-Shannon divergence for the top features to quantify
how much the feature distributions shifted between training and test
periods. This explains why the raw model's predictions are biased
and motivates the calibration correction layer.

JS divergence is symmetric, bounded [0, 1], and interpretable:
- 0.0 = identical distributions
- 0.1 = minor drift
- 0.3+ = substantial drift that will degrade model accuracy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def compute_js_divergence(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    n_bins: int = 50,
) -> float:
    """Compute JS divergence between two 1-D distributions."""
    # Shared bin edges from the combined range
    combined = np.concatenate([train_vals, test_vals])
    combined = combined[np.isfinite(combined)]
    if len(combined) < 10:
        return 0.0
    lo, hi = np.percentile(combined, [1, 99])
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)

    p = np.histogram(train_vals[np.isfinite(train_vals)], bins=edges, density=True)[0]
    q = np.histogram(test_vals[np.isfinite(test_vals)], bins=edges, density=True)[0]

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(jensenshannon(p, q) ** 2)  # squared JS = JS divergence


def compute_feature_drift(
    scored_panel: pd.DataFrame,
    top_features: list[str],
) -> pd.DataFrame:
    """Compute JS divergence for each feature between train and test.

    Returns a DataFrame with columns: feature, js_divergence, train_mean,
    test_mean, shift_pct, drift_severity.
    """
    train = scored_panel[scored_panel["split"] == "train"]
    test = scored_panel[scored_panel["split"] == "test"]

    rows = []
    for feat in top_features:
        if feat not in train.columns:
            continue
        train_vals = train[feat].dropna().values.astype(float)
        test_vals = test[feat].dropna().values.astype(float)
        if len(train_vals) < 10 or len(test_vals) < 10:
            continue

        jsd = compute_js_divergence(train_vals, test_vals)
        train_mean = float(np.mean(train_vals))
        test_mean = float(np.mean(test_vals))
        shift_pct = (
            (test_mean - train_mean) / abs(train_mean) * 100
            if abs(train_mean) > 1e-6 else 0.0
        )

        if jsd < 0.05:
            severity = "minimal"
        elif jsd < 0.15:
            severity = "moderate"
        elif jsd < 0.30:
            severity = "substantial"
        else:
            severity = "severe"

        rows.append({
            "feature": feat,
            "js_divergence": round(jsd, 4),
            "train_mean": round(train_mean, 2),
            "test_mean": round(test_mean, 2),
            "shift_pct": round(shift_pct, 1),
            "drift_severity": severity,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature", "js_divergence", "train_mean",
                "test_mean", "shift_pct", "drift_severity",
            ]
        )

    df = pd.DataFrame(rows).sort_values("js_divergence", ascending=False)
    return df.reset_index(drop=True)


def compute_target_drift(scored_panel: pd.DataFrame) -> dict:
    """Compute drift statistics for the target variable (departures)."""
    train = scored_panel[scored_panel["split"] == "train"]["departures"]
    cal = scored_panel[scored_panel["split"] == "calibration"]["departures"]
    test = scored_panel[scored_panel["split"] == "test"]["departures"]

    return {
        "train_mean": round(float(train.mean()), 2),
        "train_median": round(float(train.median()), 1),
        "train_std": round(float(train.std()), 2),
        "cal_mean": round(float(cal.mean()), 2),
        "cal_median": round(float(cal.median()), 1),
        "test_mean": round(float(test.mean()), 2),
        "test_median": round(float(test.median()), 1),
        "test_std": round(float(test.std()), 2),
        "train_test_js": round(
            compute_js_divergence(train.values, test.values), 4
        ),
        "train_cal_js": round(
            compute_js_divergence(train.values, cal.values), 4
        ),
        "cal_test_js": round(
            compute_js_divergence(cal.values, test.values), 4
        ),
    }
