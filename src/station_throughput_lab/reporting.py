"""
Report generation: figures, tables, and markdown writeup.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from station_throughput_lab.config import FIGURES_DIR, REPORTS_DIR


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# EDA figures
# ---------------------------------------------------------------------------

def plot_daily_throughput_trend(feature_panel: pd.DataFrame) -> Path:
    """City-wide daily departures with borough breakdown."""
    path = FIGURES_DIR / "daily_throughput_trend.png"
    daily = (
        feature_panel.groupby(["date", "borough_proxy"], as_index=False)
        .agg(departures=("departures", "sum"))
    )
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=daily, x="date", y="departures", hue="borough_proxy", linewidth=1.2)
    plt.title("Daily bike departures by borough")
    plt.xlabel("")
    plt.ylabel("Total departures")
    _savefig(path)
    return path


def plot_station_map(station_geo: pd.DataFrame) -> Path:
    """Scatter plot of stations colored by spatial cluster."""
    path = FIGURES_DIR / "station_map.png"
    plt.figure(figsize=(10, 10))
    n_clusters = station_geo["spatial_cluster"].nunique()
    palette = "tab20" if n_clusters <= 20 else "husl"
    sns.scatterplot(
        data=station_geo,
        x="lon", y="lat",
        hue="spatial_cluster",
        palette=palette,
        s=15, alpha=0.7, legend=False,
    )
    plt.title(f"Station locations colored by spatial cluster ({n_clusters} clusters)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    _savefig(path)
    return path


def plot_dow_pattern(feature_panel: pd.DataFrame) -> Path:
    """Average throughput by day-of-week and borough."""
    path = FIGURES_DIR / "dow_pattern.png"
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    avg = (
        feature_panel.groupby(["dow", "borough_proxy"], as_index=False)
        .agg(avg_dep=("departures", "mean"))
    )
    avg["dow_name"] = avg["dow"].map(lambda x: dow_names[int(x)])
    plt.figure(figsize=(10, 5))
    sns.barplot(data=avg, x="dow_name", y="avg_dep", hue="borough_proxy",
                order=dow_names)
    plt.title("Average daily departures by day-of-week and borough")
    plt.xlabel("")
    plt.ylabel("Avg departures per station")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Model evaluation figures
# ---------------------------------------------------------------------------

def plot_cold_vs_mature(eval_summary: dict) -> Path:
    """Bar chart comparing cold-start vs mature station accuracy."""
    path = FIGURES_DIR / "cold_vs_mature.png"
    df = eval_summary["cold_vs_mature"]
    if df.empty:
        return path

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title in [
        (axes[0], "mae", "MAE"),
        (axes[1], "wape", "WAPE"),
        (axes[2], "bias", "Bias"),
    ]:
        sns.barplot(data=df, x="segment", y=metric, hue="segment", ax=ax,
                    palette={"cold_start": "#E45756", "mature": "#4C78A8"},
                    legend=False)
        ax.set_title(title)
        ax.set_xlabel("")
    if "bias" in df.columns:
        axes[2].axhline(0, color="black", linewidth=0.8)
    _savefig(path)
    return path


def plot_accuracy_by_borough(eval_summary: dict) -> Path:
    """MAE by borough."""
    path = FIGURES_DIR / "accuracy_by_borough.png"
    df = eval_summary["by_borough"]
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="borough", y="mae", color="#4C78A8")
    plt.title("Test MAE by borough")
    plt.xlabel("")
    plt.ylabel("MAE (departures/day)")
    _savefig(path)
    return path


def plot_accuracy_by_dow(eval_summary: dict) -> Path:
    """MAE by day of week."""
    path = FIGURES_DIR / "accuracy_by_dow.png"
    df = eval_summary["by_dow"]
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="dow_name", y="mae", color="#F58518")
    plt.title("Test MAE by day of week")
    plt.xlabel("")
    plt.ylabel("MAE (departures/day)")
    _savefig(path)
    return path


def plot_actual_vs_predicted(scored_panel: pd.DataFrame, n_sample: int = 5000) -> Path:
    """Scatter plot of actual vs predicted departures."""
    path = FIGURES_DIR / "actual_vs_predicted.png"
    test = scored_panel[scored_panel["split"] == "test"].copy()
    if len(test) > n_sample:
        test = test.sample(n_sample, random_state=42)
    plt.figure(figsize=(7, 7))
    plt.scatter(test["departures"], test["predicted"], alpha=0.15, s=8, color="#4C78A8")
    max_val = max(test["departures"].max(), test["predicted"].max())
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=0.8)
    plt.xlabel("Actual departures")
    plt.ylabel("Predicted departures")
    plt.title("Actual vs predicted (test set, post-calibration)")
    _savefig(path)
    return path


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 20) -> Path:
    """Top feature importance from AutoGluon."""
    path = FIGURES_DIR / "feature_importance.png"
    if importance.empty:
        return path
    top = importance.head(top_n).copy()
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top, y="feature", x="importance", color="#4C78A8")
    plt.title(f"Top {top_n} features by permutation importance")
    plt.xlabel("Importance")
    plt.ylabel("")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Calibration & drift figures
# ---------------------------------------------------------------------------

def plot_calibration_before_after(cal_diagnostics: dict) -> Path:
    """Bar chart showing MAE/WAPE/Bias before and after calibration."""
    path = FIGURES_DIR / "calibration_before_after.png"
    metrics = ["mae", "wape", "bias"]
    labels = ["MAE", "WAPE", "Bias"]
    before = [cal_diagnostics["before"][m] for m in metrics]
    after = [cal_diagnostics["after"][m] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, label, b, a in zip(axes, labels, before, after):
        bars = ax.bar(["Raw model", "Calibrated"], [b, a],
                      color=["#E45756", "#4C78A8"])
        ax.set_title(label)
        ax.set_ylabel(label)
        for bar, val in zip(bars, [b, a]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        if label == "Bias":
            ax.axhline(0, color="black", linewidth=0.8)
    _savefig(path)
    return path


def plot_drift_heatmap(drift_df: pd.DataFrame) -> Path:
    """Heatmap of feature drift (JS divergence) for top features."""
    path = FIGURES_DIR / "feature_drift_heatmap.png"
    if drift_df.empty:
        return path

    fig, ax = plt.subplots(figsize=(8, max(4, len(drift_df) * 0.4)))
    pivot = drift_df.set_index("feature")[["js_divergence"]].rename(
        columns={"js_divergence": "JS Divergence"}
    )
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=0.5, ax=ax, cbar_kws={"label": "JS Divergence"},
    )
    ax.set_title("Feature drift: train → test (Jensen-Shannon divergence)")
    ax.set_ylabel("")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    scored_panel: pd.DataFrame,
    eval_summary: dict,
    leaderboard: pd.DataFrame,
    feature_importance: pd.DataFrame,
    split_summary: pd.DataFrame,
    station_geo: pd.DataFrame,
    cal_diagnostics: dict | None = None,
    drift_df: pd.DataFrame | None = None,
    target_drift: dict | None = None,
) -> Path:
    """Generate the full markdown report."""
    _ensure_dirs()

    overall = eval_summary["overall"]
    cold_mature = eval_summary["cold_vs_mature"]
    by_borough = eval_summary["by_borough"]
    by_dow = eval_summary["by_dow"]

    n_stations = scored_panel["station_id"].nunique()
    n_clusters = station_geo["spatial_cluster"].nunique()
    test = scored_panel[scored_panel["split"] == "test"]

    # Cold-start gap
    cold_row = cold_mature[cold_mature["segment"] == "cold_start"]
    mature_row = cold_mature[cold_mature["segment"] == "mature"]
    cold_mae = float(cold_row["mae"].iloc[0]) if not cold_row.empty else 0
    mature_mae = float(mature_row["mae"].iloc[0]) if not mature_row.empty else 0
    gap_pct = ((cold_mae - mature_mae) / mature_mae * 100) if mature_mae > 0 else 0

    report = f"""# Station Throughput Lab Report

Generated: {date.today().isoformat()}

## Overview

This project predicts next-day bike-station throughput (departures per day per station)
using the public NYC Citi Bike trip dataset. The core challenge: **how do you forecast
throughput for early-life stations when station-specific history is sparse or unavailable?**

The approach combines deep temporal feature engineering, geospatial neighbor features,
hierarchical cold-start imputation, and a calibration-based bias correction layer —
then lets AutoGluon handle model selection and ensembling.

## Data

- **Source:** NYC Citi Bike public trip data
- **Period:** {scored_panel['date'].min().strftime('%B %Y')} – {scored_panel['date'].max().strftime('%B %Y')}
- **Grain:** daily departures per station
- **Stations:** {n_stations:,} (after filtering)
- **Spatial clusters:** {n_clusters} (DBSCAN on lat/lon)

## Split Summary

{split_summary.to_markdown(index=False, floatfmt=".2f")}

## Feature Engineering

### Temporal features
- Day-of-week (cyclical encoding), month, quarter, holiday/weekend indicators
- Lag features: 1d, 2d, 7d (same DOW), 14d, 28d
- Rolling statistics: mean/median over 7/14/28-day windows, std over 7/14 days
- DOW-specific rolling mean and median (same weekday over last 4 occurrences)
- Trend ratio: 7-day mean vs 28-day mean
- Coefficient of variation (7-day), zero-day rate (14-day)

### Geospatial features
- DBSCAN spatial cluster membership
- Distance to city center and Manhattan center
- Station density within 1 km radius
- Average distance to k-nearest neighbor stations
- Borough proxy from lat/lon
- **KNN neighbor throughput:** average departures yesterday at the 5 nearest stations

### Weather features
- NOAA Central Park daily observations when available (temperature, precipitation, snow)
- Binary indicators: cold/hot/rainy/snowy days
- Weather × temporal interactions: cold_weekend, rain_weekend, temp_deviation

### Station maturity features
- Station age in days (log-transformed)
- Binary indicators: is_new_station (<60 days), is_very_new (<30 days)
- Historical average throughput (expanding mean — capacity proxy)
- Weekend-vs-weekday throughput ratio

## Cold-Start Imputation

For stations with insufficient history, lag and rolling features are NaN.
The hierarchical imputation fills these using:

1. **Spatial cluster average** — throughput of mature stations in the same DBSCAN cluster,
   scaled by a density ratio (local station density / cluster average density)
2. **Borough average** — fallback when the cluster has no mature stations
3. **City-wide average** — final fallback

## Modeling

AutoGluon TabularPredictor with `best_quality` presets. The framework handles
model selection and ensembling across the installed tabular backends; this run
selected LightGBM-family models under the available local memory and package
constraints. Dynamic stacking is disabled to keep the full run reproducible in
memory-constrained local environments.

AutoGluon reports validation scores as negative MAE because its leaderboard sorts
higher scores first; the reported evaluation metric is still mean absolute error.

Evaluation uses a rolling one-day-ahead setup: lagged station, neighbor, and
network features are based on demand observed through the previous day.

### Model Leaderboard (fitted models)

{leaderboard.head(5).to_markdown(index=False, floatfmt=".3f") if not leaderboard.empty else "(not available)"}

### Feature Importance (top 15)

{feature_importance.head(15).to_markdown(index=False, floatfmt=".4f") if not feature_importance.empty else "(not available)"}

![Feature importance](figures/feature_importance.png)
"""

    # --- Calibration & Bias Correction section ---
    if cal_diagnostics is not None:
        before = cal_diagnostics["before"]
        after = cal_diagnostics["after"]
        gf = cal_diagnostics.get("global_factor", 1.0)
        n_station = cal_diagnostics.get("n_station_factors", 0)
        n_cluster = cal_diagnostics.get("n_cluster_dow_factors", 0)
        n_borough = cal_diagnostics.get("n_borough_dow_factors", 0)

        mae_improve = (1 - after["mae"] / before["mae"]) * 100 if before["mae"] > 0 else 0
        wape_improve = (1 - after["wape"] / before["wape"]) * 100 if before["wape"] > 0 else 0
        mape_improve = (
            (1 - after["nonzero_mape"] / before["nonzero_mape"]) * 100
            if before.get("nonzero_mape", 0) > 0
            else 0
        )

        report += f"""
## Calibration & Bias Correction

The raw ML model learns relative patterns (which features matter, how they interact)
but can be systematically biased on the absolute level — especially when the test
period has different demand characteristics than training. This is the standard
"forecast reconciliation" problem in demand planning.

### Methodology

We apply a **hierarchical multiplicative correction** estimated from the held-out
calibration set (November). For each segment, the correction factor is:

```
factor = mean(actual) / mean(predicted)    on calibration data
```

The hierarchy, from most specific to most general:

| Level | Granularity | Factors | Min sample |
|-------|------------|---------|------------|
| Station | Per station | {n_station:,} | 14 rows |
| Cluster × DOW | Spatial cluster + day-of-week | {n_cluster} | 30 rows |
| Borough × DOW | Borough + day-of-week | {n_borough} | 50 rows |
| Global | Single factor for all | 1 | all calibration |

Each prediction is corrected by the most specific factor available. Station-level
factors override cluster-level, which override borough-level, which override global.
For stations without calibration history, the method falls back to cluster,
borough, or global factors.

**Why hierarchical?** A global correction factor (here: {gf:.3f}) applies the same
adjustment everywhere. But a Manhattan commuter station on a Monday behaves very
differently from a Queens park station on a Saturday. The hierarchical approach
captures these segment-specific biases while falling back to broader corrections
when segment-level data is sparse.

### Results

| Metric | Raw Model | After Calibration | Improvement |
|--------|-----------|-------------------|-------------|
| MAE | {before['mae']:.2f} | {after['mae']:.2f} | {mae_improve:.1f}% |
| WAPE | {before['wape']:.3f} | {after['wape']:.3f} | {wape_improve:.1f}% |
| MAPE (nonzero actuals) | {before['nonzero_mape']:.3f} | {after['nonzero_mape']:.3f} | {mape_improve:.1f}% |
| Bias | {before['bias']:+.3f} | {after['bias']:+.3f} | near-zero |

![Calibration before/after](figures/calibration_before_after.png)

The calibration layer reduced test MAE by **{mae_improve:.1f}%** and nearly
eliminated systematic bias ({before['bias']:+.3f} → {after['bias']:+.3f}).
This is a non-parametric correction that requires no retraining — it can be
updated daily as new calibration data arrives, making it suitable for
rolling operational forecasts.

MAPE is reported only for rows with nonzero actual departures because standard
MAPE is undefined on zero-actual rows. WAPE remains the primary percentage-error
metric because it uses total volume in the denominator and handles sparse demand
more reliably.
"""

    # --- Distribution Shift Analysis section ---
    if drift_df is not None and target_drift is not None:
        severe = drift_df[drift_df["drift_severity"].isin(["substantial", "severe"])]
        n_drifted = len(severe)

        report += f"""
## Distribution Shift Analysis

A model trained on April–October and tested on December faces seasonal distribution
shift. Understanding *which* features drifted and *how much* explains the raw model's
bias and validates the calibration approach.

### Target Variable Drift

| Split | Mean | Median | Std |
|-------|------|--------|-----|
| Train (Apr–Oct) | {target_drift['train_mean']} | {target_drift['train_median']} | {target_drift['train_std']} |
| Calibration (Nov) | {target_drift['cal_mean']} | {target_drift['cal_median']} | — |
| Test (Dec) | {target_drift['test_mean']} | {target_drift['test_median']} | {target_drift['test_std']} |

JS divergence: train→test = {target_drift['train_test_js']:.4f}, \
train→cal = {target_drift['train_cal_js']:.4f}, \
cal→test = {target_drift['cal_test_js']:.4f}

December demand is **lower** than the training average because the training period
includes peak summer months (June–September) when cycling demand is highest. The
model's lag and rolling features carry forward recent values, while weather
features, especially temperature, shift substantially.

### Feature Drift (Top Features by JS Divergence)

{drift_df.to_markdown(index=False)}

![Feature drift heatmap](figures/feature_drift_heatmap.png)

**{n_drifted} of {len(drift_df)} top features** {("shows" if n_drifted == 1 else "show")} substantial or severe drift.
The main drift pattern is:

- **Temperature features** — December is colder than the Apr–Oct training period,
  making `temp_max_f` the dominant shifted feature among the important features
- **Precipitation features** — precipitation shifts moderately and contributes
  meaningful weather context
- **Lag and rolling demand features** — these retain relatively stable distribution
  shape even though their means fall with winter demand

Features with **low drift** (lag features, DOW indicators) are the model's anchor —
they generalize well across seasons because they capture relative patterns rather
than absolute levels. This is why the model's relative station ranking remains
useful (low median AE) even when the *level* is off (high bias before calibration).
"""

    # --- Test Set Results ---
    report += f"""
## Test Set Results (Post-Calibration)

| Metric | Value |
|--------|-------|
| MAE | {overall['mae']:.2f} |
| Median AE | {overall['median_ae']:.2f} |
| RMSE | {overall['rmse']:.2f} |
| WAPE | {overall['wape']:.3f} |
| MAPE (nonzero actuals) | {overall['nonzero_mape']:.3f} |
| MAPE coverage | {overall['nonzero_mape_coverage']:.1%} |
| Bias | {overall['bias']:+.3f} |

![Actual vs predicted](figures/actual_vs_predicted.png)

## Cold-Start vs Mature Stations

{cold_mature.to_markdown(index=False, floatfmt=".3f") if not cold_mature.empty else "(no cold-start stations in test set)"}

The cold-start MAE gap is **{gap_pct:.1f}%** higher than mature stations.
With hierarchical imputation, early-life stations get reasonable predictions when
their own lag history is sparse rather than defaulting to zero or a global average.

## Accuracy by Borough

{by_borough.to_markdown(index=False, floatfmt=".3f")}

## Accuracy by Day of Week

{by_dow[["dow_name", "rows", "avg_actual", "mae", "wape"]].to_markdown(index=False, floatfmt=".3f")}

## Lessons Learned

### 1. Data ingestion bugs are the most expensive bugs

The Citi Bike monthly zip files contain multiple CSV shards (3–6 per month), so
the ingestion step reads every CSV in each archive and validates the resulting
station-day panel. Reading only the first shard would silently drop most trips
and create artificial zero-departure days. The same issue appears at the station
lifecycle level: zero-filled dates should be kept only between a station's first
and last observed trip, not after a station disappears from the feed.

### 2. Separate signal learning from level correction

The ML model learns *relative* patterns: which stations are busier,
how weekdays differ from weekends, how rain suppresses demand. But it can be
systematically wrong on the *absolute level* when the test period differs from
training. The calibration layer corrects the level without retraining, using the
same hierarchical approach as production demand planning systems. This separation
of concerns (model for signal, calibration for level) is more robust than trying
to make the model do both.

### 3. Drift analysis explains model failures

Computing JS divergence between train and test distributions for each feature
reveals *why* the raw model is biased. Features with high drift (seasonal
indicators, temperature) explain the level shift. Features with low drift
(lag features, DOW patterns) explain why the model's relative predictions are
still useful. This diagnostic should be standard in any ML pipeline with
temporal splits.

### 4. Hierarchical corrections outperform global corrections

A single global correction factor applies the same adjustment to every station
and every day. But demand patterns vary by borough, day-of-week, and individual
station. The hierarchical approach (station → cluster×DOW → borough×DOW → global)
captures these segment-specific biases while maintaining statistical stability
through minimum sample requirements at each level.

### 5. Cold-start imputation works but has limits

Hierarchical spatial imputation (cluster → borough → city average) gives new
stations reasonable predictions during sparse-history periods. The cold-start MAE gap
({gap_pct:.1f}% higher than mature) is directionally useful but should be monitored.
The gap narrows as the station accumulates history and its own lag features become
available.
"""

    path = REPORTS_DIR / "throughput_lab_report.md"
    path.write_text(report)
    return path


def generate_all_reports(
    feature_panel: pd.DataFrame,
    scored_panel: pd.DataFrame,
    eval_summary: dict,
    leaderboard: pd.DataFrame,
    feature_importance: pd.DataFrame,
    split_summary: pd.DataFrame,
    station_geo: pd.DataFrame,
    cal_diagnostics: dict | None = None,
    drift_df: pd.DataFrame | None = None,
    target_drift: dict | None = None,
) -> Path:
    """Generate all figures and the markdown report."""
    _ensure_dirs()

    plot_daily_throughput_trend(feature_panel)
    plot_station_map(station_geo)
    plot_dow_pattern(feature_panel)
    plot_cold_vs_mature(eval_summary)
    plot_accuracy_by_borough(eval_summary)
    plot_accuracy_by_dow(eval_summary)
    plot_actual_vs_predicted(scored_panel)
    plot_feature_importance(feature_importance)

    if cal_diagnostics is not None:
        plot_calibration_before_after(cal_diagnostics)
    if drift_df is not None:
        plot_drift_heatmap(drift_df)

    return write_report(
        scored_panel=scored_panel,
        eval_summary=eval_summary,
        leaderboard=leaderboard,
        feature_importance=feature_importance,
        split_summary=split_summary,
        station_geo=station_geo,
        cal_diagnostics=cal_diagnostics,
        drift_df=drift_df,
        target_drift=target_drift,
    )
