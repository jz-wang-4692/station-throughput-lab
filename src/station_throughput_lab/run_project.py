"""
End-to-end pipeline: download → panel → features → cold-start → model → evaluate → report.
"""
from __future__ import annotations

import argparse

from station_throughput_lab.config import (
    AG_TIME_LIMIT,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    TRAIN_END,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Station Throughput Lab: predict daily bike-station departures."
    )
    parser.add_argument(
        "--time-limit", type=int, default=AG_TIME_LIMIT,
        help=f"AutoGluon training time limit in seconds (default: {AG_TIME_LIMIT}).",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Force rebuild of all cached data artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create directories
    for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download trip data ---
    from station_throughput_lab.data import download_trip_data, build_daily_station_panel
    from station_throughput_lab.config import ALL_MONTHS

    months = ALL_MONTHS

    print("=" * 60)
    print("Step 1: Downloading Citi Bike trip data")
    print("=" * 60)
    trip_paths = download_trip_data(months)
    if not trip_paths:
        raise RuntimeError("No trip data files available.")

    # --- Step 2: Build daily station panel ---
    print("\n" + "=" * 60)
    print("Step 2: Building daily station panel")
    print("=" * 60)
    daily_panel = build_daily_station_panel(
        trip_paths, force_rebuild=args.force_rebuild,
    )

    # --- Step 3: Geospatial features ---
    print("\n" + "=" * 60)
    print("Step 3: Computing geospatial features")
    print("=" * 60)
    from station_throughput_lab.geo_features import build_station_geo_features

    station_meta = (
        daily_panel[["station_id", "lat", "lon"]]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )
    station_geo = build_station_geo_features(station_meta)
    n_clusters = station_geo["spatial_cluster"].nunique()
    print(f"  {len(station_geo)} stations, {n_clusters} spatial clusters.")
    borough_counts = station_geo["borough_proxy"].value_counts()
    for borough, count in borough_counts.items():
        print(f"    {borough}: {count} stations")

    # --- Step 4: Weather data ---
    print("\n" + "=" * 60)
    print("Step 4: Fetching weather data")
    print("=" * 60)
    from station_throughput_lab.weather import build_weather_data

    weather = build_weather_data(force_rebuild=args.force_rebuild)
    print(f"  Weather data: {len(weather)} days.")

    # --- Step 5: Feature engineering ---
    print("\n" + "=" * 60)
    print("Step 5: Building feature panel")
    print("=" * 60)
    from station_throughput_lab.features import build_feature_panel, fill_missing_features

    feature_panel = build_feature_panel(
        daily_panel, station_geo, force_rebuild=args.force_rebuild,
    )

    # --- Step 6: Cold-start imputation ---
    print("\n" + "=" * 60)
    print("Step 6: Cold-start imputation")
    print("=" * 60)
    from station_throughput_lab.cold_start import (
        compute_hierarchical_baselines,
        identify_cold_start_stations,
        impute_cold_start_features,
    )

    cold_ids, mature_ids = identify_cold_start_stations(feature_panel)
    print(f"  Cold-start stations: {len(cold_ids)}")
    print(f"  Mature stations: {len(mature_ids)}")

    cluster_avg, borough_avg, city_avg = compute_hierarchical_baselines(feature_panel)
    feature_panel = impute_cold_start_features(
        feature_panel, cluster_avg, borough_avg, city_avg,
    )
    feature_panel = fill_missing_features(feature_panel, train_end=TRAIN_END)

    # --- Step 7: Train model ---
    print("\n" + "=" * 60)
    print("Step 7: Training AutoGluon model")
    print("=" * 60)
    from station_throughput_lab.modeling import train_model

    result = train_model(feature_panel, time_limit=args.time_limit)

    # --- Step 8: Calibration optimization ---
    print("\n" + "=" * 60)
    print("Step 8: Calibration-based bias correction")
    print("=" * 60)
    from station_throughput_lab.optimization import run_calibration_optimization

    cal_result = run_calibration_optimization(result.scored_panel)
    # Replace scored panel with calibrated version
    scored_panel = cal_result.corrected_panel

    # --- Step 9: Evaluate (post-calibration) ---
    print("\n" + "=" * 60)
    print("Step 9: Evaluation (post-calibration)")
    print("=" * 60)
    from station_throughput_lab.evaluation import build_evaluation_summary

    eval_summary = build_evaluation_summary(scored_panel)

    print(f"\nOverall test MAE: {eval_summary['overall']['mae']:.2f}")
    print(f"Overall test WAPE: {eval_summary['overall']['wape']:.3f}")
    print("\nCold-start vs Mature:")
    print(eval_summary["cold_vs_mature"].to_string(index=False))

    # --- Step 10: Drift analysis ---
    print("\n" + "=" * 60)
    print("Step 10: Distribution shift analysis")
    print("=" * 60)
    from station_throughput_lab.drift import compute_feature_drift, compute_target_drift

    # Use top features from importance for drift analysis
    fallback_features = [
        "dep_lag_1", "dep_lag_2", "dep_lag_7",
        "rolling_mean_7", "rolling_median_7", "rolling_mean_14",
        "rolling_median_14", "rolling_min_14", "rolling_max_14",
        "dow_rolling_mean_4", "dow_rolling_median_4",
        "temp_max_f", "precip_in", "station_historical_avg",
    ]
    top_features = (
        result.feature_importance["feature"].head(15).tolist()
        if not result.feature_importance.empty
        else [f for f in fallback_features if f in scored_panel.columns]
    )
    drift_df = compute_feature_drift(scored_panel, top_features)
    target_drift = compute_target_drift(scored_panel)

    print(f"\nTarget drift (train→test JS): {target_drift['train_test_js']:.4f}")
    print(f"Features with substantial+ drift: "
          f"{len(drift_df[drift_df['drift_severity'].isin(['substantial', 'severe'])])}"
          f" of {len(drift_df)}")
    if not drift_df.empty:
        print(drift_df[["feature", "js_divergence", "drift_severity"]].to_string(index=False))

    # --- Step 11: Generate reports ---
    print("\n" + "=" * 60)
    print("Step 11: Generating reports")
    print("=" * 60)
    from station_throughput_lab.reporting import generate_all_reports

    report_path = generate_all_reports(
        feature_panel=feature_panel,
        scored_panel=scored_panel,
        eval_summary=eval_summary,
        leaderboard=result.leaderboard,
        feature_importance=result.feature_importance,
        split_summary=result.split_summary,
        station_geo=station_geo,
        cal_diagnostics=cal_result.calibration_diagnostics,
        drift_df=drift_df,
        target_drift=target_drift,
    )

    # Save artifacts
    scored_panel.to_parquet(REPORTS_DIR / "scored_panel.parquet", index=False)
    cal_result.correction_factors.to_csv(
        REPORTS_DIR / "calibration_factors.csv", index=False,
    )
    drift_df.to_csv(REPORTS_DIR / "feature_drift.csv", index=False)
    eval_summary["baseline_comparison"].to_csv(
        REPORTS_DIR / "baseline_comparison.csv", index=False,
    )
    eval_summary["cold_vs_mature"].to_csv(
        REPORTS_DIR / "cold_vs_mature.csv", index=False,
    )
    eval_summary["by_station_volume"].to_csv(
        REPORTS_DIR / "station_volume_segments.csv", index=False,
    )
    eval_summary["by_borough"].to_csv(
        REPORTS_DIR / "accuracy_by_borough.csv", index=False,
    )

    print(f"\nReport: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
