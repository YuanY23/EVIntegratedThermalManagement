"""Run synthetic parameter recovery, independent V&V, and sensitivity analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from ev_thermal.calibration.identification import (
    BATTERY_THERMAL_PARAMETERS,
    generate_synthetic_battery_observations,
    identify_battery_thermal_parameters,
)
from ev_thermal.calibration.parameters import ParameterRegistry
from ev_thermal.calibration.reporting import plot_calibration_summary
from ev_thermal.calibration.sensitivity import global_sensitivity, local_sensitivity
from ev_thermal.calibration.validation import compare_parameter_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-samples", type=int, default=128)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    registry = ParameterRegistry.from_yaml(ROOT / "configs" / "parameter_registry.yaml")
    truth = {
        "battery.core_heat_capacity_j_k": 330_000.0,
        "battery.surface_heat_capacity_j_k": 118_000.0,
        "battery.core_surface_conductance_w_k": 295.0,
    }
    observations = generate_synthetic_battery_observations(
        truth,
        episode_count=6,
        duration_s=700,
        dt_s=5,
        noise_std_c=0.015,
        seed=args.seed,
        dataset_id="synthetic-battery-thermal-v1",
    )
    training = observations.subset([0, 1, 2, 3])
    validation = observations.subset([4, 5])
    training.to_csv(output / "synthetic_training_observations.csv")
    validation.to_csv(output / "synthetic_holdout_observations.csv")

    result = identify_battery_thermal_parameters(training, registry)
    result.write_artifacts(output, registry)
    if not result.success:
        raise SystemExit(f"Identification failed: {result.message}")

    baseline = registry.defaults(BATTERY_THERMAL_PARAMETERS)
    comparison = compare_parameter_sets(validation, baseline, result.estimates)
    comparison.metrics.to_csv(output / "holdout_validation_metrics.csv", index=False)
    local = local_sensitivity(validation, registry, result.estimates, BATTERY_THERMAL_PARAMETERS)
    local.to_csv(output / "local_sensitivity.csv", index=False)
    global_result = global_sensitivity(
        validation,
        registry,
        BATTERY_THERMAL_PARAMETERS,
        sample_count=args.global_samples,
        seed=args.seed,
    )
    global_result.rankings.to_csv(output / "global_sensitivity_rankings.csv", index=False)
    global_result.metric_intervals.to_csv(output / "global_sensitivity_intervals.csv", index=False)
    parameter_table = pd.read_csv(output / "parameter_estimates.csv")
    plot_calibration_summary(
        parameter_table,
        truth,
        comparison.metrics,
        local,
        output / "calibration_summary.png",
    )

    statement = {
        "dataset_id": observations.dataset_id,
        "data_maturity": "synthetic",
        "calibration_maturity": result.maturity,
        "model_confirmation": False,
        "claim_scope": "synthetic truth recovery and method verification only",
        "training_episode_ids": list(training.episode_ids),
        "holdout_episode_ids": list(validation.episode_ids),
        "identification_message": result.message,
        "jacobian_rank": result.jacobian_rank,
        "condition_number": result.condition_number,
        "holdout_baseline_rmse_c": comparison.baseline_rmse_c,
        "holdout_calibrated_rmse_c": comparison.calibrated_rmse_c,
        "holdout_improvement_pct": comparison.improvement_pct,
        "hidden_truth": truth,
    }
    (output / "maturity_statement.json").write_text(
        json.dumps(statement, indent=2), encoding="utf-8"
    )
    print(json.dumps(statement, indent=2))


if __name__ == "__main__":
    main()
