"""Generate constrained temperature-time-energy-aging Pareto results."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from ev_thermal.artifacts import write_upgrade_suite_manifest
from ev_thermal.charging.optimization import OBJECTIVES, optimize_preconditioning
from ev_thermal.charging.preconditioning import PreconditioningPolicy
from ev_thermal.simulation.charging_scenarios import (
    charging_scenario_names,
    make_charging_scenario,
    simulate_trip_charge,
)
from ev_thermal.visualization import plot_joint_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "optimization")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    candidate_frames = []
    pareto_frames = []
    recommendations = []
    for name in charging_scenario_names():
        scenario = make_charging_scenario(name)
        optimization = optimize_preconditioning(scenario)
        candidate_frames.append(optimization.candidates)
        pareto_frames.append(optimization.pareto)
        recommendations.append(optimization.recommended)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    pareto = pd.concat(pareto_frames, ignore_index=True)
    recommended = pd.DataFrame(recommendations)
    candidates.to_csv(output / "joint_optimization_candidates.csv", index=False)
    pareto.to_csv(output / "joint_optimization_pareto.csv", index=False)
    recommended.to_csv(output / "joint_optimization_recommended.csv", index=False)
    candidates[[
        "candidate_id", "scenario", "strategy", "feasible", "constraint_failure_reason",
        "charge_complete", "final_soc_ok", "arrival_soc_ok", "temperature_ok", "power_closure_ok",
    ]].to_csv(output / "constraint_audit.csv", index=False)
    plot_joint_optimization(pareto, recommended, output / "joint_optimization_pareto.png")

    robustness_rows = []
    cold_recommended = recommended[recommended["scenario"] == "cold_arrival"].iloc[0]
    policy = PreconditioningPolicy(
        float(cold_recommended["start_before_arrival_s"]),
        float(cold_recommended["target_temp_c"]),
        max(float(cold_recommended["max_thermal_power_w"]), 500.0),
    )
    cold = make_charging_scenario("cold_arrival")
    perturbations = {
        "nominal": cold,
        "ambient_minus_5c": replace(cold, ambient_temp_c=cold.ambient_temp_c - 5.0),
        "ambient_plus_5c": replace(cold, ambient_temp_c=cold.ambient_temp_c + 5.0),
        "arrival_5min_early": replace(cold, route_time_s=cold.route_time_s - 300),
        "arrival_5min_late": replace(cold, route_time_s=cold.route_time_s + 300),
        "station_derated_100kw": replace(cold, station_power_w=100_000.0),
    }
    for case, scenario in perturbations.items():
        result = simulate_trip_charge(scenario, "optimized", policy)
        robustness_rows.append({
            "case": case,
            "arrival_core_temp_c": result.arrival_core_temp_c,
            "arrival_soc": result.arrival_soc,
            "charge_time_min": result.charge_time_s / 60.0,
            "preconditioning_energy_kwh": result.preconditioning_energy_kwh,
            "relative_aging_damage": result.relative_aging_damage,
            "peak_core_temp_c": result.peak_core_temp_c,
            "charge_status": result.charge_status,
            "feasible": result.charge_status == "charge_complete"
                        and result.arrival_soc >= 0.10 and result.peak_core_temp_c <= 50.0,
        })
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(output / "optimization_robustness.csv", index=False)

    summary = {
        "objective_columns": list(OBJECTIVES),
        "candidate_count": len(candidates),
        "feasible_candidate_count": int(candidates["feasible"].sum()),
        "pareto_count": len(pareto),
        "recommended_candidates": recommended[
            ["scenario", "candidate_id", "strategy", "recommendation_basis"]
        ].to_dict(orient="records"),
        "all_recommendations_feasible": bool(recommended["feasible"].all()),
        "robustness_cases": len(robustness),
        "all_robustness_cases_feasible": bool(robustness["feasible"].all()),
        "aging_claim_scope": "relative strategy comparison only",
    }
    (output / "joint_optimization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_upgrade_suite_manifest(ROOT, "optimization", output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
