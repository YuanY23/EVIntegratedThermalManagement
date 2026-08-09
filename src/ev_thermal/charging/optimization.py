"""Constraint-first Pareto optimization of arrival preconditioning policies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .preconditioning import PreconditioningPolicy
from ..simulation.charging_scenarios import ChargingScenario, simulate_trip_charge


OBJECTIVES = (
    "charge_time_s",
    "preconditioning_energy_kwh",
    "relative_aging_damage",
)
OBJECTIVE_RESOLUTION = {
    "charge_time_s": 60.0,
    "preconditioning_energy_kwh": 0.10,
    "relative_aging_damage": 5.0e-6,
}


@dataclass(frozen=True)
class OptimizationResult:
    candidates: pd.DataFrame
    pareto: pd.DataFrame
    recommended: dict
    fallback_used: bool
    fallback_reason: str


def pareto_front(table: pd.DataFrame, objectives: tuple[str, ...] = OBJECTIVES) -> pd.DataFrame:
    """Return deterministic non-dominated rows for minimization objectives."""
    if table.empty:
        return table.copy()
    raw_values = table.loc[:, objectives].to_numpy(dtype=float)
    keep = np.ones(len(raw_values), dtype=bool)
    for index, candidate in enumerate(raw_values):
        dominated = np.any(
            np.all(raw_values <= candidate, axis=1)
            & np.any(raw_values < candidate, axis=1)
        )
        keep[index] = not dominated
    raw_front = table.loc[table.index.to_numpy()[keep]].copy()
    resolution = np.asarray([OBJECTIVE_RESOLUTION.get(name, 0.0) for name in objectives])
    front_values = raw_front.loc[:, objectives].to_numpy(dtype=float)
    engineering_cells = np.column_stack([
        np.round(front_values[:, index] / step) if step > 0 else front_values[:, index]
        for index, step in enumerate(resolution)
    ])
    _, representative_indices = np.unique(engineering_cells, axis=0, return_index=True)
    selected_indices = raw_front.index.to_numpy()[np.sort(representative_indices)]
    return raw_front.loc[selected_indices].sort_values(
        list(objectives) + ["candidate_id"]
    ).reset_index(drop=True)


def _candidate_row(result, candidate_id: str, strategy: str,
                   policy: PreconditioningPolicy | None) -> dict:
    route = result.route_timeseries
    power_closure = float(route["power_closure_residual_w"].abs().max()) if len(route) else 0.0
    arrival_target_c = 25.0
    constraints = {
        "charge_complete": result.charge_status == "charge_complete",
        "final_soc_ok": result.final_soc >= 0.0,
        "arrival_soc_ok": result.arrival_soc >= 0.10,
        "temperature_ok": result.peak_core_temp_c <= 50.0,
        "power_closure_ok": power_closure < 1e-6,
    }
    row = {
        "candidate_id": candidate_id,
        "scenario": result.scenario,
        "strategy": strategy,
        "start_before_arrival_s": 0.0 if policy is None else policy.start_before_arrival_s,
        "target_temp_c": result.arrival_core_temp_c if policy is None else policy.target_temp_c,
        "max_thermal_power_w": 0.0 if policy is None else policy.max_thermal_power_w,
        "arrival_core_temp_c": result.arrival_core_temp_c,
        "arrival_surface_temp_c": result.arrival_surface_temp_c,
        "arrival_temperature_deviation_c": abs(result.arrival_core_temp_c - arrival_target_c),
        "arrival_soc": result.arrival_soc,
        "preconditioning_energy_kwh": result.preconditioning_energy_kwh,
        "charge_time_s": result.charge_time_s,
        "charge_time_min": result.charge_time_s / 60.0,
        "grid_energy_kwh": result.grid_energy_kwh,
        "final_soc": result.final_soc,
        "peak_core_temp_c": result.peak_core_temp_c,
        "relative_aging_damage": result.relative_aging_damage,
        "charge_status": result.charge_status,
        "preconditioning_status": result.preconditioning_status,
        "max_route_power_closure_residual_w": power_closure,
        **constraints,
    }
    row["feasible"] = bool(all(constraints.values()) and result.final_soc >= 0.10)
    failed = [name for name, passed in constraints.items() if not passed]
    row["constraint_failure_reason"] = "ok" if row["feasible"] else ";".join(failed or ["final_soc"])
    return row


def _recommended(front: pd.DataFrame) -> dict:
    normalized = pd.DataFrame(index=front.index)
    for objective in OBJECTIVES:
        low = float(front[objective].min())
        high = float(front[objective].max())
        normalized[objective] = (
            (front[objective] - low) / (high - low) if high - low > 1e-12 else 0.0
        )
    weights = {
        "charge_time_s": 0.45,
        "preconditioning_energy_kwh": 0.20,
        "relative_aging_damage": 0.35,
    }
    scores = sum(normalized[name] * weight for name, weight in weights.items())
    selected_index = scores.idxmin()
    selected = front.loc[selected_index].to_dict()
    selected["normalized_engineering_score"] = float(scores.loc[selected_index])
    weight_text = "/".join(f"{weights[name]:.2f}" for name in OBJECTIVES)
    selected["recommendation_basis"] = (
        f"normalized Pareto objectives with {weight_text} weights"
    )
    return selected


def optimize_preconditioning(
    scenario: ChargingScenario,
    lead_times_s: tuple[float, ...] = (600.0, 1200.0, 1800.0, 2400.0),
    target_temperatures_c: tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 35.0),
    thermal_powers_w: tuple[float, ...] = (2500.0, 3750.0, 5000.0),
) -> OptimizationResult:
    """Evaluate baselines and a bounded grid, then select from the Pareto set."""
    rows = [
        _candidate_row(simulate_trip_charge(scenario, "none"), "baseline-none", "none", None),
        _candidate_row(simulate_trip_charge(scenario, "rule"), "baseline-rule", "rule", None),
    ]
    if not scenario.route_preview_valid or not scenario.route_active:
        candidates = pd.DataFrame(rows)
        recommended = candidates[candidates["strategy"] == "none"].iloc[0].to_dict()
        return OptimizationResult(
            candidates,
            pareto_front(candidates[candidates["feasible"]], OBJECTIVES),
            recommended,
            True,
            "route preview unavailable or inactive; deterministic no-preconditioning fallback",
        )

    simulations = {}
    for candidate_index, (lead_time, target, power) in enumerate(
        product(lead_times_s, target_temperatures_c, thermal_powers_w)
    ):
        policy = PreconditioningPolicy(
            min(float(lead_time), float(scenario.route_time_s)),
            float(target),
            float(power),
        )
        key = (
            policy.start_before_arrival_s,
            policy.target_temp_c,
            policy.max_thermal_power_w,
        )
        result = simulations.get(key)
        if result is None:
            result = simulate_trip_charge(scenario, "optimized", policy)
            simulations[key] = result
        rows.append(_candidate_row(
            result, f"optimized-{candidate_index:03d}", "optimized", policy
        ))
    candidates = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    feasible = candidates[candidates["feasible"]].copy()
    if feasible.empty:
        recommended = candidates[candidates["strategy"] == "none"].iloc[0].to_dict()
        return OptimizationResult(
            candidates,
            feasible,
            recommended,
            True,
            "no feasible optimization candidate; deterministic no-preconditioning fallback",
        )
    front = pareto_front(feasible, OBJECTIVES)
    return OptimizationResult(candidates, front, _recommended(front), False, "")
