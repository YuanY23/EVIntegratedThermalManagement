import numpy as np
import pandas as pd

from ev_thermal.charging.optimization import OBJECTIVES, optimize_preconditioning, pareto_front
from ev_thermal.simulation.charging_scenarios import ChargingScenario


def _cold_scenario(**overrides):
    values = {
        "name": "cold_optimization",
        "ambient_temp_c": -15.0,
        "initial_battery_temp_c": -15.0,
        "initial_soc": 0.35,
        "route_time_s": 2400,
        "target_soc": 0.80,
        "station_power_w": 220_000.0,
    }
    values.update(overrides)
    return ChargingScenario(**values)


def test_pareto_front_contains_no_dominated_candidate():
    result = optimize_preconditioning(
        _cold_scenario(),
        lead_times_s=(600.0, 1200.0, 2400.0),
        target_temperatures_c=(5.0, 12.0, 20.0),
        thermal_powers_w=(2500.0, 5000.0),
    )

    assert not result.fallback_used
    assert {"none", "rule"}.issubset(set(result.candidates["strategy"]))
    assert result.recommended["feasible"]
    assert result.recommended["charge_status"] == "charge_complete"
    assert result.recommended["final_soc"] >= 0.80 - 1e-6
    assert result.recommended["peak_core_temp_c"] <= 50.0

    front = result.pareto
    assert len(front) > 1
    for index, row in front.iterrows():
        other = front.drop(index=index)
        dominated = (
            (other[list(OBJECTIVES)] <= row[list(OBJECTIVES)].to_numpy()).all(axis=1)
            & (other[list(OBJECTIVES)] < row[list(OBJECTIVES)].to_numpy()).any(axis=1)
        )
        assert not dominated.any()
    assert pareto_front(front, OBJECTIVES).shape[0] == len(front)


def test_engineering_resolution_never_keeps_a_raw_dominated_representative():
    table = pd.DataFrame([
        {
            "candidate_id": "dominated-first",
            "charge_time_s": 100.0,
            "preconditioning_energy_kwh": 0.04,
            "relative_aging_damage": 1.0e-6,
        },
        {
            "candidate_id": "dominant-second",
            "charge_time_s": 90.0,
            "preconditioning_energy_kwh": 0.03,
            "relative_aging_damage": 0.5e-6,
        },
    ])

    front = pareto_front(table)

    assert front["candidate_id"].tolist() == ["dominant-second"]


def test_optimization_is_deterministic_and_recommendation_improves_one_baseline_objective():
    kwargs = {
        "lead_times_s": (600.0, 1200.0),
        "target_temperatures_c": (8.0, 15.0, 22.0),
        "thermal_powers_w": (3000.0, 5000.0),
    }
    first = optimize_preconditioning(_cold_scenario(), **kwargs)
    second = optimize_preconditioning(_cold_scenario(), **kwargs)
    assert first.recommended == second.recommended
    assert first.pareto.equals(second.pareto)

    none = first.candidates[first.candidates["strategy"] == "none"].iloc[0]
    recommended = first.recommended
    improved = [recommended[name] < none[name] for name in OBJECTIVES]
    assert any(improved)
    assert np.isfinite(first.candidates.select_dtypes(include=[np.number]).to_numpy()).all()


def test_invalid_preview_and_no_arrival_soc_margin_return_explicit_fallback():
    invalid_preview = optimize_preconditioning(
        _cold_scenario(route_preview_valid=False),
        lead_times_s=(600.0,),
        target_temperatures_c=(15.0,),
        thermal_powers_w=(3000.0,),
    )
    assert invalid_preview.fallback_used
    assert "preview" in invalid_preview.fallback_reason
    assert invalid_preview.recommended["strategy"] == "none"

    depleted = optimize_preconditioning(
        _cold_scenario(initial_soc=0.05, route_time_s=2400),
        lead_times_s=(600.0,),
        target_temperatures_c=(15.0,),
        thermal_powers_w=(3000.0,),
    )
    assert depleted.fallback_used
    assert "feasible" in depleted.fallback_reason
