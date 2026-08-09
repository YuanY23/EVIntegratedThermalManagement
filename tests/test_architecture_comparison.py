import numpy as np

from ev_thermal.config import load_config
from ev_thermal.pipeline import run_architecture_comparison, run_architecture_sizing
from ev_thermal.simulation.integrated import IntegratedSimulator
from ev_thermal.simulation.scenarios import make_scenario
from ev_thermal.thermal_hydraulics.topologies import (
    HydraulicDesign,
    architecture_names,
    cross_loop_exchange,
    shared_heat_sink_rejection,
)
from ev_thermal.thermal_hydraulics.heat_exchanger import Radiator


def test_integrated_simulator_uses_network_and_resistance_changes_physics():
    config = load_config()
    scenario = make_scenario("urban_hot", 300, config.simulation.dt_s, seed=21)
    reference = IntegratedSimulator(
        config, hydraulic_design=HydraulicDesign(local_resistance_scale=1.0)
    ).run(scenario)
    restricted = IntegratedSimulator(
        config, hydraulic_design=HydraulicDesign(local_resistance_scale=3.0)
    ).run(scenario)

    assert restricted.timeseries["battery_flow_kg_s"].mean() < reference.timeseries["battery_flow_kg_s"].mean()
    assert not np.isclose(
        restricted.timeseries["battery_surface_temp_c"].iloc[-1],
        reference.timeseries["battery_surface_temp_c"].iloc[-1],
    )
    required = {
        "battery_system_pressure_drop_pa",
        "battery_cold_plate_pressure_drop_pa",
        "battery_hydraulic_closure_residual_pa",
        "powertrain_system_pressure_drop_pa",
        "hydraulic_solver_failure_count",
    }
    assert required.issubset(reference.timeseries.columns)
    assert reference.timeseries["battery_hydraulic_closure_residual_pa"].abs().max() < 1e-3
    assert reference.metrics["thermal_balance_error_pct"] < 2.0


def test_liquid_hx_heat_is_equal_opposite_and_limited_by_temperature_direction():
    independent = cross_loop_exchange(
        "independent_dual_loop", 70.0, 25.0, 0.2, 0.15, ua_w_k=900.0
    )
    coupled = cross_loop_exchange(
        "coupled_dual_loop", 70.0, 25.0, 0.2, 0.15, ua_w_k=900.0
    )
    reversed_flow = cross_loop_exchange(
        "coupled_dual_loop", 20.0, 35.0, 0.2, 0.15, ua_w_k=900.0
    )

    assert independent.drive_to_battery_heat_w == 0
    assert coupled.drive_to_battery_heat_w > 0
    assert coupled.powertrain_out_temp_c < 70.0
    assert coupled.battery_out_temp_c > 25.0
    assert 0 < coupled.effectiveness <= 1
    assert reversed_flow.drive_to_battery_heat_w < 0


def test_shared_sink_preserves_signed_branch_enthalpy_when_temperatures_cross_ambient():
    result = shared_heat_sink_rejection(
        Radiator(),
        battery_temp_c=10.0,
        powertrain_temp_c=60.0,
        ambient_temp_c=25.0,
        battery_flow_kg_s=0.25,
        powertrain_flow_kg_s=0.25,
        vehicle_speed_mps=0.0,
        fan_fraction=1.0,
    )

    assert result.battery_heat_rejected_w < 0.0
    assert result.powertrain_heat_rejected_w > 0.0
    assert np.isclose(
        result.battery_heat_rejected_w + result.powertrain_heat_rejected_w,
        result.total_heat_rejected_w,
    )


def test_three_architectures_produce_complete_reproducible_tables_and_infeasible_specs():
    config = load_config()
    first = run_architecture_comparison(config, ["urban_hot"], duration_s=180)
    second = run_architecture_comparison(config, ["urban_hot"], duration_s=180)

    assert set(first["architecture"]) == set(architecture_names())
    assert len(first) == 3
    assert np.isfinite(first.select_dtypes(include=[np.number]).to_numpy()).all()
    assert first["feasible"].all()
    assert first.equals(second)

    sizing = run_architecture_sizing(
        config,
        scenario_name="hill_high_load",
        duration_s=180,
        pump_scales=(0.2, 1.0),
        radiator_ua_scales=(1.0,),
    )
    assert len(sizing) == 6
    assert (~sizing["feasible"]).any()
    assert sizing["infeasibility_reason"].notna().all()
