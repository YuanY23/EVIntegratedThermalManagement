from ev_thermal.charging.preconditioning import RoutePreview, rule_preconditioning_command
from ev_thermal.simulation.charging_scenarios import ChargingScenario, simulate_trip_charge


def test_rule_preheating_raises_cold_arrival_temperature_and_reduces_charge_time():
    scenario = ChargingScenario(
        name="cold_arrival",
        ambient_temp_c=-15.0,
        initial_battery_temp_c=-15.0,
        initial_soc=0.35,
        route_time_s=2400,
        target_soc=0.80,
        station_power_w=220_000.0,
    )

    none = simulate_trip_charge(scenario, "none")
    rule = simulate_trip_charge(scenario, "rule")

    assert rule.arrival_core_temp_c > none.arrival_core_temp_c
    assert rule.preconditioning_energy_kwh > 0
    assert rule.charge_time_s < none.charge_time_s
    assert rule.final_soc >= scenario.target_soc - 1e-6
    assert rule.peak_core_temp_c < 50.0
    assert rule.route_timeseries["power_closure_residual_w"].abs().max() < 1e-8
    assert (
        rule.route_timeseries["actual_traction_power_w"]
        <= rule.route_timeseries["available_traction_power_w"]
    ).all()


def test_rule_precooling_handles_hot_arrival_and_invalid_preview_falls_back():
    hot = ChargingScenario(
        name="hot_arrival",
        ambient_temp_c=40.0,
        initial_battery_temp_c=45.0,
        initial_soc=0.40,
        route_time_s=1800,
        target_soc=0.75,
        station_power_w=220_000.0,
    )
    none = simulate_trip_charge(hot, "none")
    rule = simulate_trip_charge(hot, "rule")
    assert rule.arrival_surface_temp_c < none.arrival_surface_temp_c
    assert rule.peak_core_temp_c <= 50.0

    preview = RoutePreview(
        remaining_time_s=1200,
        predicted_arrival_soc=0.3,
        ambient_temp_c=-10.0,
        valid=False,
        route_active=True,
    )
    command = rule_preconditioning_command(-10.0, preview)
    assert not command.active
    assert command.thermal_power_w == 0
    assert command.reason == "preview_unavailable"


def test_route_cancellation_disables_preconditioning_deterministically():
    preview = RoutePreview(
        remaining_time_s=1200,
        predicted_arrival_soc=0.3,
        ambient_temp_c=-10.0,
        valid=True,
        route_active=False,
    )
    command = rule_preconditioning_command(-10.0, preview)
    assert not command.active
    assert command.reason == "route_inactive"
