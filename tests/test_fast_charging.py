import numpy as np

from ev_thermal.components.battery import BatteryState
from ev_thermal.charging.fast_charge import FastChargeConfig, simulate_fast_charge


def test_mild_fast_charge_reaches_target_with_power_and_energy_ledger_limits():
    config = FastChargeConfig(station_power_w=250_000.0, target_soc=0.80, dt_s=5.0)
    result = simulate_fast_charge(
        BatteryState(0.20, 25.0, 25.0), ambient_temp_c=25.0, config=config
    )

    frame = result.timeseries
    assert result.status == "charge_complete"
    assert result.final_state.soc >= 0.80 - 1e-6
    assert (frame["accepted_battery_power_w"] <= config.station_power_w * config.charger_efficiency).all()
    assert (np.diff(frame["soc"]) >= -1e-12).all()
    assert frame["grid_power_w"].max() <= config.station_power_w + 1e-9
    assert frame["power_ledger_residual_w"].abs().max() < 1e-8
    assert set(frame["limiting_reason"]).issubset({
        "station", "temperature", "soc_taper", "current_voltage", "target_soc"
    })
    assert result.charge_time_s > 0


def test_cold_and_hot_batteries_are_derated_and_hard_temperature_stops_safely():
    config = FastChargeConfig(station_power_w=200_000.0, target_soc=0.35, dt_s=5.0)
    mild = simulate_fast_charge(BatteryState(0.20, 25.0, 25.0), 25.0, config)
    cold = simulate_fast_charge(BatteryState(0.20, -15.0, -15.0), -15.0, config)
    hot = simulate_fast_charge(BatteryState(0.20, 44.0, 44.0), 40.0, config)

    assert cold.timeseries["accepted_battery_power_w"].iloc[0] < mild.timeseries["accepted_battery_power_w"].iloc[0]
    assert hot.timeseries["accepted_battery_power_w"].iloc[0] < mild.timeseries["accepted_battery_power_w"].iloc[0]
    assert cold.timeseries["limiting_reason"].iloc[0] == "temperature"
    assert hot.peak_core_temp_c <= config.hard_max_temp_c

    stopped = simulate_fast_charge(
        BatteryState(0.20, config.hard_max_temp_c + 0.1, config.hard_max_temp_c + 0.1),
        45.0,
        config,
    )
    assert stopped.status == "safe_stop_temperature"
