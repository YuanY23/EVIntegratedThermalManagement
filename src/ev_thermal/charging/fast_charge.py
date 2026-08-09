"""SOC/temperature-limited fast-charge event with explicit power accounting."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..components.battery import BatteryModel, BatteryParameters, BatteryState
from .aging import AgingParameters, incremental_aging


@dataclass(frozen=True)
class FastChargeConfig:
    station_power_w: float = 250_000.0
    charger_efficiency: float = 0.95
    target_soc: float = 0.80
    dt_s: float = 5.0
    max_duration_s: float = 14_400.0
    max_current_a: float = 550.0
    hard_max_temp_c: float = 50.0
    active_cooling_on_c: float = 34.0
    active_heating_below_c: float = 12.0
    battery_capacity_kwh: float = 75.0
    nominal_voltage_v: float = 380.0

    def __post_init__(self) -> None:
        if self.station_power_w <= 0 or not 0 < self.charger_efficiency <= 1:
            raise ValueError("Station power and charger efficiency must be physical")
        if not 0 < self.target_soc <= 1 or self.dt_s <= 0 or self.max_duration_s <= 0:
            raise ValueError("Invalid target SOC or charge duration settings")
        if self.max_current_a <= 0 or self.hard_max_temp_c <= self.active_cooling_on_c:
            raise ValueError("Invalid fast-charge current or temperature limits")


@dataclass(frozen=True)
class ChargeAcceptance:
    accepted_battery_power_w: float
    limiting_reason: str
    temperature_factor: float
    soc_factor: float
    current_voltage_limit_w: float


@dataclass(frozen=True)
class FastChargeResult:
    status: str
    final_state: BatteryState
    timeseries: pd.DataFrame
    charge_time_s: float
    grid_energy_kwh: float
    battery_energy_kwh: float
    peak_core_temp_c: float
    relative_aging_damage: float


def _temperature_factor(temperature_c: float) -> float:
    if temperature_c <= -20.0 or temperature_c >= 55.0:
        return 0.0
    if temperature_c < 0.0:
        return 0.05 + 0.25 * (temperature_c + 20.0) / 20.0
    if temperature_c < 15.0:
        return 0.30 + 0.70 * temperature_c / 15.0
    if temperature_c <= 38.0:
        return 1.0
    return max(0.05, 1.0 - 0.90 * (temperature_c - 38.0) / 17.0)


def _soc_factor(soc: float) -> float:
    if soc <= 0.55:
        return 1.0
    return max(0.08, (1.0 - soc) / 0.45)


def charge_acceptance(state: BatteryState, model: BatteryModel, config: FastChargeConfig,
                      auxiliary_power_w: float) -> ChargeAcceptance:
    dc_available = max(config.station_power_w * config.charger_efficiency - auxiliary_power_w, 0.0)
    temperature_factor = _temperature_factor(state.core_temp_c)
    soc_factor = _soc_factor(state.soc)
    resistance = model.resistance_ohm(state.soc, state.core_temp_c)
    ocv = model.ocv_v(state.soc)
    current_voltage_limit = config.max_current_a * (ocv + config.max_current_a * resistance)
    candidates = {
        "station": dc_available,
        "temperature": config.station_power_w * config.charger_efficiency * temperature_factor,
        "soc_taper": config.station_power_w * config.charger_efficiency * soc_factor,
        "current_voltage": current_voltage_limit,
    }
    reason = min(candidates, key=candidates.get)
    return ChargeAcceptance(
        max(0.0, float(candidates[reason])),
        reason,
        temperature_factor,
        soc_factor,
        current_voltage_limit,
    )


def simulate_fast_charge(initial_state: BatteryState, ambient_temp_c: float,
                         config: FastChargeConfig | None = None,
                         aging_parameters: AgingParameters | None = None) -> FastChargeResult:
    config = config or FastChargeConfig()
    params = replace(
        BatteryParameters(),
        capacity_kwh=config.battery_capacity_kwh,
        nominal_voltage_v=config.nominal_voltage_v,
        max_charge_power_w=config.station_power_w,
    )
    model = BatteryModel(params)
    state = initial_state
    capacity_ah = config.battery_capacity_kwh * 1000.0 / config.nominal_voltage_v
    rows = []
    status = "max_duration"
    total_damage = 0.0
    for time_s in np.arange(0.0, config.max_duration_s + config.dt_s, config.dt_s):
        if state.soc >= config.target_soc - 1e-9:
            status = "charge_complete"
            break
        if state.core_temp_c >= config.hard_max_temp_c:
            status = "safe_stop_temperature"
            break

        heating_thermal_w = 5_000.0 if state.core_temp_c < config.active_heating_below_c else 0.0
        cooling_thermal_w = 8_000.0 if state.core_temp_c > config.active_cooling_on_c else 0.0
        thermal_aux_power = heating_thermal_w / 0.95 + cooling_thermal_w / 3.0
        pump_power = 180.0 if heating_thermal_w > 0 or cooling_thermal_w > 0 else 80.0
        auxiliary_power = 500.0 + pump_power + thermal_aux_power
        acceptance = charge_acceptance(state, model, config, auxiliary_power)

        remaining_ah = max(config.target_soc - state.soc, 0.0) * capacity_ah
        target_current = remaining_ah * 3600.0 / config.dt_s
        target_current = min(target_current, config.max_current_a)
        target_power = target_current * (
            model.ocv_v(state.soc) + target_current * model.resistance_ohm(state.soc, state.core_temp_c)
        )
        accepted_power = min(acceptance.accepted_battery_power_w, target_power)
        limiting_reason = acceptance.limiting_reason
        if target_power <= acceptance.accepted_battery_power_w:
            limiting_reason = "target_soc"

        passive_ua = 45.0
        cooling_ua = cooling_thermal_w / max(state.surface_temp_c - 15.0, 5.0)
        total_ua = passive_ua + cooling_ua
        coolant_temp = (
            passive_ua * ambient_temp_c + cooling_ua * 15.0
        ) / max(total_ua, 1e-9)
        step = model.step(
            state,
            -accepted_power,
            coolant_temp,
            total_ua,
            config.dt_s,
            external_surface_heat_w=heating_thermal_w,
        )
        actual_battery_power = max(-step.diagnostics.terminal_power_w, 0.0)
        dc_bus_power = actual_battery_power + auxiliary_power
        grid_power = dc_bus_power / config.charger_efficiency
        charger_loss = grid_power - dc_bus_power
        ledger_residual = grid_power - charger_loss - auxiliary_power - actual_battery_power
        aging = incremental_aging(
            step.diagnostics.current_a,
            state.core_temp_c,
            state.soc,
            config.dt_s,
            capacity_ah,
            aging_parameters,
        )
        total_damage += aging.total_damage
        state = step.state
        rows.append({
            "time_s": time_s,
            "soc": state.soc,
            "battery_core_temp_c": state.core_temp_c,
            "battery_surface_temp_c": state.surface_temp_c,
            "requested_station_power_w": config.station_power_w,
            "grid_power_w": grid_power,
            "charger_loss_w": charger_loss,
            "dc_bus_power_w": dc_bus_power,
            "auxiliary_power_w": auxiliary_power,
            "accepted_battery_power_w": actual_battery_power,
            "curtailed_station_power_w": max(config.station_power_w - grid_power, 0.0),
            "battery_current_a": step.diagnostics.current_a,
            "battery_heat_w": step.diagnostics.heat_generation_w,
            "temperature_factor": acceptance.temperature_factor,
            "soc_factor": acceptance.soc_factor,
            "limiting_reason": limiting_reason,
            "power_ledger_residual_w": ledger_residual,
            "incremental_aging_damage": aging.total_damage,
            "cumulative_aging_damage": total_damage,
        })
        if state.core_temp_c > config.hard_max_temp_c:
            status = "safe_stop_temperature"
            break

    frame = pd.DataFrame(rows)
    charge_time = len(frame) * config.dt_s
    grid_energy = float(frame["grid_power_w"].sum() * config.dt_s / 3.6e6) if len(frame) else 0.0
    battery_energy = float(frame["accepted_battery_power_w"].sum() * config.dt_s / 3.6e6) if len(frame) else 0.0
    peak_temp = float(frame["battery_core_temp_c"].max()) if len(frame) else initial_state.core_temp_c
    return FastChargeResult(
        status, state, frame, charge_time, grid_energy, battery_energy, peak_temp, total_damage
    )
