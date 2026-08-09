"""Drive-to-station preconditioning followed by a constrained fast-charge event."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..charging.fast_charge import FastChargeConfig, FastChargeResult, simulate_fast_charge
from ..charging.preconditioning import RoutePreview, rule_preconditioning_command
from ..components.battery import BatteryModel, BatteryParameters, BatteryState


@dataclass(frozen=True)
class ChargingScenario:
    name: str
    ambient_temp_c: float
    initial_battery_temp_c: float
    initial_soc: float
    route_time_s: int
    target_soc: float
    station_power_w: float
    route_preview_valid: bool = True
    route_active: bool = True
    traction_power_w: float = 18_000.0


@dataclass(frozen=True)
class TripChargeResult:
    scenario: str
    strategy: str
    arrival_core_temp_c: float
    arrival_surface_temp_c: float
    arrival_soc: float
    preconditioning_energy_kwh: float
    charge_time_s: float
    grid_energy_kwh: float
    final_soc: float
    peak_core_temp_c: float
    relative_aging_damage: float
    preconditioning_status: str
    charge_status: str
    route_timeseries: pd.DataFrame
    charge_timeseries: pd.DataFrame


def _simulate_route(scenario: ChargingScenario, strategy: str, dt_s: float = 5.0):
    if strategy not in {"none", "rule"}:
        raise ValueError("strategy must be 'none' or 'rule'")
    model = BatteryModel(replace(BatteryParameters(), max_discharge_power_w=220_000.0))
    state = BatteryState(
        scenario.initial_soc,
        scenario.initial_battery_temp_c,
        scenario.initial_battery_temp_c,
    )
    rows = []
    preconditioning_energy_j = 0.0
    last_reason = "strategy_none"
    for time_s in np.arange(0.0, scenario.route_time_s, dt_s):
        remaining = max(scenario.route_time_s - time_s, dt_s)
        preview = RoutePreview(
            remaining,
            max(state.soc - scenario.traction_power_w * remaining / 3.6e9 / 75.0, 0.0),
            scenario.ambient_temp_c,
            valid=scenario.route_preview_valid,
            route_active=scenario.route_active,
        )
        command = (
            rule_preconditioning_command(state.core_temp_c, preview)
            if strategy == "rule"
            else None
        )
        thermal_power = command.thermal_power_w if command is not None else 0.0
        last_reason = command.reason if command is not None else "strategy_none"
        heating = max(thermal_power, 0.0)
        cooling = max(-thermal_power, 0.0)
        thermal_aux = heating / 0.95 + cooling / 3.0
        pump_power = 150.0 if abs(thermal_power) > 0 else 0.0
        auxiliary_power = thermal_aux + pump_power
        requested_traction = scenario.traction_power_w * (
            0.85 + 0.15 * np.sin(2 * np.pi * time_s / 300.0)
        )
        available_traction = max(model.params.max_discharge_power_w - auxiliary_power, 0.0)
        actual_traction = min(requested_traction, available_traction)
        passive_ua = 35.0
        cooling_ua = cooling / max(state.surface_temp_c - 12.0, 5.0)
        total_ua = passive_ua + cooling_ua
        coolant_temp = (
            passive_ua * scenario.ambient_temp_c + cooling_ua * 12.0
        ) / max(total_ua, 1e-9)
        step = model.step(
            state,
            actual_traction + auxiliary_power,
            coolant_temp,
            total_ua,
            dt_s,
            external_surface_heat_w=heating,
        )
        closure = step.diagnostics.terminal_power_w - actual_traction - auxiliary_power
        preconditioning_energy_j += auxiliary_power * dt_s
        state = step.state
        rows.append({
            "time_s": time_s,
            "soc": state.soc,
            "battery_core_temp_c": state.core_temp_c,
            "battery_surface_temp_c": state.surface_temp_c,
            "requested_traction_power_w": requested_traction,
            "available_traction_power_w": available_traction,
            "actual_traction_power_w": actual_traction,
            "preconditioning_thermal_power_w": thermal_power,
            "preconditioning_aux_power_w": auxiliary_power,
            "battery_terminal_power_w": step.diagnostics.terminal_power_w,
            "power_closure_residual_w": closure,
            "preconditioning_reason": last_reason,
        })
    return state, preconditioning_energy_j / 3.6e6, pd.DataFrame(rows), last_reason


def simulate_trip_charge(scenario: ChargingScenario, strategy: str) -> TripChargeResult:
    arrival, preconditioning_energy, route_frame, status = _simulate_route(scenario, strategy)
    charge_config = FastChargeConfig(
        station_power_w=scenario.station_power_w,
        target_soc=scenario.target_soc,
    )
    charge: FastChargeResult = simulate_fast_charge(
        arrival, scenario.ambient_temp_c, charge_config
    )
    peak_route = float(route_frame["battery_core_temp_c"].max()) if len(route_frame) else arrival.core_temp_c
    return TripChargeResult(
        scenario.name,
        strategy,
        arrival.core_temp_c,
        arrival.surface_temp_c,
        arrival.soc,
        preconditioning_energy,
        charge.charge_time_s,
        charge.grid_energy_kwh,
        charge.final_state.soc,
        max(peak_route, charge.peak_core_temp_c),
        charge.relative_aging_damage,
        status,
        charge.status,
        route_frame,
        charge.timeseries,
    )


def charging_scenario_names() -> list[str]:
    return ["cold_arrival", "mild_arrival", "hot_arrival"]


def make_charging_scenario(name: str) -> ChargingScenario:
    definitions = {
        "cold_arrival": ChargingScenario(name, -15.0, -15.0, 0.35, 2400, 0.80, 220_000.0),
        "mild_arrival": ChargingScenario(name, 22.0, 24.0, 0.35, 1800, 0.80, 250_000.0),
        "hot_arrival": ChargingScenario(name, 40.0, 45.0, 0.40, 1800, 0.75, 220_000.0),
    }
    if name not in definitions:
        raise ValueError(f"Unknown charging scenario: {name}")
    return definitions[name]
