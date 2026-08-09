"""Integrated thermal, comfort, hydraulic, and vehicle energy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulation_metrics(frame: pd.DataFrame, battery_capacity_kwh: float = 75.0) -> dict[str, float]:
    dt = float(np.median(np.diff(frame["time_s"]))) if len(frame) > 1 else 0.0
    distance_km = float(np.trapz(frame["speed_mps"], frame["time_s"]) / 1000.0)
    battery_energy_kwh = float(np.sum(np.maximum(frame["battery_terminal_power_w"], 0)) * dt / 3.6e6)
    aux_energy_kwh = float(np.sum(frame["auxiliary_power_w"]) * dt / 3.6e6)
    net_energy_kwh = float(np.sum(frame["battery_total_power_w"]) * dt / 3.6e6)
    consumption = net_energy_kwh / max(distance_km, 1e-9) * 100.0
    balance_input = float(np.sum(np.abs(frame["battery_total_power_w"])) * dt)
    residual = float(np.sum(np.abs(frame["energy_balance_residual_w"])) * dt)
    thermal_residual = float(np.sum(np.abs(frame["thermal_balance_residual_w"])) * dt)
    thermal_throughput = float(np.sum(
        np.abs(frame["battery_heat_w"]) + np.abs(frame["powertrain_heat_w"]) +
        np.abs(frame["cabin_load_w"]) + np.abs(frame["hvac_heat_w"]) +
        np.abs(frame["radiator_heat_w"]) + np.abs(frame["battery_chiller_heat_w"])) * dt)
    return {
        "distance_km": distance_km,
        "battery_energy_kwh": battery_energy_kwh,
        "auxiliary_energy_kwh": aux_energy_kwh,
        "net_energy_kwh": net_energy_kwh,
        "consumption_kwh_100km": consumption,
        "equivalent_range_km": battery_capacity_kwh / max(consumption, 1e-9) * 100.0,
        "max_battery_temp_c": float(frame["battery_core_temp_c"].max()),
        "max_motor_temp_c": float(frame["motor_temp_c"].max()),
        "cabin_comfort_rmse_c": float(np.sqrt(np.mean(frame["cabin_temp_error_c"] ** 2))),
        "waste_heat_recovered_kwh": float(np.sum(frame["waste_heat_recovered_w"]) * dt / 3.6e6),
        "energy_balance_error_pct": 100.0 * residual / max(balance_input, 1.0),
        "thermal_balance_error_pct": 100.0 * thermal_residual / max(thermal_throughput, 1.0),
        "pump_energy_kwh": float(np.sum(frame["pump_power_w"]) * dt / 3.6e6),
        "mean_battery_flow_kg_s": float(frame["battery_flow_kg_s"].mean()),
        "mean_powertrain_flow_kg_s": float(frame["powertrain_flow_kg_s"].mean()),
        "max_battery_pressure_drop_kpa": float(frame["battery_system_pressure_drop_pa"].max() / 1000.0),
        "max_powertrain_pressure_drop_kpa": float(frame["powertrain_system_pressure_drop_pa"].max() / 1000.0),
        "hydraulic_solver_failures": float(frame["hydraulic_solver_failure_count"].max()),
        "liquid_hx_energy_kwh": float(
            np.sum(np.abs(frame["liquid_hx_drive_to_battery_heat_w"])) * dt / 3.6e6
        ),
    }
