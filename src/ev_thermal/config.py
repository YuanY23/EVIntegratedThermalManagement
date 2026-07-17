"""Configuration loading with unit-aware, validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: int = 5
    default_duration_s: int = 1800


@dataclass(frozen=True)
class PredictionConfig:
    horizon_s: int = 300
    history_s: int = 300
    hidden_size: int = 48
    num_layers: int = 2
    batch_size: int = 64
    max_epochs: int = 35
    patience: int = 6
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class VehicleConfig:
    mass_kg: float = 1950.0
    frontal_area_m2: float = 2.35
    drag_coefficient: float = 0.27
    rolling_resistance: float = 0.0105
    wheel_radius_m: float = 0.34
    final_drive_ratio: float = 9.1
    drivetrain_efficiency: float = 0.97
    max_traction_power_kw: float = 180.0
    max_regen_power_kw: float = 80.0


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 75.0
    nominal_voltage_v: float = 380.0
    nominal_resistance_ohm: float = 0.065
    core_heat_capacity_j_k: float = 380_000.0
    surface_heat_capacity_j_k: float = 95_000.0
    core_surface_conductance_w_k: float = 380.0
    initial_soc: float = 0.85
    initial_temp_c: float = 25.0


@dataclass(frozen=True)
class ControlConfig:
    cabin_setpoint_c: float = 24.0
    battery_cooling_on_c: float = 34.0
    battery_high_cooling_c: float = 39.0
    battery_heating_on_c: float = 12.0
    motor_cooling_on_c: float = 70.0
    inverter_cooling_on_c: float = 65.0


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    simulation: SimulationConfig
    prediction: PredictionConfig
    vehicle: VehicleConfig
    battery: BatteryConfig
    control: ControlConfig
    outputs: dict[str, Any]


def _filtered(cls, values: dict[str, Any] | None):
    values = values or {}
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load YAML and reject grids that cannot represent the 300 s horizon exactly."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "default_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    simulation = _filtered(SimulationConfig, raw.get("simulation"))
    prediction = _filtered(PredictionConfig, raw.get("prediction"))
    if simulation.dt_s <= 0:
        raise ValueError("simulation.dt_s must be positive")
    if prediction.horizon_s % simulation.dt_s != 0:
        raise ValueError("prediction horizon must be divisible by simulation.dt_s")
    if prediction.horizon_s != 300:
        raise ValueError("the research design requires a 300 second forecast horizon")
    return ProjectConfig(
        seed=int(raw.get("seed", 42)),
        simulation=simulation,
        prediction=prediction,
        vehicle=_filtered(VehicleConfig, raw.get("vehicle")),
        battery=_filtered(BatteryConfig, raw.get("battery")),
        control=_filtered(ControlConfig, raw.get("control")),
        outputs=dict(raw.get("outputs", {"equivalent_battery_kwh": 75.0})),
    )


def set_seed(seed: int) -> None:
    """Set numerical and neural-network seeds used by all reproducible pipelines."""
    np.random.seed(seed)
    torch.manual_seed(seed)

