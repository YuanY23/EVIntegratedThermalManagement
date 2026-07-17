"""Representative multi-environment EV thermal-management scenarios."""

from dataclasses import dataclass

import numpy as np

from ..vehicle.drive_cycles import built_in_speed


@dataclass(frozen=True)
class Scenario:
    name: str
    time_s: np.ndarray
    speed_mps: np.ndarray
    grade_rad: np.ndarray
    ambient_temp_c: np.ndarray
    solar_w_m2: np.ndarray
    occupants: np.ndarray
    initial_battery_temp_c: float
    initial_cabin_temp_c: float
    initial_soc: float


def make_scenario(name: str, duration_s: int = 1800, dt_s: int = 5, seed: int = 42) -> Scenario:
    rng = np.random.default_rng(seed)
    time_s = np.arange(0, duration_s + dt_s, dt_s, dtype=float)
    definitions = {
        "urban_hot": ("urban", 40.0, 38.0, 37.0, 750.0),
        "cold_start": ("urban", -20.0, -18.0, -15.0, 100.0),
        "highway_hot": ("highway", 38.0, 35.0, 34.0, 650.0),
        "aggressive": ("aggressive", 30.0, 30.0, 28.0, 450.0),
        "hill_high_load": ("hill", 25.0, 27.0, 24.0, 400.0),
        "mixed_mild": ("mixed", 15.0, 20.0, 18.0, 250.0),
    }
    if name not in definitions:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(definitions)}")
    cycle, ambient, battery_temp, cabin_temp, solar = definitions[name]
    speed = built_in_speed(cycle, time_s, rng)
    grade = np.zeros_like(time_s)
    if name == "hill_high_load":
        grade = np.deg2rad(4.5 + 1.5 * np.sin(2 * np.pi * time_s / 400))
    ambient_series = ambient + 1.5 * np.sin(2 * np.pi * time_s / max(duration_s, 600))
    solar_series = np.clip(solar * (0.9 + 0.1 * np.sin(2 * np.pi * time_s / 900)), 0, None)
    return Scenario(name, time_s, speed, grade, ambient_series, solar_series,
                    np.full_like(time_s, 2.0), battery_temp, cabin_temp, 0.85)


def scenario_names() -> list[str]:
    return ["urban_hot", "cold_start", "highway_hot", "aggressive", "hill_high_load", "mixed_mild"]

