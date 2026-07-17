"""Pack-level equivalent-circuit and two-node thermal battery model."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BatteryParameters:
    capacity_kwh: float = 75.0
    nominal_voltage_v: float = 380.0
    nominal_resistance_ohm: float = 0.065
    core_heat_capacity_j_k: float = 380_000.0
    surface_heat_capacity_j_k: float = 95_000.0
    core_surface_conductance_w_k: float = 380.0
    entropy_coefficient_v_k: float = -0.035
    max_discharge_power_w: float = 200_000.0
    max_charge_power_w: float = 100_000.0


@dataclass(frozen=True)
class BatteryState:
    soc: float
    core_temp_c: float
    surface_temp_c: float


@dataclass(frozen=True)
class BatteryDiagnostics:
    current_a: float
    terminal_power_w: float
    heat_generation_w: float
    core_surface_heat_w: float
    coolant_heat_w: float
    resistance_ohm: float
    external_heat_w: float


@dataclass(frozen=True)
class BatteryStep:
    state: BatteryState
    diagnostics: BatteryDiagnostics


class BatteryModel:
    """Positive terminal power and current denote pack discharge."""

    def __init__(self, params: BatteryParameters | None = None):
        self.params = params or BatteryParameters()

    def ocv_v(self, soc: float) -> float:
        # Smooth pack OCV approximation, suitable for system energy studies.
        x = float(np.clip(soc, 0.02, 0.98))
        return self.params.nominal_voltage_v * (0.88 + 0.20 * x - 0.04 * x**2)

    def resistance_ohm(self, soc: float, temperature_c: float) -> float:
        # Arrhenius-like cold penalty and end-of-SOC penalty. Bounds prevent an
        # empirical lumped model from being extrapolated into nonphysical values.
        cold_factor = math.exp(np.clip(0.018 * (25.0 - temperature_c), -0.5, 1.2))
        soc_factor = 1.0 + 1.8 * max(0.15 - soc, 0.0) ** 2 / 0.15**2
        return self.params.nominal_resistance_ohm * cold_factor * soc_factor

    def _current_for_power(self, power_w: float, ocv_v: float, resistance_ohm: float) -> float:
        # Terminal relation P=(OCV-I*R)I. The low-current root is the stable,
        # physically relevant solution; power is clipped before the discriminant.
        if power_w >= 0:
            maximum = min(self.params.max_discharge_power_w, 0.95 * ocv_v**2 / (4 * resistance_ohm))
            p = min(power_w, maximum)
        else:
            p = max(power_w, -self.params.max_charge_power_w)
        discriminant = max(ocv_v**2 - 4.0 * resistance_ohm * p, 0.0)
        return (ocv_v - math.sqrt(discriminant)) / (2.0 * resistance_ohm)

    def step(self, state: BatteryState, terminal_power_w: float, coolant_temp_c: float,
             coolant_ua_w_per_k: float, dt_s: float,
             external_surface_heat_w: float = 0.0) -> BatteryStep:
        p = self.params
        ocv = self.ocv_v(state.soc)
        resistance = self.resistance_ohm(state.soc, state.core_temp_c)
        current = self._current_for_power(terminal_power_w, ocv, resistance)
        actual_terminal_power = (ocv - current * resistance) * current
        # Bernardi heat: irreversible Joule heat plus reversible entropic heat.
        # With discharge-positive current, -I*T*dOCV/dT is positive for the
        # negative pack entropy coefficient used here.
        heat_generation = current**2 * resistance - current * (state.core_temp_c + 273.15) * p.entropy_coefficient_v_k
        core_surface_heat = p.core_surface_conductance_w_k * (state.core_temp_c - state.surface_temp_c)
        coolant_heat = max(coolant_ua_w_per_k, 0.0) * (state.surface_temp_c - coolant_temp_c)
        core_temp = state.core_temp_c + dt_s * (heat_generation - core_surface_heat) / p.core_heat_capacity_j_k
        # A coolant/PTC battery heater deposits heat at the pack boundary, hence
        # it enters the surface node rather than the electrochemical core source.
        external_heat = max(external_surface_heat_w, 0.0)
        surface_temp = state.surface_temp_c + dt_s * (core_surface_heat - coolant_heat + external_heat) / p.surface_heat_capacity_j_k
        # OCV*I is electrochemical power. Coulomb counting is obtained by using
        # nominal pack ampere-hours = energy/voltage.
        capacity_ah = p.capacity_kwh * 1000.0 / p.nominal_voltage_v
        soc = np.clip(state.soc - current * dt_s / (capacity_ah * 3600.0), 0.0, 1.0)
        return BatteryStep(
            BatteryState(float(soc), float(core_temp), float(surface_temp)),
            BatteryDiagnostics(current, actual_terminal_power, heat_generation, core_surface_heat,
                               coolant_heat, resistance, external_heat),
        )
