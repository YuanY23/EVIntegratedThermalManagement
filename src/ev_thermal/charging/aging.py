"""Relative Arrhenius/Ah-throughput battery damage accumulator."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AgingParameters:
    reference_temperature_c: float = 25.0
    activation_energy_j_mol: float = 24_000.0
    gas_constant_j_mol_k: float = 8.314
    soc_stress_coefficient: float = 1.8
    c_rate_exponent: float = 1.25
    cycle_scale: float = 1.0e-4
    calendar_scale_per_day: float = 2.0e-6

    def __post_init__(self) -> None:
        if self.reference_temperature_c <= -273.15:
            raise ValueError("reference_temperature_c must be above absolute zero")
        if min(self.activation_energy_j_mol, self.gas_constant_j_mol_k,
               self.c_rate_exponent, self.cycle_scale, self.calendar_scale_per_day) <= 0:
            raise ValueError("Aging coefficients must be positive")
        if self.soc_stress_coefficient < 0:
            raise ValueError("soc_stress_coefficient must be non-negative")


@dataclass(frozen=True)
class AgingStep:
    throughput_ah: float
    cycle_damage: float
    calendar_damage: float
    total_damage: float
    temperature_factor: float
    soc_factor: float
    c_rate: float


def incremental_aging(current_a: float, temperature_c: float, soc: float, dt_s: float,
                      capacity_ah: float, parameters: AgingParameters | None = None) -> AgingStep:
    """Return a dimensionless relative damage increment for strategy comparison."""
    parameters = parameters or AgingParameters()
    if dt_s < 0 or capacity_ah <= 0 or temperature_c <= -273.15 or not 0 <= soc <= 1:
        raise ValueError("Invalid aging state or duration")
    if dt_s == 0:
        return AgingStep(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    temperature_k = temperature_c + 273.15
    reference_k = parameters.reference_temperature_c + 273.15
    temperature_factor = math.exp(
        parameters.activation_energy_j_mol / parameters.gas_constant_j_mol_k
        * (1.0 / reference_k - 1.0 / temperature_k)
    )
    soc_factor = math.exp(parameters.soc_stress_coefficient * max(soc - 0.5, 0.0))
    throughput_ah = abs(current_a) * dt_s / 3600.0
    c_rate = abs(current_a) / capacity_ah
    cycle_damage = (
        parameters.cycle_scale
        * (throughput_ah / capacity_ah)
        * temperature_factor
        * soc_factor
        * (1.0 + c_rate**parameters.c_rate_exponent)
    )
    calendar_damage = (
        parameters.calendar_scale_per_day * dt_s / 86_400.0
        * temperature_factor * soc_factor
    )
    return AgingStep(
        throughput_ah,
        cycle_damage,
        calendar_damage,
        cycle_damage + calendar_damage,
        temperature_factor,
        soc_factor,
        c_rate,
    )
