"""Centrifugal-pump curve and hydraulic working-point solution."""

from dataclasses import dataclass
import math

from scipy.optimize import brentq


@dataclass(frozen=True)
class PumpPoint:
    mass_flow_kg_s: float
    pressure_rise_pa: float
    hydraulic_power_w: float
    electrical_power_w: float
    efficiency: float


class Pump:
    def __init__(self, shutoff_pressure_pa: float = 85_000.0, max_mass_flow_kg_s: float = 0.45,
                 nominal_efficiency: float = 0.52, fluid_density_kg_m3: float = 1040.0):
        self.shutoff_pressure_pa = shutoff_pressure_pa
        self.max_mass_flow_kg_s = max_mass_flow_kg_s
        self.nominal_efficiency = nominal_efficiency
        self.fluid_density_kg_m3 = fluid_density_kg_m3

    def working_point(self, speed_fraction: float, system_resistance_pa_per_kg2_s2: float) -> PumpPoint:
        speed = min(max(speed_fraction, 0.0), 1.0)
        if speed < 1e-6:
            return PumpPoint(0, 0, 0, 0, 0)
        # Affinity laws: shutoff head scales with n^2 and zero-head flow with n.
        shutoff = self.shutoff_pressure_pa * speed**2
        zero_head_flow = self.max_mass_flow_kg_s * speed

        def residual(mdot: float) -> float:
            pump_head = shutoff * (1.0 - (mdot / zero_head_flow) ** 2)
            system_head = max(system_resistance_pa_per_kg2_s2, 0.0) * mdot**2
            return pump_head - system_head

        mass_flow = brentq(residual, 0.0, zero_head_flow * (1.0 - 1e-9))
        pressure = max(system_resistance_pa_per_kg2_s2, 0.0) * mass_flow**2
        # Efficiency falls near shutoff and runout; a floor limits the empirical
        # curve where a real pump controller would normally prohibit operation.
        flow_ratio = mass_flow / max(zero_head_flow, 1e-9)
        efficiency = max(0.18, self.nominal_efficiency * (1.0 - 0.8 * (flow_ratio - 0.65) ** 2))
        hydraulic = pressure * (mass_flow / self.fluid_density_kg_m3)
        electrical = hydraulic / efficiency
        return PumpPoint(mass_flow, pressure, hydraulic, electrical, efficiency)

