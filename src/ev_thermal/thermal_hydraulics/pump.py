"""Centrifugal-pump curve and hydraulic working-point solution."""

from dataclasses import dataclass

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

    def pressure_rise(self, speed_fraction: float, mass_flow_kg_s: float) -> float:
        """Return pump-curve head for a commanded speed and candidate flow."""
        speed = min(max(speed_fraction, 0.0), 1.0)
        flow = max(mass_flow_kg_s, 0.0)
        if speed < 1e-6:
            return 0.0
        zero_head_flow = self.max_mass_flow_kg_s * speed
        return max(self.shutoff_pressure_pa * speed**2 * (1.0 - (flow / zero_head_flow) ** 2), 0.0)

    def point_at(self, speed_fraction: float, mass_flow_kg_s: float,
                 pressure_rise_pa: float | None = None) -> PumpPoint:
        """Build electrical/hydraulic diagnostics at a solved working point."""
        speed = min(max(speed_fraction, 0.0), 1.0)
        flow = max(mass_flow_kg_s, 0.0)
        if speed < 1e-6 or flow < 1e-12:
            return PumpPoint(0, 0, 0, 0, 0)
        pressure = self.pressure_rise(speed, flow) if pressure_rise_pa is None else max(pressure_rise_pa, 0.0)
        zero_head_flow = self.max_mass_flow_kg_s * speed
        flow_ratio = flow / max(zero_head_flow, 1e-9)
        efficiency = max(0.18, self.nominal_efficiency * (1.0 - 0.8 * (flow_ratio - 0.65) ** 2))
        hydraulic = pressure * (flow / self.fluid_density_kg_m3)
        electrical = hydraulic / efficiency
        return PumpPoint(flow, pressure, hydraulic, electrical, efficiency)

    def working_point(self, speed_fraction: float, system_resistance_pa_per_kg2_s2: float) -> PumpPoint:
        speed = min(max(speed_fraction, 0.0), 1.0)
        if speed < 1e-6:
            return PumpPoint(0, 0, 0, 0, 0)
        # Affinity laws: shutoff head scales with n^2 and zero-head flow with n.
        zero_head_flow = self.max_mass_flow_kg_s * speed

        def residual(mdot: float) -> float:
            pump_head = self.pressure_rise(speed, mdot)
            system_head = max(system_resistance_pa_per_kg2_s2, 0.0) * mdot**2
            return pump_head - system_head

        upper_residual = residual(zero_head_flow)
        mass_flow = (
            zero_head_flow
            if abs(upper_residual) <= 1e-12
            else brentq(residual, 0.0, zero_head_flow)
        )
        pressure = max(system_resistance_pa_per_kg2_s2, 0.0) * mass_flow**2
        return self.point_at(speed, mass_flow, pressure)
