"""Battery cold-plate convection and single-pass coolant heating."""

from dataclasses import dataclass
import math

from .fluid import FluidProperties


@dataclass(frozen=True)
class ColdPlateResult:
    heat_transfer_w: float
    coolant_out_temp_c: float
    effectiveness: float
    reynolds: float
    nusselt: float
    h_w_m2k: float
    pressure_drop_pa: float


class ColdPlate:
    def __init__(self, channel_count: int = 12, hydraulic_diameter_m: float = 0.0035,
                 channel_area_m2: float = 8.0e-6, channel_length_m: float = 0.7,
                 heat_transfer_area_m2: float = 1.8, wall_resistance_k_w: float = 0.002):
        self.channel_count = channel_count
        self.hydraulic_diameter_m = hydraulic_diameter_m
        self.channel_area_m2 = channel_area_m2
        self.channel_length_m = channel_length_m
        self.heat_transfer_area_m2 = heat_transfer_area_m2
        self.wall_resistance_k_w = wall_resistance_k_w

    def exchange(self, surface_temp_c: float, coolant_in_temp_c: float,
                 mass_flow_kg_s: float, fluid: FluidProperties) -> ColdPlateResult:
        mdot = max(mass_flow_kg_s, 0.0)
        if mdot < 1e-8:
            return ColdPlateResult(0, coolant_in_temp_c, 0, 0, 0, 0, 0)
        per_channel = mdot / self.channel_count
        velocity = per_channel / (fluid.density_kg_m3 * self.channel_area_m2)
        reynolds = fluid.density_kg_m3 * velocity * self.hydraulic_diameter_m / fluid.viscosity_pa_s
        pr = fluid.prandtl
        if reynolds < 2300:
            nusselt = 4.36  # fully developed constant-heat-flux rectangular channel approximation
            friction = 64.0 / max(reynolds, 1.0)
        else:
            # Gnielinski correlation is used in its standard turbulent range.
            friction = (0.79 * math.log(reynolds) - 1.64) ** -2
            nusselt = ((friction / 8) * (reynolds - 1000) * pr /
                       (1 + 12.7 * math.sqrt(friction / 8) * (pr ** (2 / 3) - 1)))
        h = nusselt * fluid.conductivity_w_mk / self.hydraulic_diameter_m
        convective_resistance = 1.0 / max(h * self.heat_transfer_area_m2, 1e-9)
        ua = 1.0 / (self.wall_resistance_k_w + convective_resistance)
        capacity = mdot * fluid.cp_j_kgk
        effectiveness = 1.0 - math.exp(-ua / capacity)
        heat = effectiveness * capacity * (surface_temp_c - coolant_in_temp_c)
        outlet = coolant_in_temp_c + heat / capacity
        dynamic_pressure = 0.5 * fluid.density_kg_m3 * velocity**2
        pressure = (friction * self.channel_length_m / self.hydraulic_diameter_m + 2.5) * dynamic_pressure
        return ColdPlateResult(heat, outlet, effectiveness, reynolds, nusselt, h, pressure)

