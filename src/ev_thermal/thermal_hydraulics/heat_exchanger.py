"""Epsilon-NTU heat exchangers for coolant loops and ambient air."""

from dataclasses import dataclass
import math

from .fluid import glycol_properties


def epsilon_ntu(ua_w_per_k: float, hot_capacity_w_per_k: float, cold_capacity_w_per_k: float) -> float:
    """Counter-flow effectiveness for two finite-capacity streams."""
    c_min = max(min(hot_capacity_w_per_k, cold_capacity_w_per_k), 1e-9)
    c_max = max(max(hot_capacity_w_per_k, cold_capacity_w_per_k), 1e-9)
    cr = c_min / c_max
    ntu = max(ua_w_per_k, 0.0) / c_min
    if abs(1.0 - cr) < 1e-8:
        return ntu / (1.0 + ntu)
    exp_term = math.exp(-ntu * (1.0 - cr))
    return (1.0 - exp_term) / (1.0 - cr * exp_term)


@dataclass(frozen=True)
class RadiatorResult:
    heat_rejected_w: float
    coolant_out_temp_c: float
    effectiveness: float
    fan_power_w: float
    air_mass_flow_kg_s: float


class Radiator:
    def __init__(self, ua_nominal_w_k: float = 850.0, fan_max_power_w: float = 420.0):
        self.ua_nominal_w_k = ua_nominal_w_k
        self.fan_max_power_w = fan_max_power_w

    def exchange(self, coolant_in_temp_c: float, ambient_temp_c: float,
                 coolant_mass_flow_kg_s: float, vehicle_speed_mps: float,
                 fan_fraction: float) -> RadiatorResult:
        fluid = glycol_properties(coolant_in_temp_c)
        fan = min(max(fan_fraction, 0.0), 1.0)
        # Ram air and fan air are combined as independent flow contributions.
        air_flow = 0.12 + 0.035 * max(vehicle_speed_mps, 0.0) + 0.85 * fan
        coolant_capacity = max(coolant_mass_flow_kg_s, 0.0) * fluid.cp_j_kgk
        air_capacity = air_flow * 1006.0
        if coolant_capacity < 1e-6:
            return RadiatorResult(0, coolant_in_temp_c, 0, self.fan_max_power_w * fan**3, air_flow)
        ua = self.ua_nominal_w_k * (0.25 + 0.75 * (air_flow / 1.82) ** 0.65)
        effectiveness = epsilon_ntu(ua, coolant_capacity, air_capacity)
        c_min = min(coolant_capacity, air_capacity)
        heat = effectiveness * c_min * (coolant_in_temp_c - ambient_temp_c)
        outlet = coolant_in_temp_c - heat / coolant_capacity
        # Fan affinity law: shaft/electrical power is approximately cubic in speed.
        return RadiatorResult(heat, outlet, effectiveness, self.fan_max_power_w * fan**3, air_flow)


@dataclass(frozen=True)
class LiquidHeatExchangerResult:
    heat_hot_to_cold_w: float
    hot_out_temp_c: float
    cold_out_temp_c: float
    effectiveness: float


def liquid_liquid_exchange(hot_in_temp_c: float, cold_in_temp_c: float,
                           hot_mass_flow_kg_s: float, cold_mass_flow_kg_s: float,
                           ua_w_per_k: float = 900.0) -> LiquidHeatExchangerResult:
    hot = glycol_properties(hot_in_temp_c)
    cold = glycol_properties(cold_in_temp_c)
    ch = max(hot_mass_flow_kg_s, 0.0) * hot.cp_j_kgk
    cc = max(cold_mass_flow_kg_s, 0.0) * cold.cp_j_kgk
    if min(ch, cc) < 1e-6:
        return LiquidHeatExchangerResult(0, hot_in_temp_c, cold_in_temp_c, 0)
    eff = epsilon_ntu(ua_w_per_k, ch, cc)
    heat = eff * min(ch, cc) * (hot_in_temp_c - cold_in_temp_c)
    return LiquidHeatExchangerResult(heat, hot_in_temp_c - heat / ch, cold_in_temp_c + heat / cc, eff)

