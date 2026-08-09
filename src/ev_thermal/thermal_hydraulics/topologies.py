"""Hydraulic designs and thermal coupling for three vehicle loop architectures."""

from __future__ import annotations

from dataclasses import dataclass

from .fluid import glycol_properties
from .heat_exchanger import Radiator, liquid_liquid_exchange
from .network import ColdPlateElement, PipeElement, QuadraticLoss, SeriesHydraulicNetwork
from .pump import Pump
from .valve import ValveElement


ARCHITECTURES = (
    "independent_dual_loop",
    "coupled_dual_loop",
    "shared_heat_sink",
)


def architecture_names() -> list[str]:
    return list(ARCHITECTURES)


def validate_architecture(name: str) -> str:
    if name not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    return name


@dataclass(frozen=True)
class HydraulicDesign:
    local_resistance_scale: float = 1.0
    pump_scale: float = 1.0
    radiator_ua_scale: float = 1.0
    liquid_hx_ua_w_k: float = 900.0

    def __post_init__(self) -> None:
        if min(self.local_resistance_scale, self.pump_scale, self.radiator_ua_scale) <= 0:
            raise ValueError("Hydraulic design scales must be positive")
        if self.liquid_hx_ua_w_k < 0:
            raise ValueError("liquid_hx_ua_w_k must be non-negative")


def make_pump(design: HydraulicDesign) -> Pump:
    return Pump(
        shutoff_pressure_pa=85_000.0 * design.pump_scale**2,
        max_mass_flow_kg_s=0.45 * design.pump_scale,
    )


def build_battery_network(design: HydraulicDesign) -> SeriesHydraulicNetwork:
    return SeriesHydraulicNetwork((
        PipeElement("battery_supply_return_pipe", 0.019, 7.0, 8.0),
        ColdPlateElement("battery_cold_plate"),
        QuadraticLoss("battery_radiator_core", 55_000.0 * design.local_resistance_scale),
        QuadraticLoss("battery_manifold_and_fittings", 45_000.0 * design.local_resistance_scale),
        ValveElement("battery_routing_valve", 25_000.0 * design.local_resistance_scale, 1.0),
    ))


def build_powertrain_network(design: HydraulicDesign) -> SeriesHydraulicNetwork:
    return SeriesHydraulicNetwork((
        PipeElement("powertrain_supply_return_pipe", 0.021, 6.0, 7.0),
        QuadraticLoss("motor_inverter_water_jackets", 135_000.0 * design.local_resistance_scale),
        QuadraticLoss("powertrain_radiator_core", 55_000.0 * design.local_resistance_scale),
        ValveElement("powertrain_routing_valve", 25_000.0 * design.local_resistance_scale, 1.0),
    ))


@dataclass(frozen=True)
class CrossLoopExchange:
    drive_to_battery_heat_w: float
    powertrain_out_temp_c: float
    battery_out_temp_c: float
    effectiveness: float


def cross_loop_exchange(architecture: str, powertrain_temp_c: float, battery_temp_c: float,
                        powertrain_flow_kg_s: float, battery_flow_kg_s: float,
                        ua_w_k: float) -> CrossLoopExchange:
    architecture = validate_architecture(architecture)
    if architecture != "coupled_dual_loop":
        return CrossLoopExchange(0.0, powertrain_temp_c, battery_temp_c, 0.0)
    exchange = liquid_liquid_exchange(
        powertrain_temp_c,
        battery_temp_c,
        powertrain_flow_kg_s,
        battery_flow_kg_s,
        ua_w_k,
    )
    return CrossLoopExchange(
        exchange.heat_hot_to_cold_w,
        exchange.hot_out_temp_c,
        exchange.cold_out_temp_c,
        exchange.effectiveness,
    )


@dataclass(frozen=True)
class SharedSinkResult:
    battery_heat_rejected_w: float
    powertrain_heat_rejected_w: float
    total_heat_rejected_w: float
    mixed_inlet_temp_c: float
    fan_power_w: float
    effectiveness: float


def shared_heat_sink_rejection(radiator: Radiator, battery_temp_c: float,
                               powertrain_temp_c: float, ambient_temp_c: float,
                               battery_flow_kg_s: float, powertrain_flow_kg_s: float,
                               vehicle_speed_mps: float, fan_fraction: float) -> SharedSinkResult:
    """Mix two branches through one radiator and split its heat by enthalpy potential."""
    battery_capacity = (
        max(battery_flow_kg_s, 0.0) * glycol_properties(battery_temp_c).cp_j_kgk
    )
    powertrain_capacity = (
        max(powertrain_flow_kg_s, 0.0) * glycol_properties(powertrain_temp_c).cp_j_kgk
    )
    total_capacity = battery_capacity + powertrain_capacity
    if total_capacity < 1e-9:
        result = radiator.exchange(
            ambient_temp_c, ambient_temp_c, 0.0, vehicle_speed_mps, fan_fraction
        )
        return SharedSinkResult(0.0, 0.0, 0.0, ambient_temp_c,
                                result.fan_power_w, result.effectiveness)
    mixed = (
        battery_capacity * battery_temp_c + powertrain_capacity * powertrain_temp_c
    ) / total_capacity
    result = radiator.exchange(
        mixed,
        ambient_temp_c,
        battery_flow_kg_s + powertrain_flow_kg_s,
        vehicle_speed_mps,
        fan_fraction,
    )
    common_outlet = mixed - result.heat_rejected_w / total_capacity
    battery_heat = battery_capacity * (battery_temp_c - common_outlet)
    powertrain_heat = powertrain_capacity * (powertrain_temp_c - common_outlet)
    return SharedSinkResult(
        battery_heat,
        powertrain_heat,
        result.heat_rejected_w,
        mixed,
        result.fan_power_w,
        result.effectiveness,
    )
