"""Route-preview contract and deterministic arrival-temperature baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutePreview:
    remaining_time_s: float
    predicted_arrival_soc: float
    ambient_temp_c: float
    valid: bool = True
    route_active: bool = True


@dataclass(frozen=True)
class PreconditioningCommand:
    active: bool
    thermal_power_w: float
    target_temp_c: float
    reason: str


def rule_preconditioning_command(battery_temp_c: float, preview: RoutePreview,
                                 heat_capacity_j_k: float = 475_000.0,
                                 max_heating_w: float = 5_000.0,
                                 max_cooling_w: float = 6_000.0) -> PreconditioningCommand:
    """Select bounded heating/cooling only when route preview supports a benefit."""
    if not preview.route_active:
        return PreconditioningCommand(False, 0.0, battery_temp_c, "route_inactive")
    if not preview.valid or preview.remaining_time_s <= 0:
        return PreconditioningCommand(False, 0.0, battery_temp_c, "preview_unavailable")
    if preview.predicted_arrival_soc <= 0.10:
        return PreconditioningCommand(False, 0.0, battery_temp_c, "soc_reserve")
    if battery_temp_c < 18.0:
        target = 20.0
        required = heat_capacity_j_k * (target - battery_temp_c) / preview.remaining_time_s
        power = min(max(required, 750.0), max_heating_w)
        return PreconditioningCommand(True, power, target, "cold_arrival_preheat")
    if battery_temp_c > 38.0:
        target = 30.0
        required = heat_capacity_j_k * (battery_temp_c - target) / preview.remaining_time_s
        power = -min(max(required, 1_000.0), max_cooling_w)
        return PreconditioningCommand(True, power, target, "hot_arrival_precool")
    return PreconditioningCommand(False, 0.0, battery_temp_c, "temperature_in_benefit_band")
