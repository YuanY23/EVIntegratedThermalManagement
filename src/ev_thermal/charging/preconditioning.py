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


@dataclass(frozen=True)
class PreconditioningPolicy:
    start_before_arrival_s: float
    target_temp_c: float
    max_thermal_power_w: float

    def __post_init__(self) -> None:
        if self.start_before_arrival_s < 0 or self.max_thermal_power_w <= 0:
            raise ValueError("Policy lead time must be non-negative and power must be positive")
        if not -10.0 <= self.target_temp_c <= 45.0:
            raise ValueError("Policy target temperature is outside the engineering search range")


def _preview_blocked_command(
    battery_temp_c: float, preview: RoutePreview
) -> PreconditioningCommand | None:
    """Return the common safety-gate command, or allow strategy evaluation."""
    if not preview.route_active:
        return PreconditioningCommand(False, 0.0, battery_temp_c, "route_inactive")
    if not preview.valid or preview.remaining_time_s <= 0:
        return PreconditioningCommand(False, 0.0, battery_temp_c, "preview_unavailable")
    if preview.predicted_arrival_soc <= 0.10:
        return PreconditioningCommand(False, 0.0, battery_temp_c, "soc_reserve")
    return None


def rule_preconditioning_command(battery_temp_c: float, preview: RoutePreview,
                                 heat_capacity_j_k: float = 475_000.0,
                                 max_heating_w: float = 5_000.0,
                                 max_cooling_w: float = 6_000.0) -> PreconditioningCommand:
    """Select bounded heating/cooling only when route preview supports a benefit."""
    blocked = _preview_blocked_command(battery_temp_c, preview)
    if blocked is not None:
        return blocked
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


def policy_preconditioning_command(battery_temp_c: float, preview: RoutePreview,
                                   policy: PreconditioningPolicy,
                                   heat_capacity_j_k: float = 475_000.0) -> PreconditioningCommand:
    """Apply one bounded optimization candidate without bypassing preview safety gates."""
    blocked = _preview_blocked_command(battery_temp_c, preview)
    if blocked is not None:
        return blocked
    if preview.remaining_time_s > policy.start_before_arrival_s:
        return PreconditioningCommand(False, 0.0, policy.target_temp_c, "waiting_for_start")
    error = policy.target_temp_c - battery_temp_c
    if abs(error) <= 0.5:
        return PreconditioningCommand(False, 0.0, policy.target_temp_c, "target_deadband")
    required = heat_capacity_j_k * abs(error) / max(preview.remaining_time_s, 1.0)
    power = min(max(required, 500.0), policy.max_thermal_power_w)
    return PreconditioningCommand(
        True,
        power if error > 0 else -power,
        policy.target_temp_c,
        "optimized_preheat" if error > 0 else "optimized_precool",
    )
