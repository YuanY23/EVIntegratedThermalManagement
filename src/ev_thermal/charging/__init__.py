"""Fast charging, arrival preconditioning, aging, and optimization."""

from .aging import AgingParameters, AgingStep, incremental_aging
from .fast_charge import FastChargeConfig, FastChargeResult, simulate_fast_charge
from .preconditioning import (
    PreconditioningCommand,
    PreconditioningPolicy,
    RoutePreview,
    rule_preconditioning_command,
)

__all__ = [
    "AgingParameters",
    "AgingStep",
    "FastChargeConfig",
    "FastChargeResult",
    "PreconditioningCommand",
    "PreconditioningPolicy",
    "RoutePreview",
    "incremental_aging",
    "rule_preconditioning_command",
    "simulate_fast_charge",
]
