"""Hydraulic valve loss with explicit blocked-state semantics."""

from __future__ import annotations

from dataclasses import dataclass

from .fluid import FluidProperties


@dataclass(frozen=True)
class ValveElement:
    name: str
    full_open_resistance_pa_per_kg2_s2: float
    opening_fraction: float
    minimum_opening_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Valve name is required")
        if self.full_open_resistance_pa_per_kg2_s2 < 0:
            raise ValueError("Valve resistance must be non-negative")
        if not 0.0 <= self.opening_fraction <= 1.0:
            raise ValueError("Valve opening_fraction must be within [0, 1]")

    @property
    def is_blocked(self) -> bool:
        return self.opening_fraction < self.minimum_opening_fraction

    def pressure_drop_pa(self, mass_flow_kg_s: float, fluid: FluidProperties) -> float:
        del fluid
        if self.is_blocked:
            return float("inf") if mass_flow_kg_s > 0 else 0.0
        effective_resistance = self.full_open_resistance_pa_per_kg2_s2 / self.opening_fraction**2
        return effective_resistance * max(mass_flow_kg_s, 0.0) ** 2
