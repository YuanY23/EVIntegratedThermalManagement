"""Composable one-dimensional series hydraulic network with diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import brentq

from .cold_plate import ColdPlate
from .fluid import FluidProperties, glycol_properties
from .pipe import pipe_pressure_drop
from .pump import Pump, PumpPoint


class HydraulicElement(Protocol):
    name: str

    def pressure_drop_pa(self, mass_flow_kg_s: float, fluid: FluidProperties) -> float:
        ...


@dataclass(frozen=True)
class QuadraticLoss:
    name: str
    resistance_pa_per_kg2_s2: float

    def __post_init__(self) -> None:
        if not self.name or self.resistance_pa_per_kg2_s2 < 0:
            raise ValueError("Quadratic loss requires a name and non-negative resistance")

    def pressure_drop_pa(self, mass_flow_kg_s: float, fluid: FluidProperties) -> float:
        del fluid
        return self.resistance_pa_per_kg2_s2 * max(mass_flow_kg_s, 0.0) ** 2


@dataclass(frozen=True)
class PipeElement:
    name: str
    diameter_m: float
    length_m: float
    local_loss_coefficient: float
    roughness_m: float = 1.5e-6

    def __post_init__(self) -> None:
        if not self.name or self.diameter_m <= 0 or self.length_m < 0:
            raise ValueError("Pipe requires a name, positive diameter, and non-negative length")
        if self.local_loss_coefficient < 0 or self.roughness_m < 0:
            raise ValueError("Pipe local loss and roughness must be non-negative")

    def pressure_drop_pa(self, mass_flow_kg_s: float, fluid: FluidProperties) -> float:
        return pipe_pressure_drop(
            mass_flow_kg_s,
            self.diameter_m,
            self.length_m,
            self.local_loss_coefficient,
            fluid,
            self.roughness_m,
        ).pressure_drop_pa


@dataclass(frozen=True)
class ColdPlateElement:
    name: str
    cold_plate: ColdPlate = field(default_factory=ColdPlate)

    def pressure_drop_pa(self, mass_flow_kg_s: float, fluid: FluidProperties) -> float:
        return self.cold_plate.exchange(1.0, 0.0, mass_flow_kg_s, fluid).pressure_drop_pa


@dataclass(frozen=True)
class HydraulicSolveResult:
    status: str
    converged: bool
    point: PumpPoint
    component_pressure_drop_pa: dict[str, float]
    closure_residual_pa: float
    iterations: int
    failed_component: str | None = None
    message: str = ""


class SeriesHydraulicNetwork:
    """Solve pump head against named series-component pressure losses."""

    def __init__(self, elements: tuple[HydraulicElement, ...] | list[HydraulicElement]):
        self.elements = tuple(elements)
        if not self.elements:
            raise ValueError("Hydraulic topology must contain at least one element")
        names = [element.name for element in self.elements]
        if len(names) != len(set(names)):
            raise ValueError("Hydraulic element names must be unique")

    @staticmethod
    def _zero_point() -> PumpPoint:
        return PumpPoint(0.0, 0.0, 0.0, 0.0, 0.0)

    def solve(self, pump: Pump, speed_fraction: float,
              coolant_temp_c: float) -> HydraulicSolveResult:
        speed = min(max(float(speed_fraction), 0.0), 1.0)
        if speed < 1e-6:
            return HydraulicSolveResult("stopped", True, self._zero_point(),
                                        {element.name: 0.0 for element in self.elements}, 0.0, 0)
        for element in self.elements:
            if bool(getattr(element, "is_blocked", False)):
                return HydraulicSolveResult(
                    "blocked", False, self._zero_point(), {}, float("nan"), 0,
                    failed_component=element.name, message="A commanded valve blocks the series path",
                )

        fluid = glycol_properties(coolant_temp_c)

        def pressure_budget(mass_flow: float) -> tuple[dict[str, float], float]:
            drops = {
                element.name: float(element.pressure_drop_pa(mass_flow, fluid))
                for element in self.elements
            }
            return drops, float(sum(drops.values()))

        evaluations = 0

        def residual(mass_flow: float) -> float:
            nonlocal evaluations
            evaluations += 1
            _, system_pressure = pressure_budget(mass_flow)
            if not np.isfinite(system_pressure) or system_pressure < 0:
                raise ValueError("Component returned an invalid pressure loss")
            return pump.pressure_rise(speed, mass_flow) - system_pressure

        upper = pump.max_mass_flow_kg_s * speed * (1.0 - 1e-9)
        try:
            lower_residual = residual(0.0)
            upper_residual = residual(upper)
            if lower_residual < 0 or upper_residual > 0:
                return HydraulicSolveResult(
                    "no_intersection", False, self._zero_point(), {}, float("nan"), evaluations,
                    message="Pump and system curves do not bracket a working point",
                )
            mass_flow = float(brentq(residual, 0.0, upper, xtol=1e-12, rtol=1e-12))
            drops, system_pressure = pressure_budget(mass_flow)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return HydraulicSolveResult(
                "solver_failure", False, self._zero_point(), {}, float("nan"), evaluations,
                message=str(exc),
            )
        point = pump.point_at(speed, mass_flow, system_pressure)
        closure = point.pressure_rise_pa - system_pressure
        return HydraulicSolveResult(
            "ok", True, point, drops, float(closure), evaluations,
            message="Pump/system intersection converged",
        )
