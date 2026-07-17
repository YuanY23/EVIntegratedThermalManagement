"""Darcy-Weisbach pipe and local-component pressure losses."""

from dataclasses import dataclass
import math

from .fluid import FluidProperties


@dataclass(frozen=True)
class PipeLoss:
    pressure_drop_pa: float
    reynolds: float
    friction_factor: float
    velocity_m_s: float


def pipe_pressure_drop(mass_flow_kg_s: float, diameter_m: float, length_m: float,
                       local_loss_coefficient: float, fluid: FluidProperties,
                       roughness_m: float = 1.5e-6) -> PipeLoss:
    """Calculate distributed and local loss using the Darcy friction factor."""
    mdot = max(mass_flow_kg_s, 0.0)
    if mdot == 0:
        return PipeLoss(0.0, 0.0, 0.0, 0.0)
    area = math.pi * diameter_m**2 / 4.0
    velocity = mdot / (fluid.density_kg_m3 * area)
    reynolds = fluid.density_kg_m3 * velocity * diameter_m / fluid.viscosity_pa_s
    if reynolds < 2300:
        friction = 64.0 / max(reynolds, 1.0)
    else:
        # Swamee-Jain is an explicit approximation to Colebrook and avoids an
        # iterative solve at every vehicle time step.
        term = roughness_m / (3.7 * diameter_m) + 5.74 / reynolds**0.9
        friction = 0.25 / math.log10(term) ** 2
    dynamic_pressure = 0.5 * fluid.density_kg_m3 * velocity**2
    pressure_drop = (friction * length_m / diameter_m + local_loss_coefficient) * dynamic_pressure
    return PipeLoss(pressure_drop, reynolds, friction, velocity)

