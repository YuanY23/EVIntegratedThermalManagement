"""Bounded properties for a 50/50 water-ethylene-glycol coolant."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FluidProperties:
    density_kg_m3: float
    cp_j_kgk: float
    viscosity_pa_s: float
    conductivity_w_mk: float

    @property
    def prandtl(self) -> float:
        return self.cp_j_kgk * self.viscosity_pa_s / self.conductivity_w_mk


def glycol_properties(temperature_c: float) -> FluidProperties:
    """Return engineering correlations bounded to -30..100 degC.

    The polynomial/exponential approximations are intended for system simulation,
    not coolant formulation or freezing-point certification.
    """
    t = float(np.clip(temperature_c, -30.0, 100.0))
    density = 1068.0 - 0.52 * (t - 20.0)
    cp = 3440.0 + 3.2 * (t - 20.0)
    viscosity = 0.0042 * math.exp(-0.032 * (t - 20.0))
    conductivity = 0.385 + 0.00045 * (t - 20.0)
    return FluidProperties(density, cp, viscosity, conductivity)

