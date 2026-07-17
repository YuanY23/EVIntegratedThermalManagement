"""Two-resistance/two-capacitance passenger-cabin thermal model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CabinParameters:
    air_heat_capacity_j_k: float = 65_000.0
    interior_heat_capacity_j_k: float = 650_000.0
    envelope_ua_w_k: float = 110.0
    air_interior_ua_w_k: float = 180.0
    infiltration_ua_w_k: float = 35.0
    solar_aperture_m2: float = 3.2
    solar_absorptance: float = 0.55
    occupant_sensible_heat_w: float = 90.0


@dataclass(frozen=True)
class CabinState:
    air_temp_c: float
    interior_temp_c: float


@dataclass(frozen=True)
class CabinDiagnostics:
    ambient_heat_w: float
    solar_heat_w: float
    occupant_heat_w: float
    interior_to_air_heat_w: float
    hvac_heat_w: float
    net_unconditioned_load_w: float


@dataclass(frozen=True)
class CabinStep:
    state: CabinState
    diagnostics: CabinDiagnostics


class CabinModel:
    def __init__(self, params: CabinParameters | None = None):
        self.params = params or CabinParameters()

    def step(self, state: CabinState, ambient_temp_c: float, solar_w_m2: float,
             occupants: int, hvac_heat_w: float, dt_s: float) -> CabinStep:
        p = self.params
        # Positive heat enters cabin air/interior. Envelope heat first enters the
        # high-inertia trim/body node; infiltration acts directly on cabin air.
        envelope_heat = p.envelope_ua_w_k * (ambient_temp_c - state.interior_temp_c)
        infiltration_heat = p.infiltration_ua_w_k * (ambient_temp_c - state.air_temp_c)
        solar_heat = max(solar_w_m2, 0.0) * p.solar_aperture_m2 * p.solar_absorptance
        occupant_heat = max(occupants, 0) * p.occupant_sensible_heat_w
        interior_to_air = p.air_interior_ua_w_k * (state.interior_temp_c - state.air_temp_c)
        d_interior = (envelope_heat + 0.65 * solar_heat - interior_to_air) / p.interior_heat_capacity_j_k
        d_air = (infiltration_heat + 0.35 * solar_heat + occupant_heat + interior_to_air + hvac_heat_w) / p.air_heat_capacity_j_k
        return CabinStep(
            CabinState(state.air_temp_c + dt_s * d_air, state.interior_temp_c + dt_s * d_interior),
            CabinDiagnostics(envelope_heat + infiltration_heat, solar_heat, occupant_heat,
                             interior_to_air, hvac_heat_w,
                             envelope_heat + infiltration_heat + solar_heat + occupant_heat),
        )

