"""Bounded grey-box identification for the battery two-node thermal model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .observations import ObservationDataset
from .parameters import CalibratedParameterSet, ParameterRegistry


BATTERY_THERMAL_PARAMETERS = (
    "battery.core_heat_capacity_j_k",
    "battery.surface_heat_capacity_j_k",
    "battery.core_surface_conductance_w_k",
)


@dataclass(frozen=True)
class IdentificationResult:
    success: bool
    message: str
    dataset_id: str
    maturity: str
    estimates: dict[str, float]
    standard_errors: dict[str, float]
    confidence_95: dict[str, tuple[float, float]]
    residuals: pd.DataFrame
    cost: float
    nfev: int
    jacobian_rank: int
    condition_number: float
    boundary_parameters: tuple[str, ...]

    def parameter_set(self, registry: ParameterRegistry) -> CalibratedParameterSet:
        if not self.success:
            raise ValueError("A failed identification result cannot be serialized as calibrated parameters")
        return CalibratedParameterSet(
            dataset_id=self.dataset_id,
            maturity=self.maturity,
            values=self.estimates,
            units={name: registry[name].unit for name in self.estimates},
            method="bounded_least_squares",
        )

    def write_artifacts(self, output_dir: str | Path, registry: ParameterRegistry) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.success:
            self.parameter_set(registry).to_json(output / "identified_parameters.json")
        self.residuals.to_csv(output / "identification_residuals.csv", index=False)
        rows = []
        for name, estimate in self.estimates.items():
            lower, upper = self.confidence_95[name]
            rows.append({
                "parameter": name,
                "estimate": estimate,
                "unit": registry[name].unit,
                "standard_error": self.standard_errors[name],
                "ci95_lower": lower,
                "ci95_upper": upper,
                "at_boundary": name in self.boundary_parameters,
            })
        pd.DataFrame(rows).to_csv(output / "parameter_estimates.csv", index=False)


def simulate_two_node(frame: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    """Simulate episode trajectories with observed heat/coolant boundary inputs."""
    core_capacity = float(values[BATTERY_THERMAL_PARAMETERS[0]])
    surface_capacity = float(values[BATTERY_THERMAL_PARAMETERS[1]])
    conductance = float(values[BATTERY_THERMAL_PARAMETERS[2]])
    if min(core_capacity, surface_capacity, conductance) <= 0:
        raise ValueError("Battery thermal parameters must be positive")

    output = pd.DataFrame(index=frame.index, columns=(
        "predicted_core_temp_c", "predicted_surface_temp_c", "coolant_heat_w"
    ), dtype=float)
    for _, episode in frame.groupby("episode_id", sort=False):
        episode = episode.sort_values("time_s")
        core = float(episode.iloc[0]["core_temp_c"])
        surface = float(episode.iloc[0]["surface_temp_c"])
        for position, (index, row) in enumerate(episode.iterrows()):
            coolant_heat = float(row["coolant_ua_w_k"]) * (surface - float(row["coolant_temp_c"]))
            output.loc[index] = (core, surface, coolant_heat)
            if position == len(episode) - 1:
                continue
            next_time = float(episode.iloc[position + 1]["time_s"])
            dt = next_time - float(row["time_s"])
            core_surface_heat = conductance * (core - surface)
            next_core = core + dt * (float(row["heat_generation_w"]) - core_surface_heat) / core_capacity
            next_surface = surface + dt * (
                core_surface_heat - coolant_heat + float(row.get("external_heat_w", 0.0))
            ) / surface_capacity
            core, surface = next_core, next_surface
    return output.sort_index()


def _fit_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("episode_id", sort=False).cumcount().to_numpy() > 0


def identify_battery_thermal_parameters(
    observations: ObservationDataset,
    registry: ParameterRegistry,
    parameter_names: tuple[str, ...] = BATTERY_THERMAL_PARAMETERS,
) -> IdentificationResult:
    """Estimate bounded parameters and return explicit identifiability diagnostics."""
    frame = observations.frame
    maturity = "synthetic_recovery" if observations.maturity == "synthetic" else "measured_calibration"
    minimum_rows = max(20, 6 * len(parameter_names))
    if len(frame) < minimum_rows or len(observations.episode_ids) < 2:
        return IdentificationResult(
            False,
            f"Insufficient observations: need at least {minimum_rows} rows across two episodes",
            observations.dataset_id,
            maturity,
            {}, {}, {}, pd.DataFrame(), float("nan"), 0, 0, float("inf"), (),
        )

    specs = [registry[name] for name in parameter_names]
    x0 = np.asarray([spec.default_value for spec in specs], dtype=float)
    lower = np.asarray([spec.lower_bound for spec in specs], dtype=float)
    upper = np.asarray([spec.upper_bound for spec in specs], dtype=float)
    mask = _fit_mask(frame)
    core_observed = frame.loc[mask, "core_temp_c"].to_numpy(dtype=float)
    surface_observed = frame.loc[mask, "surface_temp_c"].to_numpy(dtype=float)
    core_std = frame.loc[mask, "core_temp_std_c"].to_numpy(dtype=float)
    surface_std = frame.loc[mask, "surface_temp_std_c"].to_numpy(dtype=float)

    def unpack(vector: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(parameter_names, vector)}

    def residual(vector: np.ndarray) -> np.ndarray:
        predicted = simulate_two_node(frame, unpack(vector)).loc[mask]
        core_residual = (predicted["predicted_core_temp_c"].to_numpy() - core_observed) / core_std
        surface_residual = (
            predicted["predicted_surface_temp_c"].to_numpy() - surface_observed
        ) / surface_std
        return np.concatenate((core_residual, surface_residual))

    span = upper - lower

    def from_normalized(vector: np.ndarray) -> np.ndarray:
        return lower + np.clip(vector, 0.0, 1.0) * span

    def objective(normalized: np.ndarray) -> float:
        errors = residual(from_normalized(normalized))
        return 0.5 * float(np.dot(errors, errors))

    normalized_x0 = (x0 - lower) / span
    fit = minimize(
        objective,
        normalized_x0,
        method="Powell",
        bounds=[(0.0, 1.0)] * len(parameter_names),
        options={"maxiter": 300, "xtol": 1e-7, "ftol": 1e-9},
    )
    fitted_values = from_normalized(fit.x)
    fitted_residual = residual(fitted_values)
    estimates = unpack(fitted_values)
    jacobian = np.empty((len(fitted_residual), len(parameter_names)), dtype=float)
    for index in range(len(parameter_names)):
        step = max(1e-5 * span[index], 1e-9)
        low_vector = fitted_values.copy()
        high_vector = fitted_values.copy()
        low_vector[index] = max(lower[index], fitted_values[index] - step)
        high_vector[index] = min(upper[index], fitted_values[index] + step)
        denominator = high_vector[index] - low_vector[index]
        jacobian[:, index] = (residual(high_vector) - residual(low_vector)) / denominator
    scaled_jacobian = jacobian * span[np.newaxis, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    tolerance = np.finfo(float).eps * max(scaled_jacobian.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > tolerance else float("inf")
    boundary = tuple(
        name for name, value, low, high, width in zip(parameter_names, fitted_values, lower, upper, span)
        if min(value - low, high - value) <= 1e-4 * width
    )

    degrees_of_freedom = max(len(fitted_residual) - len(fitted_values), 1)
    residual_variance = float(np.dot(fitted_residual, fitted_residual) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.pinv(jacobian.T @ jacobian)
    standard_error_vector = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    standard_errors = {
        name: float(error) for name, error in zip(parameter_names, standard_error_vector)
    }
    confidence = {
        name: (float(max(low, value - 1.96 * error)), float(min(high, value + 1.96 * error)))
        for name, value, error, low, high in zip(
            parameter_names, fitted_values, standard_error_vector, lower, upper
        )
    }

    predicted = simulate_two_node(frame, estimates)
    residuals = frame[["dataset_id", "maturity", "episode_id", "time_s"]].copy()
    residuals["observed_core_temp_c"] = frame["core_temp_c"]
    residuals["predicted_core_temp_c"] = predicted["predicted_core_temp_c"]
    residuals["core_residual_c"] = residuals["predicted_core_temp_c"] - residuals["observed_core_temp_c"]
    residuals["observed_surface_temp_c"] = frame["surface_temp_c"]
    residuals["predicted_surface_temp_c"] = predicted["predicted_surface_temp_c"]
    residuals["surface_residual_c"] = (
        residuals["predicted_surface_temp_c"] - residuals["observed_surface_temp_c"]
    )

    identifiable = rank == len(parameter_names) and np.isfinite(condition) and condition < 1e8
    success = bool(fit.success and identifiable and not boundary)
    diagnostics = []
    if not fit.success:
        diagnostics.append(f"optimizer: {fit.message}")
    if not identifiable:
        diagnostics.append(f"unidentifiable Jacobian (rank={rank}, condition={condition:.3g})")
    if boundary:
        diagnostics.append(f"parameters at bounds: {', '.join(boundary)}")
    message = "Identification converged with full-rank local sensitivity" if success else "; ".join(diagnostics)
    return IdentificationResult(
        success, message, observations.dataset_id, maturity, estimates, standard_errors,
        confidence, residuals, float(fit.fun), int(fit.nfev), rank, condition, boundary,
    )


def generate_synthetic_battery_observations(
    values: dict[str, float],
    episode_count: int = 6,
    duration_s: int = 900,
    dt_s: int = 5,
    noise_std_c: float = 0.03,
    seed: int = 42,
    dataset_id: str = "synthetic-battery-thermal-v1",
) -> ObservationDataset:
    """Create persistently excited truth data for method verification only."""
    if episode_count <= 0 or duration_s < dt_s or dt_s <= 0 or noise_std_c < 0:
        raise ValueError("Invalid synthetic observation settings")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    core_capacity = float(values[BATTERY_THERMAL_PARAMETERS[0]])
    surface_capacity = float(values[BATTERY_THERMAL_PARAMETERS[1]])
    conductance = float(values[BATTERY_THERMAL_PARAMETERS[2]])
    times = np.arange(0, duration_s + dt_s, dt_s, dtype=float)
    measurement_std = max(noise_std_c, 0.01)
    for episode_id in range(episode_count):
        phase = 0.55 * episode_id
        core = 20.0 + 2.2 * episode_id
        surface = core - 1.0 + 0.2 * np.sin(phase)
        for position, time_s in enumerate(times):
            heat = max(
                100.0,
                1700.0 + 2100.0 * (np.sin(2 * np.pi * time_s / 190.0 + phase) > 0)
                + 850.0 * np.sin(2 * np.pi * time_s / 73.0 + phase) + 180.0 * episode_id,
            )
            coolant = 16.0 + 1.4 * episode_id + 4.5 * np.sin(2 * np.pi * time_s / 310.0 + phase)
            coolant_ua = 80.0 + 310.0 * (0.5 + 0.5 * np.sin(2 * np.pi * time_s / 235.0 + 0.7 * phase))
            external_heat = 900.0 if (position // 24 + episode_id) % 4 == 0 else 0.0
            rows.append({
                "dataset_id": dataset_id,
                "maturity": "synthetic",
                "episode_id": episode_id,
                "time_s": time_s,
                "heat_generation_w": heat,
                "coolant_temp_c": coolant,
                "coolant_ua_w_k": coolant_ua,
                "external_heat_w": external_heat,
                "core_temp_c": core + rng.normal(0.0, noise_std_c),
                "surface_temp_c": surface + rng.normal(0.0, noise_std_c),
                "core_temp_std_c": measurement_std,
                "surface_temp_std_c": measurement_std,
            })
            core_surface_heat = conductance * (core - surface)
            coolant_heat = coolant_ua * (surface - coolant)
            next_core = core + dt_s * (heat - core_surface_heat) / core_capacity
            next_surface = surface + dt_s * (
                core_surface_heat - coolant_heat + external_heat
            ) / surface_capacity
            core, surface = next_core, next_surface
    return ObservationDataset.from_frame(pd.DataFrame(rows))
