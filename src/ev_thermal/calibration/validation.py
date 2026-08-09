"""Independent holdout validation for calibrated parameter sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .identification import simulate_two_node
from .observations import ObservationDataset


@dataclass(frozen=True)
class ValidationComparison:
    dataset_id: str
    maturity: str
    episode_ids: tuple[int, ...]
    baseline_rmse_c: float
    calibrated_rmse_c: float
    improvement_pct: float
    metrics: pd.DataFrame


def _temperature_rmse(observations: ObservationDataset,
                      values: dict[str, float]) -> tuple[float, float, float]:
    predicted = simulate_two_node(observations.frame, values)
    core_error = predicted["predicted_core_temp_c"].to_numpy() - observations.frame["core_temp_c"].to_numpy()
    surface_error = (
        predicted["predicted_surface_temp_c"].to_numpy()
        - observations.frame["surface_temp_c"].to_numpy()
    )
    core_rmse = float(np.sqrt(np.mean(core_error**2)))
    surface_rmse = float(np.sqrt(np.mean(surface_error**2)))
    combined = float(np.sqrt(np.mean(np.concatenate((core_error, surface_error)) ** 2)))
    return core_rmse, surface_rmse, combined


def compare_parameter_sets(observations: ObservationDataset,
                           baseline_values: dict[str, float],
                           calibrated_values: dict[str, float]) -> ValidationComparison:
    """Evaluate baseline and calibrated values on observations not used for fitting."""
    base_core, base_surface, base_combined = _temperature_rmse(observations, baseline_values)
    cal_core, cal_surface, cal_combined = _temperature_rmse(observations, calibrated_values)
    metrics = pd.DataFrame([
        {"parameter_set": "baseline", "core_rmse_c": base_core,
         "surface_rmse_c": base_surface, "combined_rmse_c": base_combined},
        {"parameter_set": "calibrated", "core_rmse_c": cal_core,
         "surface_rmse_c": cal_surface, "combined_rmse_c": cal_combined},
    ])
    improvement = 100.0 * (base_combined - cal_combined) / max(base_combined, 1e-12)
    return ValidationComparison(
        observations.dataset_id,
        observations.maturity,
        observations.episode_ids,
        base_combined,
        cal_combined,
        float(improvement),
        metrics,
    )
