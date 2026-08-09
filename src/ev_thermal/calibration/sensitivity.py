"""Deterministic local and global sensitivity analysis for calibration models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .identification import simulate_two_node
from .observations import ObservationDataset
from .parameters import ParameterRegistry


def _metrics(observations: ObservationDataset, values: dict[str, float]) -> dict[str, float]:
    predicted = simulate_two_node(observations.frame, values)
    frame = observations.frame
    energy_j = 0.0
    for _, episode in frame.groupby("episode_id", sort=False):
        indices = episode.sort_values("time_s").index
        times = frame.loc[indices, "time_s"].to_numpy(dtype=float)
        heat = predicted.loc[indices, "coolant_heat_w"].to_numpy(dtype=float)
        if len(times) > 1:
            energy_j += float(np.trapz(np.maximum(heat, 0.0), times))
    return {
        "peak_core_temp_c": float(predicted["predicted_core_temp_c"].max()),
        "peak_surface_temp_c": float(predicted["predicted_surface_temp_c"].max()),
        "coolant_heat_rejection_kwh": energy_j / 3.6e6,
    }


def local_sensitivity(observations: ObservationDataset, registry: ParameterRegistry,
                      base_values: dict[str, float], parameter_names: tuple[str, ...],
                      relative_step: float = 0.01) -> pd.DataFrame:
    """Return central-difference elasticities around one parameter set."""
    if not 0 < relative_step < 0.5:
        raise ValueError("relative_step must be between zero and 0.5")
    base_metrics = _metrics(observations, base_values)
    rows = []
    for name in parameter_names:
        spec = registry[name]
        value = float(base_values[name])
        low_value = max(spec.lower_bound, value * (1.0 - relative_step))
        high_value = min(spec.upper_bound, value * (1.0 + relative_step))
        low = dict(base_values)
        high = dict(base_values)
        low[name], high[name] = low_value, high_value
        low_metrics = _metrics(observations, low)
        high_metrics = _metrics(observations, high)
        fractional_parameter_change = (high_value - low_value) / max(abs(value), 1e-12)
        for metric, base_metric in base_metrics.items():
            delta_metric = high_metrics[metric] - low_metrics[metric]
            normalized = (delta_metric / max(abs(base_metric), 1e-12)) / fractional_parameter_change
            rows.append({
                "parameter": name,
                "unit": spec.unit,
                "metric": metric,
                "base_metric": base_metric,
                "normalized_sensitivity": float(normalized),
                "absolute_sensitivity": float(abs(normalized)),
            })
    return pd.DataFrame(rows).sort_values(
        ["metric", "absolute_sensitivity", "parameter"], ascending=[True, False, True]
    ).reset_index(drop=True)


@dataclass(frozen=True)
class GlobalSensitivityResult:
    rankings: pd.DataFrame
    metric_intervals: pd.DataFrame
    sample_count: int
    seed: int


def global_sensitivity(observations: ObservationDataset, registry: ParameterRegistry,
                       parameter_names: tuple[str, ...], sample_count: int = 128,
                       seed: int = 42) -> GlobalSensitivityResult:
    """Uniform bounded sampling with rank correlation and output intervals."""
    if sample_count < 16:
        raise ValueError("sample_count must be at least 16")
    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.uniform(registry[name].lower_bound, registry[name].upper_bound, sample_count)
        for name in parameter_names
    ])
    metric_rows = [
        _metrics(observations, {name: float(value) for name, value in zip(parameter_names, row)})
        for row in samples
    ]
    metric_frame = pd.DataFrame(metric_rows)
    rankings = []
    for metric in metric_frame.columns:
        for index, name in enumerate(parameter_names):
            correlation, p_value = spearmanr(samples[:, index], metric_frame[metric].to_numpy())
            rankings.append({
                "metric": metric,
                "parameter": name,
                "spearman_r": float(correlation),
                "absolute_spearman_r": float(abs(correlation)),
                "p_value": float(p_value),
            })
    ranking_frame = pd.DataFrame(rankings).sort_values(
        ["metric", "absolute_spearman_r", "parameter"], ascending=[True, False, True]
    ).reset_index(drop=True)
    intervals = pd.DataFrame([
        {
            "metric": metric,
            "p05": float(metric_frame[metric].quantile(0.05)),
            "median": float(metric_frame[metric].quantile(0.50)),
            "p95": float(metric_frame[metric].quantile(0.95)),
        }
        for metric in metric_frame.columns
    ])
    return GlobalSensitivityResult(ranking_frame, intervals, sample_count, seed)
