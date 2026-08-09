import pandas as pd

from ev_thermal.calibration.identification import (
    BATTERY_THERMAL_PARAMETERS,
    generate_synthetic_battery_observations,
)
from ev_thermal.calibration.parameters import ParameterRegistry
from ev_thermal.calibration.sensitivity import global_sensitivity, local_sensitivity
from ev_thermal.config import PROJECT_ROOT


def test_sensitivity_is_finite_ranked_and_reproducible():
    registry = ParameterRegistry.from_yaml(PROJECT_ROOT / "configs" / "parameter_registry.yaml")
    values = registry.defaults(BATTERY_THERMAL_PARAMETERS)
    observations = generate_synthetic_battery_observations(
        values, episode_count=3, duration_s=300, dt_s=5, noise_std_c=0.0, seed=9
    )

    local = local_sensitivity(observations, registry, values, BATTERY_THERMAL_PARAMETERS)
    first = global_sensitivity(
        observations, registry, BATTERY_THERMAL_PARAMETERS, sample_count=32, seed=42
    )
    second = global_sensitivity(
        observations, registry, BATTERY_THERMAL_PARAMETERS, sample_count=32, seed=42
    )

    assert set(local["metric"]) >= {"peak_core_temp_c", "coolant_heat_rejection_kwh"}
    assert local["normalized_sensitivity"].notna().all()
    pd.testing.assert_frame_equal(first.rankings, second.rankings)
    pd.testing.assert_frame_equal(first.metric_intervals, second.metric_intervals)
    assert set(first.rankings["parameter"]) == set(BATTERY_THERMAL_PARAMETERS)
