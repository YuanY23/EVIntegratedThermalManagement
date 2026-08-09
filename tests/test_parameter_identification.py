import numpy as np
import pandas as pd

from ev_thermal.calibration.identification import (
    BATTERY_THERMAL_PARAMETERS,
    generate_synthetic_battery_observations,
    identify_battery_thermal_parameters,
)
from ev_thermal.calibration.observations import ObservationDataset
from ev_thermal.calibration.parameters import ParameterRegistry
from ev_thermal.calibration.validation import compare_parameter_sets
from ev_thermal.config import PROJECT_ROOT


def _registry():
    return ParameterRegistry.from_yaml(PROJECT_ROOT / "configs" / "parameter_registry.yaml")


def test_synthetic_truth_recovery_improves_independent_validation():
    truth = {
        "battery.core_heat_capacity_j_k": 330_000.0,
        "battery.surface_heat_capacity_j_k": 118_000.0,
        "battery.core_surface_conductance_w_k": 295.0,
    }
    observations = generate_synthetic_battery_observations(
        truth,
        episode_count=6,
        duration_s=700,
        dt_s=5,
        noise_std_c=0.015,
        seed=17,
        dataset_id="synthetic-recovery-test",
    )
    training = observations.subset([0, 1, 2, 3])
    validation = observations.subset([4, 5])
    registry = _registry()

    result = identify_battery_thermal_parameters(training, registry)

    assert result.success, result.message
    assert result.dataset_id == "synthetic-recovery-test"
    assert result.maturity == "synthetic_recovery"
    for name in BATTERY_THERMAL_PARAMETERS:
        assert np.isclose(result.estimates[name], truth[name], rtol=0.08)

    comparison = compare_parameter_sets(
        validation,
        registry.defaults(BATTERY_THERMAL_PARAMETERS),
        result.estimates,
    )
    assert comparison.calibrated_rmse_c < 0.5 * comparison.baseline_rmse_c
    assert set(comparison.episode_ids) == {4, 5}


def test_insufficient_observations_return_failed_diagnostics():
    truth = {
        "battery.core_heat_capacity_j_k": 330_000.0,
        "battery.surface_heat_capacity_j_k": 118_000.0,
        "battery.core_surface_conductance_w_k": 295.0,
    }
    observations = generate_synthetic_battery_observations(
        truth, episode_count=1, duration_s=20, dt_s=5, noise_std_c=0.0, seed=2
    )

    result = identify_battery_thermal_parameters(observations, _registry())

    assert not result.success
    assert "insufficient" in result.message.lower()
    assert result.estimates == {}


def test_unexcited_data_is_reported_as_unidentifiable():
    rows = []
    for episode_id in (0, 1):
        for time_s in range(0, 80, 5):
            rows.append({
                "dataset_id": "unexcited",
                "maturity": "synthetic",
                "episode_id": episode_id,
                "time_s": time_s,
                "heat_generation_w": 0.0,
                "coolant_temp_c": 25.0,
                "coolant_ua_w_k": 0.0,
                "core_temp_c": 25.0,
                "surface_temp_c": 25.0,
                "core_temp_std_c": 0.1,
                "surface_temp_std_c": 0.1,
            })
    observations = ObservationDataset.from_frame(pd.DataFrame(rows))

    result = identify_battery_thermal_parameters(observations, _registry())

    assert not result.success
    assert "unidentifiable" in result.message
    assert result.jacobian_rank < len(BATTERY_THERMAL_PARAMETERS)
