from pathlib import Path

import pandas as pd
import pytest

from ev_thermal.calibration.observations import ObservationDataset, ObservationValidationError
from ev_thermal.calibration.parameters import (
    CalibratedParameterSet,
    ParameterRegistry,
    battery_parameters_from_config,
)
from ev_thermal.config import PROJECT_ROOT, load_config


def test_registry_defaults_match_project_config_and_round_trip(tmp_path):
    registry = ParameterRegistry.from_yaml(PROJECT_ROOT / "configs" / "parameter_registry.yaml")
    config = load_config()

    registry.validate_project_defaults(config)
    adapted = battery_parameters_from_config(
        config, {"battery.core_heat_capacity_j_k": 400_000.0}
    )
    assert adapted.core_heat_capacity_j_k == 400_000.0
    assert adapted.surface_heat_capacity_j_k == config.battery.surface_heat_capacity_j_k
    battery_specs = registry.select(model="battery.two_node", calibratable=True)
    assert {spec.name for spec in battery_specs} >= {
        "battery.core_heat_capacity_j_k",
        "battery.surface_heat_capacity_j_k",
        "battery.core_surface_conductance_w_k",
    }
    assert all(spec.lower_bound <= spec.default_value <= spec.upper_bound for spec in battery_specs)

    calibrated = CalibratedParameterSet(
        dataset_id="synthetic-unit-test",
        maturity="synthetic_recovery",
        values={spec.name: spec.default_value for spec in battery_specs[:3]},
        units={spec.name: spec.unit for spec in battery_specs[:3]},
        method="bounded_least_squares",
    )
    output = tmp_path / "parameters.json"
    calibrated.to_json(output)
    assert CalibratedParameterSet.from_json(output) == calibrated


def test_registry_rejects_duplicate_names_and_invalid_bounds(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
parameters:
  - name: battery.capacity
    unit: J/K
    default: 10
    lower: 20
    upper: 30
    source: test
    confidence: low
    model: battery.two_node
  - name: battery.capacity
    unit: J/K
    default: 25
    lower: 10
    upper: 30
    source: test
    confidence: low
    model: battery.two_node
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bound|Duplicate"):
        ParameterRegistry.from_yaml(registry_path)


def _valid_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["synthetic-a"] * 3,
            "maturity": ["synthetic"] * 3,
            "episode_id": [0, 0, 0],
            "time_s": [0.0, 5.0, 10.0],
            "heat_generation_w": [1000.0, 1200.0, 800.0],
            "coolant_temp_c": [25.0, 25.0, 25.0],
            "coolant_ua_w_k": [200.0, 200.0, 200.0],
            "core_temp_c": [30.0, 30.01, 30.02],
            "surface_temp_c": [29.0, 29.01, 29.02],
            "core_temp_std_c": [0.1, 0.1, 0.1],
            "surface_temp_std_c": [0.1, 0.1, 0.1],
        }
    )


def test_observation_contract_rejects_missing_columns_and_mixed_maturity():
    frame = _valid_observation_frame()
    dataset = ObservationDataset.from_frame(frame)
    assert dataset.dataset_id == "synthetic-a"
    assert dataset.maturity == "synthetic"

    with pytest.raises(ObservationValidationError, match="Missing observation columns"):
        ObservationDataset.from_frame(frame.drop(columns=["surface_temp_c"]))

    mixed = frame.copy()
    mixed.loc[2, "maturity"] = "measured"
    with pytest.raises(ObservationValidationError, match="one maturity"):
        ObservationDataset.from_frame(mixed)
