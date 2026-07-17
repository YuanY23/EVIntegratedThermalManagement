import numpy as np
import pandas as pd
import torch

from ev_thermal.prediction.dataset import SequenceDatasetBuilder
from ev_thermal.prediction.model import ThermalLoadLSTM
from ev_thermal.prediction.training import forecast_metrics


def _episode(episode_id: int, rows: int = 90) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "episode_id": episode_id,
        "time_s": t * 5,
        "speed_mps": 10 + np.sin(t / 10),
        "ambient_temp_c": np.full(rows, 25.0 + episode_id),
        "battery_temp_c": 30 + 0.01 * t,
        "motor_temp_c": 45 + 0.02 * t,
        "cabin_temp_c": np.full(rows, 24.0),
        "battery_power_w": 20_000 + 100 * t,
        "pump_fraction": np.full(rows, 0.4),
        "battery_heat_w": 1000 + 10 * t,
        "powertrain_heat_w": 2000 + 20 * t,
        "cabin_load_w": 500 + 2 * t,
    })


def test_dataset_windows_have_300_second_horizon_without_episode_crossing():
    frame = pd.concat([_episode(i) for i in range(4)], ignore_index=True)
    builder = SequenceDatasetBuilder(history_steps=12, horizon_steps=60, stride=5)
    split = builder.build(frame)
    assert split.train.targets.shape[-2:] == (60, 3)
    assert set(split.train.episode_ids).isdisjoint(split.test.episode_ids)
    assert split.feature_scaler.mean_.shape[0] == len(builder.feature_columns)
    assert {"battery_heat_w", "powertrain_heat_w", "cabin_load_w"}.issubset(builder.feature_columns)


def test_lstm_output_shape_matches_three_future_loads():
    model = ThermalLoadLSTM(input_size=8, output_size=3, hidden_size=16, num_layers=1)
    history = torch.randn(4, 12, 8)
    output = model(history, horizon_steps=60)
    assert output.shape == (4, 60, 3)


def test_forecast_metrics_include_targets_and_horizon_checkpoints():
    true = np.ones((2, 60, 3), dtype=float) * 100
    predicted = true + 10
    metrics = forecast_metrics(predicted, true)
    assert metrics["mae_battery_heat_w"] == 10
    assert metrics["rmse_powertrain_heat_w"] == 10
    assert metrics["mae_all_targets_at_300s"] == 10
    assert "r2_cabin_load_w" in metrics
