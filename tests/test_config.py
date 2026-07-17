from pathlib import Path

import pytest

from ev_thermal.config import load_config


def test_default_config_has_complete_300_second_forecast():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default_config.yaml")
    assert cfg.simulation.dt_s > 0
    assert cfg.prediction.horizon_s == 300
    assert cfg.prediction.horizon_s % cfg.simulation.dt_s == 0
    assert cfg.vehicle.mass_kg > 1000
    assert cfg.battery.capacity_kwh > 40


def test_invalid_forecast_grid_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("simulation:\n  dt_s: 7\nprediction:\n  horizon_s: 300\n", encoding="utf-8")
    with pytest.raises(ValueError, match="divisible"):
        load_config(path)

