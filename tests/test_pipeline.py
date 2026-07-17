from pathlib import Path

import pandas as pd

from ev_thermal.config import load_config
from ev_thermal.pipeline import generate_simulation_dataset, run_robustness_checks, run_strategy_comparison
from ev_thermal.visualization import plot_scenario_overview


def test_quick_pipeline_writes_dataset_comparison_and_figure(tmp_path):
    cfg = load_config()
    dataset_path = tmp_path / "episodes.csv"
    frame = generate_simulation_dataset(cfg, dataset_path, episode_count=6,
                                        duration_s=650, seed=11)
    assert dataset_path.exists()
    assert frame["episode_id"].nunique() == 6
    required = {"battery_heat_w", "powertrain_heat_w", "cabin_load_w"}
    assert required.issubset(frame.columns)

    table, trajectories = run_strategy_comparison(cfg, ["urban_hot"], duration_s=300)
    assert set(table["strategy"]) == {"baseline", "predictive"}
    comparison_path = tmp_path / "comparison.csv"
    table.to_csv(comparison_path, index=False)
    figure_path = tmp_path / "overview.png"
    plot_scenario_overview(trajectories[("urban_hot", "predictive")], figure_path)
    assert comparison_path.stat().st_size > 100
    assert figure_path.stat().st_size > 1000


def test_robustness_table_confirms_invalid_forecast_fallback():
    table = run_robustness_checks(load_config(), duration_s=180)
    invalid = table.loc[table["case"] == "invalid_forecast"].iloc[0]
    assert invalid["max_temperature_deviation_c"] < 1e-9
    assert invalid["energy_deviation_pct"] < 1e-9
