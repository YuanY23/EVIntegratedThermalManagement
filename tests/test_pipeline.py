from pathlib import Path

import pandas as pd

from ev_thermal.config import load_config
from ev_thermal.prediction.training import TrainingResult
from ev_thermal.pipeline import generate_simulation_dataset, run_all, run_robustness_checks, run_strategy_comparison
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


def test_quick_run_is_isolated_from_canonical_formal_artifacts(tmp_path, monkeypatch):
    canonical = tmp_path / "results" / "tables" / "strategy_comparison.csv"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("formal-result", encoding="utf-8")

    def fake_dataset(config, output_path, episode_count, duration_s, seed):
        frame = pd.DataFrame({"episode_id": list(range(episode_count)), "value": 1.0})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return frame

    def fake_training(config, frame, model_dir, quick):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"mae_battery_heat_w": 1.0}
        for name in ("thermal_load_lstm.pt", "feature_scaler.joblib", "target_scaler.joblib"):
            (model_dir / name).write_bytes(b"model")
        (model_dir / "training_history.json").write_text("[]", encoding="utf-8")
        (model_dir / "test_metrics.json").write_text('{"mae_battery_heat_w": 1.0}', encoding="utf-8")
        return TrainingResult(1, 0.1, [], metrics, str(model_dir / "thermal_load_lstm.pt"))

    def fake_comparison(config, scenarios, duration_s, predictor):
        rows = [
            {"scenario": scenario, "strategy": strategy, "thermal_balance_error_pct": 0.1}
            for scenario in scenarios
            for strategy in ("baseline", "predictive")
        ]
        trajectories = {(row["scenario"], row["strategy"]): pd.DataFrame() for row in rows}
        return pd.DataFrame(rows), trajectories

    def fake_plot(*args):
        output = Path(args[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"x" * 1200)

    import ev_thermal.pipeline as pipeline

    monkeypatch.setattr(pipeline, "load_config", lambda path: load_config())
    monkeypatch.setattr(pipeline, "generate_simulation_dataset", fake_dataset)
    monkeypatch.setattr(pipeline, "train_predictor_from_frame", fake_training)
    monkeypatch.setattr(pipeline, "ThermalLoadPredictor", lambda model_dir: object())
    monkeypatch.setattr(pipeline, "run_strategy_comparison", fake_comparison)
    monkeypatch.setattr(
        pipeline,
        "run_robustness_checks",
        lambda config, duration_s: pd.DataFrame(
            [
                {"case": "invalid_forecast", "thermal_balance_error_pct": 0.1},
                {"case": "high_heat_bias", "thermal_balance_error_pct": 0.1},
            ]
        ),
    )
    monkeypatch.setattr(pipeline, "plot_strategy_comparison", fake_plot)
    monkeypatch.setattr(pipeline, "plot_training_history", fake_plot)
    monkeypatch.setattr(pipeline, "plot_scenario_overview", fake_plot)
    monkeypatch.setattr(pipeline, "_plant_sha256", lambda root: "plant-hash")
    monkeypatch.setattr(pipeline, "_config_sha256", lambda root: "config-hash")

    manifest = run_all(tmp_path, quick=True, run_id="quick-test")

    assert canonical.read_text(encoding="utf-8") == "formal-result"
    assert manifest["profile"] == "quick"
    assert (tmp_path / "artifacts" / "runs" / "quick-test" / "results" / "logs" / "run_manifest.json").exists()
    assert (tmp_path / "artifacts" / "latest" / "quick.json").exists()
    assert not (tmp_path / "results" / "logs" / "run_manifest.json").exists()
