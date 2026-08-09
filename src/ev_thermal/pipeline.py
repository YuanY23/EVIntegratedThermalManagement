"""Reproducible end-to-end orchestration for data, training, and experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .artifacts import (
    compute_plant_sha256,
    create_run_layout,
    promote_run,
    update_latest_pointer,
    write_run_manifest,
)
from .config import PROJECT_ROOT, ProjectConfig, load_config, set_seed
from .control.predictive import ForecastSummary
from .prediction.dataset import SequenceDatasetBuilder
from .prediction.predictor import ThermalLoadPredictor
from .prediction.training import TrainingOptions, TrainingResult, train_model
from .simulation.integrated import IntegratedSimulator
from .simulation.scenarios import Scenario, make_scenario, scenario_names
from .thermal_hydraulics.topologies import HydraulicDesign, architecture_names
from .visualization import plot_scenario_overview, plot_strategy_comparison, plot_training_history


def _perturb_scenario(scenario: Scenario, rng: np.random.Generator, episode_id: int) -> Scenario:
    ambient_offset = rng.uniform(-4.0, 4.0)
    temperature_offset = rng.uniform(-3.0, 3.0)
    speed_scale = rng.uniform(0.88, 1.12)
    return replace(
        scenario,
        name=f"{scenario.name}_{episode_id:03d}",
        speed_mps=np.clip(scenario.speed_mps * speed_scale, 0, 40),
        ambient_temp_c=scenario.ambient_temp_c + ambient_offset,
        solar_w_m2=scenario.solar_w_m2 * rng.uniform(0.7, 1.25),
        initial_battery_temp_c=scenario.initial_battery_temp_c + temperature_offset,
        initial_cabin_temp_c=scenario.initial_cabin_temp_c + temperature_offset,
        initial_soc=float(rng.uniform(0.55, 0.95)),
    )


def generate_simulation_dataset(config: ProjectConfig, output_path: str | Path,
                                episode_count: int = 24, duration_s: int = 1200,
                                seed: int = 42) -> pd.DataFrame:
    """Generate physics-based episodes used by the thermal-load predictor."""
    rng = np.random.default_rng(seed)
    simulator = IntegratedSimulator(config)
    frames = []
    names = scenario_names()
    for episode_id in range(episode_count):
        base = make_scenario(names[episode_id % len(names)], duration_s,
                             config.simulation.dt_s, seed + episode_id)
        scenario = _perturb_scenario(base, rng, episode_id)
        trajectory = simulator.run(scenario, "baseline").timeseries.copy()
        trajectory.insert(0, "episode_id", episode_id)
        trajectory["battery_temp_c"] = trajectory["battery_core_temp_c"]
        trajectory["battery_power_w"] = trajectory["battery_total_power_w"]
        frames.append(trajectory)
    dataset = pd.concat(frames, ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    return dataset


def train_predictor_from_frame(config: ProjectConfig, frame: pd.DataFrame,
                               model_dir: str | Path, quick: bool = False) -> TrainingResult:
    steps = config.prediction.history_s // config.simulation.dt_s
    horizon = config.prediction.horizon_s // config.simulation.dt_s
    builder = SequenceDatasetBuilder(steps, horizon, stride=5 if quick else 3)
    split = builder.build(frame)
    options = TrainingOptions(
        epochs=3 if quick else config.prediction.max_epochs,
        batch_size=config.prediction.batch_size,
        learning_rate=config.prediction.learning_rate,
        patience=2 if quick else config.prediction.patience,
        seed=config.seed,
        hidden_size=24 if quick else config.prediction.hidden_size,
        num_layers=1 if quick else config.prediction.num_layers,
    )
    return train_model(split, model_dir, options)


def run_strategy_comparison(config: ProjectConfig, scenarios: list[str] | None = None,
                            duration_s: int = 1800, predictor=None):
    simulator = IntegratedSimulator(config)
    tables, trajectories = [], {}
    for index, scenario_name in enumerate(scenarios or scenario_names()):
        scenario = make_scenario(scenario_name, duration_s, config.simulation.dt_s,
                                 config.seed + index)
        for strategy in ("baseline", "predictive"):
            result = simulator.run(scenario, strategy, predictor if strategy == "predictive" else None)
            row = {"scenario": scenario_name, "strategy": strategy, **result.metrics}
            tables.append(row)
            trajectories[(scenario_name, strategy)] = result.timeseries
    return pd.DataFrame(tables), trajectories


def run_robustness_checks(config: ProjectConfig, duration_s: int = 600) -> pd.DataFrame:
    """Quantify fail-safe fallback and a conservative high-heat forecast bias."""
    class FixedPredictor:
        def __init__(self, summary: ForecastSummary):
            self._summary = summary

        def summary(self, rows):
            return self._summary

    scenario = make_scenario("aggressive", duration_s, config.simulation.dt_s, config.seed + 900)
    simulator = IntegratedSimulator(config)
    baseline = simulator.run(scenario, "baseline")
    cases = {
        "invalid_forecast": ForecastSummary.invalid("injected_sensor_fault"),
        "high_heat_bias": ForecastSummary(True, 7000.0, 2500.0, 1500.0, "injected_bias"),
    }
    rows = []
    baseline_energy = baseline.metrics["net_energy_kwh"]
    for name, summary in cases.items():
        result = simulator.run(scenario, "predictive", FixedPredictor(summary))
        temperature_deviation = float(np.max(np.abs(
            result.timeseries["battery_core_temp_c"] - baseline.timeseries["battery_core_temp_c"])))
        energy_deviation = 100.0 * abs(result.metrics["net_energy_kwh"] - baseline_energy) / max(abs(baseline_energy), 1e-9)
        rows.append({
            "case": name,
            "max_temperature_deviation_c": temperature_deviation,
            "energy_deviation_pct": energy_deviation,
            "fallback_expected": not summary.valid,
            "thermal_balance_error_pct": result.metrics["thermal_balance_error_pct"],
        })
    return pd.DataFrame(rows)


def _architecture_feasibility(result) -> tuple[bool, str]:
    metrics = result.metrics
    reasons = []
    if metrics["hydraulic_solver_failures"] > 0:
        reasons.append("hydraulic_solver_failure")
    if metrics["thermal_balance_error_pct"] >= 2.0:
        reasons.append("thermal_balance")
    if metrics["max_battery_temp_c"] >= 50.0:
        reasons.append("battery_temperature")
    if metrics["max_motor_temp_c"] >= 140.0:
        reasons.append("motor_temperature")
    return not reasons, "ok" if not reasons else ";".join(reasons)


def run_architecture_comparison(config: ProjectConfig, scenarios: list[str] | None = None,
                                duration_s: int = 900,
                                design: HydraulicDesign | None = None) -> pd.DataFrame:
    """Compare three loop architectures under identical scenario initial conditions."""
    rows = []
    selected = scenarios or scenario_names()
    design = design or HydraulicDesign()
    for scenario_index, scenario_name in enumerate(selected):
        scenario = make_scenario(
            scenario_name, duration_s, config.simulation.dt_s, config.seed + scenario_index
        )
        for architecture in architecture_names():
            simulator = IntegratedSimulator(
                config, architecture=architecture, hydraulic_design=design
            )
            result = simulator.run(scenario, "baseline")
            feasible, reason = _architecture_feasibility(result)
            rows.append({
                "scenario": scenario_name,
                "architecture": architecture,
                "feasible": feasible,
                "infeasibility_reason": reason,
                "pump_scale": design.pump_scale,
                "radiator_ua_scale": design.radiator_ua_scale,
                "local_resistance_scale": design.local_resistance_scale,
                **result.metrics,
            })
    return pd.DataFrame(rows)


def run_architecture_sizing(config: ProjectConfig, scenario_name: str = "hill_high_load",
                            duration_s: int = 900,
                            pump_scales: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2),
                            radiator_ua_scales: tuple[float, ...] = (0.75, 1.0, 1.25)) -> pd.DataFrame:
    """Scan pump/radiator sizes and retain explicit infeasibility reasons."""
    scenario = make_scenario(
        scenario_name, duration_s, config.simulation.dt_s, config.seed + 700
    )
    rows = []
    for architecture in architecture_names():
        for pump_scale in pump_scales:
            for radiator_scale in radiator_ua_scales:
                design = HydraulicDesign(
                    pump_scale=float(pump_scale), radiator_ua_scale=float(radiator_scale)
                )
                try:
                    simulator = IntegratedSimulator(
                        config, architecture=architecture, hydraulic_design=design
                    )
                    result = simulator.run(scenario, "baseline")
                    feasible, reason = _architecture_feasibility(result)
                    battery_capacity = simulator.battery_network.solve(
                        simulator.battery_pump_model, 1.0, 25.0
                    )
                    powertrain_capacity = simulator.powertrain_network.solve(
                        simulator.powertrain_pump_model, 1.0, 25.0
                    )
                    reasons = [] if reason == "ok" else reason.split(";")
                    if not battery_capacity.converged or battery_capacity.point.mass_flow_kg_s < 0.15:
                        reasons.append("battery_design_flow_envelope")
                    if not powertrain_capacity.converged or powertrain_capacity.point.mass_flow_kg_s < 0.18:
                        reasons.append("powertrain_design_flow_envelope")
                    feasible = not reasons
                    reason = "ok" if feasible else ";".join(reasons)
                    row = {
                        "architecture": architecture,
                        "scenario": scenario_name,
                        "pump_scale": pump_scale,
                        "radiator_ua_scale": radiator_scale,
                        "feasible": feasible,
                        "infeasibility_reason": reason,
                        "max_battery_flow_kg_s": float(result.timeseries["battery_flow_kg_s"].max()),
                        "max_powertrain_flow_kg_s": float(result.timeseries["powertrain_flow_kg_s"].max()),
                        "battery_design_flow_capacity_kg_s": battery_capacity.point.mass_flow_kg_s,
                        "powertrain_design_flow_capacity_kg_s": powertrain_capacity.point.mass_flow_kg_s,
                        **result.metrics,
                    }
                except RuntimeError as exc:
                    row = {
                        "architecture": architecture,
                        "scenario": scenario_name,
                        "pump_scale": pump_scale,
                        "radiator_ua_scale": radiator_scale,
                        "feasible": False,
                        "infeasibility_reason": f"solver:{type(exc).__name__}",
                        "max_battery_flow_kg_s": 0.0,
                        "max_powertrain_flow_kg_s": 0.0,
                    }
                rows.append(row)
    return pd.DataFrame(rows)


def _config_sha256(project_root: Path) -> str:
    return hashlib.sha256(
        (project_root / "configs" / "default_config.yaml").read_bytes()
    ).hexdigest()


def _plant_sha256(project_root: Path) -> str:
    return compute_plant_sha256(project_root)


def run_all(project_root: str | Path = PROJECT_ROOT, quick: bool = False,
            run_id: str | None = None) -> dict:
    root = Path(project_root)
    config = load_config(root / "configs" / "default_config.yaml")
    set_seed(config.seed)
    profile = "quick" if quick else "formal"
    layout = create_run_layout(root, profile, run_id=run_id)
    data_path = layout.data_dir / "thermal_load_episodes.csv"
    model_dir = layout.model_dir
    episode_count = 6 if quick else 24
    duration = 650 if quick else 1200
    frame = generate_simulation_dataset(config, data_path, episode_count, duration, config.seed)
    training = train_predictor_from_frame(config, frame, model_dir, quick)
    predictor = ThermalLoadPredictor(model_dir)
    selected_scenarios = ["urban_hot", "cold_start"] if quick else scenario_names()
    comparison, trajectories = run_strategy_comparison(
        config, selected_scenarios,
        600 if quick else 1800, predictor,
    )
    tables = layout.tables_dir
    figures = layout.figures_dir
    logs = layout.logs_dir
    for path in (tables, figures, logs):
        path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tables / "strategy_comparison.csv", index=False)
    robustness = run_robustness_checks(config, 300 if quick else 900)
    robustness.to_csv(tables / "robustness_checks.csv", index=False)
    plot_strategy_comparison(comparison, figures / "strategy_comparison.png")
    plot_training_history(training.history, figures / "training_history.png")
    for key, trajectory in trajectories.items():
        plot_scenario_overview(trajectory, figures / f"overview_{key[0]}_{key[1]}.png")

    config_hash = _config_sha256(root)
    plant_hash = _plant_sha256(root)
    model_manifest = {
        "schema_version": 1,
        "profile": profile,
        "config_sha256": config_hash,
        "plant_sha256": plant_hash,
        "episode_count": episode_count,
        "training_best_epoch": training.best_epoch,
        "forecast_metrics": training.metrics,
    }
    (model_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2), encoding="utf-8"
    )

    expected_files = [
        data_path.relative_to(layout.run_root),
        *[path.relative_to(layout.run_root) for path in model_dir.iterdir() if path.is_file()],
        (tables / "strategy_comparison.csv").relative_to(layout.run_root),
        (tables / "robustness_checks.csv").relative_to(layout.run_root),
        *[path.relative_to(layout.run_root) for path in figures.glob("*.png")],
    ]
    manifest = write_run_manifest(layout, {
        "config_sha256": config_hash,
        "plant_sha256": plant_hash,
        "seed": config.seed,
        "quick": quick,
        "episode_count": episode_count,
        "training_best_epoch": training.best_epoch,
        "forecast_metrics": training.metrics,
        "comparison_rows": len(comparison),
        "robustness_cases": len(robustness),
        "scenarios": selected_scenarios,
        "strategies": ["baseline", "predictive"],
    }, expected_files)
    if quick:
        update_latest_pointer(layout)
    else:
        promote_run(layout)
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    return manifest
