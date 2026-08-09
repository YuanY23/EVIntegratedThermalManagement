import json
from pathlib import Path

import pandas as pd
import pytest

from ev_thermal.artifacts import (
    ArtifactValidationError,
    _validate_pareto_by_scenario,
    _validate_upgrade_suite_manifests,
    compute_plant_sha256,
    create_run_layout,
    hash_source_tree,
    promote_run,
    validate_run_artifacts,
    validate_upgrade_artifacts,
    write_upgrade_suite_manifest,
    write_run_manifest,
)
from ev_thermal.simulation.scenarios import scenario_names


def test_quick_and_formal_runs_use_disjoint_directories(tmp_path):
    quick = create_run_layout(tmp_path, "quick", run_id="quick-001")
    formal = create_run_layout(tmp_path, "formal", run_id="formal-001")

    assert quick.run_root != formal.run_root
    assert quick.data_dir != tmp_path / "data" / "processed"
    assert formal.model_dir != tmp_path / "models"
    with pytest.raises(ValueError, match="Only formal"):
        promote_run(quick)
    with pytest.raises(FileExistsError, match="already exists"):
        create_run_layout(tmp_path, "quick", run_id="quick-001")


def test_source_hash_is_independent_of_checkout_location(tmp_path):
    first = tmp_path / "checkout-a"
    second = tmp_path / "checkout-b"
    for root in (first, second):
        source = root / "src" / "plant.py"
        source.parent.mkdir(parents=True)
        source.write_text("PARAMETER = 42\n", encoding="utf-8")

    first_hash = hash_source_tree([first / "src" / "plant.py"], base_dir=first)
    second_hash = hash_source_tree([second / "src" / "plant.py"], base_dir=second)

    assert first_hash == second_hash


def test_plant_hash_includes_parameter_registry(tmp_path):
    source = tmp_path / "src" / "ev_thermal" / "plant.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "default_config.yaml").write_text("seed: 42\n", encoding="utf-8")
    registry = configs / "parameter_registry.yaml"
    registry.write_text("parameters: []\n", encoding="utf-8")

    before = compute_plant_sha256(tmp_path)
    registry.write_text("parameters: [{name: changed}]\n", encoding="utf-8")

    assert compute_plant_sha256(tmp_path) != before


def _write_minimal_formal_run(root: Path):
    layout = create_run_layout(root, "formal", run_id="formal-test")
    pd.DataFrame({"episode_id": list(range(24)), "value": list(range(24))}).to_csv(
        layout.data_dir / "thermal_load_episodes.csv", index=False
    )
    comparison = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "strategy": strategy,
                "thermal_balance_error_pct": 0.1,
                "net_energy_kwh": 1.0,
            }
            for scenario in scenario_names()
            for strategy in ("baseline", "predictive")
        ]
    )
    comparison.to_csv(layout.tables_dir / "strategy_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"case": "invalid_forecast", "thermal_balance_error_pct": 0.1},
            {"case": "high_heat_bias", "thermal_balance_error_pct": 0.1},
        ]
    ).to_csv(layout.tables_dir / "robustness_checks.csv", index=False)

    metrics = {"mae_battery_heat_w": 1.0, "r2_battery_heat_w": 0.9}
    (layout.model_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (layout.model_dir / "training_history.json").write_text("[]", encoding="utf-8")
    (layout.model_dir / "model_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "formal",
                "config_sha256": "config-hash",
                "plant_sha256": "plant-hash",
                "episode_count": 24,
                "forecast_metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    for name in ("thermal_load_lstm.pt", "feature_scaler.joblib", "target_scaler.joblib"):
        (layout.model_dir / name).write_bytes(b"model")
    figure_names = ["strategy_comparison.png", "training_history.png"] + [
        f"overview_{scenario}_{strategy}.png"
        for scenario in scenario_names()
        for strategy in ("baseline", "predictive")
    ]
    for name in figure_names:
        (layout.figures_dir / name).write_bytes(b"x" * 1200)

    expected_files = [
        path.relative_to(layout.run_root)
        for path in layout.run_root.rglob("*")
        if path.is_file()
    ]
    write_run_manifest(
        layout,
        {
            "config_sha256": "config-hash",
            "plant_sha256": "plant-hash",
            "seed": 42,
            "episode_count": 24,
            "training_best_epoch": 3,
            "forecast_metrics": metrics,
            "comparison_rows": 12,
            "robustness_cases": 2,
            "scenarios": scenario_names(),
            "strategies": ["baseline", "predictive"],
        },
        expected_files,
    )
    return layout


def test_formal_validation_checks_semantics_and_hashes(tmp_path):
    layout = _write_minimal_formal_run(tmp_path)

    summary = validate_run_artifacts(layout.run_root, require_formal=True)
    assert summary["comparison_rows"] == 12
    assert summary["episode_count"] == 24

    comparison_path = layout.tables_dir / "strategy_comparison.csv"
    comparison_path.write_text(comparison_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="hash"):
        validate_run_artifacts(layout.run_root, require_formal=True)


def test_formal_validation_rejects_an_unhashed_required_artifact(tmp_path):
    layout = _write_minimal_formal_run(tmp_path)
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"].pop("models/thermal_load_lstm.pt")
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="hash set mismatch"):
        validate_run_artifacts(layout.run_root, require_formal=True)


def test_upgrade_suite_manifest_binds_complete_evidence_to_plant(tmp_path, monkeypatch):
    import ev_thermal.artifacts as artifacts

    monkeypatch.setattr(artifacts, "UPGRADE_SUITE_ARTIFACTS", {"demo": ("evidence.csv",)})
    monkeypatch.setattr(artifacts, "compute_plant_sha256", lambda root: "plant-hash")
    suite_root = tmp_path / "results" / "demo"
    suite_root.mkdir(parents=True)
    evidence = suite_root / "evidence.csv"
    evidence.write_text("value\n1\n", encoding="utf-8")

    write_upgrade_suite_manifest(tmp_path, "demo")
    _validate_upgrade_suite_manifests(tmp_path)
    evidence.write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="artifact hash mismatch"):
        _validate_upgrade_suite_manifests(tmp_path)


def test_upgrade_validation_reports_missing_manifest(tmp_path):
    with pytest.raises(ArtifactValidationError, match="upgrade manifest"):
        validate_upgrade_artifacts(tmp_path)


def test_pareto_validation_never_compares_candidates_across_scenarios():
    candidates = pd.DataFrame(
        [
            {
                "scenario": "cold",
                "feasible": True,
                "charge_time_s": 20.0,
                "preconditioning_energy_kwh": 2.0,
                "relative_aging_damage": 2.0,
            },
            {
                "scenario": "hot",
                "feasible": True,
                "charge_time_s": 10.0,
                "preconditioning_energy_kwh": 1.0,
                "relative_aging_damage": 1.0,
            },
        ]
    )
    pareto = candidates.copy()

    _validate_pareto_by_scenario(candidates, pareto)
