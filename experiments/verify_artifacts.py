"""Verify provenance, hashes, completeness, and numerical validity of a formal run."""

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import ROOT
from ev_thermal.artifacts import (
    ArtifactValidationError,
    compute_plant_sha256,
    resolve_latest_run,
    validate_run_artifacts,
    validate_upgrade_artifacts,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_documented_summary(root: Path, run_root: Path) -> None:
    """Reject a result summary whose key claims drift from machine-readable evidence."""

    document = (root / "docs" / "results_summary.md").read_text(encoding="utf-8")
    metrics = _load_json(root / "models" / "test_metrics.json")
    run_manifest = _load_json(run_root / "results" / "logs" / "run_manifest.json")
    calibration = _load_json(root / "results" / "calibration" / "maturity_statement.json")
    architecture = _load_json(root / "results" / "architecture" / "architecture_summary.json")
    optimization = _load_json(
        root / "results" / "optimization" / "joint_optimization_summary.json"
    )

    with (root / "results" / "tables" / "strategy_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_scenario: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], {})[row["strategy"]] = row
    battery_delta = sum(
        float(pair["predictive"]["max_battery_temp_c"])
        - float(pair["baseline"]["max_battery_temp_c"])
        for pair in by_scenario.values()
    ) / len(by_scenario)

    with (root / "results" / "charging" / "preconditioning_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        charging_rows = list(csv.DictReader(handle))
    cold = {
        row["strategy"]: row
        for row in charging_rows
        if row["scenario"] == "cold_arrival"
    }

    expected_tokens = {
        "formal run ID": run_manifest["run_id"],
        "battery MAE": f'{metrics["mae_battery_heat_w"]:.2f} W',
        "battery R2": f'{metrics["r2_battery_heat_w"]:.3f}',
        "mean battery peak delta": f'{abs(battery_delta):.3f} °C',
        "calibration improvement": f'{calibration["holdout_improvement_pct"]:.1f}%',
        "architecture comparison rows": f'{architecture["comparison_rows"]}行比较',
        "architecture sizing rows": f'{architecture["sizing_rows"]}个泵/散热器规格点',
        "cold no-preheat charge time": f'{float(cold["none"]["charge_time_s"]) / 60:.2f} min',
        "cold rule charge time": f'{float(cold["rule"]["charge_time_s"]) / 60:.2f} min',
        "optimization candidates": f'{optimization["candidate_count"]}个候选',
        "Pareto representatives": f'{optimization["pareto_count"]}个Pareto代表点',
    }
    missing = [label for label, token in expected_tokens.items() if token not in document]
    if missing:
        details = ", ".join(
            f"{label}={expected_tokens[label]!r}" for label in missing
        )
        raise ArtifactValidationError(f"docs/results_summary.md is stale: {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Run directory to verify; defaults to artifacts/latest/formal.json",
    )
    args = parser.parse_args()
    try:
        run_root = args.run_root.resolve() if args.run_root else resolve_latest_run(ROOT, "formal")
        summary = validate_run_artifacts(
            run_root,
            require_formal=True,
            expected_plant_sha256=compute_plant_sha256(ROOT),
        )
        upgrade_summary = validate_upgrade_artifacts(ROOT)
        verify_documented_summary(ROOT, run_root)
    except ArtifactValidationError as exc:
        raise SystemExit(f"Artifact verification failed: {exc}") from exc
    print(
        "Verified formal run {run_id}: {episode_count} episodes, "
        "{comparison_rows} comparison rows, {figure_count} figures, "
        "and {artifact_count} hashed artifacts.".format(**summary)
    )
    print(
        "Verified upgrade evidence: {upgrade_artifact_count} hashed artifacts, "
        "{architecture_rows} architecture rows, {charging_rows} charging rows, "
        "{optimization_candidates} optimization candidates, and {pareto_rows} Pareto rows.".format(
            **upgrade_summary
        )
    )
    print("Verified documented summary against machine-readable evidence.")


if __name__ == "__main__":
    main()
