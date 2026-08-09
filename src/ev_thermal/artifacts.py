"""Run-scoped experiment artifacts and verified formal-result promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from .simulation.scenarios import scenario_names


MANIFEST_SCHEMA_VERSION = 2
FORMAL_EPISODE_COUNT = 24
STRATEGIES = ("baseline", "predictive")


class ArtifactValidationError(ValueError):
    """Raised when a run cannot be trusted as a complete experiment result."""


@dataclass(frozen=True)
class RunLayout:
    project_root: Path
    run_id: str
    profile: str
    run_root: Path
    data_dir: Path
    model_dir: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.logs_dir / "run_manifest.json"


def _validate_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized not in {"quick", "formal"}:
        raise ValueError("profile must be 'quick' or 'formal'")
    return normalized


def _validate_run_id(run_id: str) -> str:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    return run_id


def create_run_layout(project_root: str | Path, profile: str,
                      run_id: str | None = None) -> RunLayout:
    """Create an isolated directory tree for one quick or formal run."""
    root = Path(project_root).resolve()
    profile = _validate_profile(profile)
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{profile}-{uuid.uuid4().hex[:8]}"
    run_id = _validate_run_id(run_id)
    run_root = root / "artifacts" / "runs" / run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run directory already exists and is not empty: {run_root}")
    layout = RunLayout(
        project_root=root,
        run_id=run_id,
        profile=profile,
        run_root=run_root,
        data_dir=run_root / "data" / "processed",
        model_dir=run_root / "models",
        tables_dir=run_root / "results" / "tables",
        figures_dir=run_root / "results" / "figures",
        logs_dir=run_root / "results" / "logs",
    )
    for directory in (
        layout.data_dir,
        layout.model_dir,
        layout.tables_dir,
        layout.figures_dir,
        layout.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_tree(paths: list[Path], base_dir: str | Path | None = None) -> str:
    """Hash named source/config files, including their relative names and bytes."""
    digest = hashlib.sha256()
    resolved = sorted(path.resolve() for path in paths if path.is_file())
    if not resolved:
        raise ValueError("at least one source file is required for a plant hash")
    base = Path(base_dir).resolve() if base_dir is not None else Path(Path(*resolved[0].parts[:1]))
    for path in resolved:
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"source file is outside base_dir: {path}") from exc
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_plant_sha256(project_root: str | Path) -> str:
    """Hash the executable thermal-management source and default configuration."""
    root = Path(project_root).resolve()
    source_files = list((root / "src" / "ev_thermal").rglob("*.py"))
    source_files.append(root / "configs" / "default_config.yaml")
    return hash_source_tree(source_files, base_dir=root)


def _safe_relative_path(value: str | Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path must stay inside the run: {value}")
    return relative


def write_run_manifest(layout: RunLayout, metadata: dict,
                       expected_files: list[str | Path]) -> dict:
    """Write a verified-state manifest with content hashes for all run artifacts."""
    relative_files = sorted({_safe_relative_path(value) for value in expected_files}, key=str)
    hashes: dict[str, str] = {}
    for relative in relative_files:
        path = layout.run_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ArtifactValidationError(f"Missing or empty artifact: {relative.as_posix()}")
        hashes[relative.as_posix()] = _sha256(path)
    reserved = {"schema_version", "run_id", "profile", "status", "created_at_utc", "artifact_sha256"}
    overlap = reserved.intersection(metadata)
    if overlap:
        raise ValueError(f"metadata uses reserved manifest keys: {sorted(overlap)}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": layout.run_id,
        "profile": layout.profile,
        "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "artifact_sha256": hashes,
    }
    layout.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_manifest(run_root: Path) -> tuple[Path, dict]:
    manifest_path = run_root / "results" / "logs" / "run_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"Missing run manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactValidationError(f"Invalid run manifest: {exc}") from exc
    return manifest_path, manifest


def validate_run_artifacts(run_root: str | Path, require_formal: bool = True,
                           expected_plant_sha256: str | None = None) -> dict:
    """Validate provenance, hashes, completeness, and numerical result semantics."""
    root = Path(run_root).resolve()
    _, manifest = _load_manifest(root)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported manifest schema version")
    if require_formal and manifest.get("profile") != "formal":
        raise ArtifactValidationError("A formal run is required")
    if manifest.get("status") not in {"verified", "promoted"}:
        raise ArtifactValidationError("Run status is not verified")
    if not manifest.get("config_sha256") or not manifest.get("plant_sha256"):
        raise ArtifactValidationError("Manifest is missing config or plant provenance")
    if expected_plant_sha256 is not None and manifest.get("plant_sha256") != expected_plant_sha256:
        raise ArtifactValidationError("Formal artifacts were generated by a different plant version")

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ArtifactValidationError("Manifest has no artifact hashes")
    for raw_relative, expected_hash in hashes.items():
        relative = _safe_relative_path(raw_relative)
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ArtifactValidationError(f"Missing or empty artifact: {raw_relative}")
        if _sha256(path) != expected_hash:
            raise ArtifactValidationError(f"Artifact hash mismatch: {raw_relative}")

    data_path = root / "data" / "processed" / "thermal_load_episodes.csv"
    comparison_path = root / "results" / "tables" / "strategy_comparison.csv"
    robustness_path = root / "results" / "tables" / "robustness_checks.csv"
    metrics_path = root / "models" / "test_metrics.json"
    model_manifest_path = root / "models" / "model_manifest.json"
    required = [
        data_path,
        root / "models" / "thermal_load_lstm.pt",
        root / "models" / "feature_scaler.joblib",
        root / "models" / "target_scaler.joblib",
        root / "models" / "training_history.json",
        model_manifest_path,
        metrics_path,
        comparison_path,
        robustness_path,
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ArtifactValidationError("Missing core artifacts: " + ", ".join(missing))

    dataset = pd.read_csv(data_path)
    episode_count = int(dataset["episode_id"].nunique()) if "episode_id" in dataset else 0
    expected_episode_count = FORMAL_EPISODE_COUNT if require_formal else int(manifest.get("episode_count", 0))
    if episode_count != expected_episode_count or manifest.get("episode_count") != expected_episode_count:
        raise ArtifactValidationError(
            f"Expected {expected_episode_count} episodes, found {episode_count}"
        )

    comparison = pd.read_csv(comparison_path)
    expected_pairs = {(scenario, strategy) for scenario in scenario_names() for strategy in STRATEGIES}
    actual_pairs = set(zip(comparison.get("scenario", []), comparison.get("strategy", [])))
    if len(comparison) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ArtifactValidationError("Strategy comparison is not the complete 6x2 scenario-strategy matrix")
    numeric = comparison.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy()).all():
        raise ArtifactValidationError("Strategy comparison contains non-finite numeric values")
    if "thermal_balance_error_pct" not in comparison or comparison["thermal_balance_error_pct"].max() >= 2.0:
        raise ArtifactValidationError("Thermal balance acceptance criterion failed")
    if manifest.get("comparison_rows") != len(comparison):
        raise ArtifactValidationError("Manifest comparison row count does not match the table")

    robustness = pd.read_csv(robustness_path)
    if len(robustness) != manifest.get("robustness_cases"):
        raise ArtifactValidationError("Manifest robustness count does not match the table")
    robustness_numeric = robustness.select_dtypes(include=[np.number])
    if robustness_numeric.empty or not np.isfinite(robustness_numeric.to_numpy()).all():
        raise ArtifactValidationError("Robustness checks contain non-finite numeric values")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in metrics.values()):
        raise ArtifactValidationError("Forecast metrics contain non-finite values")
    if metrics != manifest.get("forecast_metrics"):
        raise ArtifactValidationError("Manifest forecast metrics do not match the model metrics")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    for key in ("profile", "config_sha256", "plant_sha256", "episode_count", "forecast_metrics"):
        if model_manifest.get(key) != manifest.get(key):
            raise ArtifactValidationError(f"Model manifest does not match run manifest: {key}")

    expected_figures = {"strategy_comparison.png", "training_history.png"} | {
        f"overview_{scenario}_{strategy}.png" for scenario, strategy in expected_pairs
    }
    figures_dir = root / "results" / "figures"
    actual_figures = {path.name for path in figures_dir.glob("*.png")}
    if actual_figures != expected_figures:
        raise ArtifactValidationError("Figure set does not exactly match the formal scenario matrix")
    if any((figures_dir / name).stat().st_size < 1000 for name in expected_figures):
        raise ArtifactValidationError("One or more result figures are unexpectedly small")

    return {
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "episode_count": episode_count,
        "comparison_rows": len(comparison),
        "figure_count": len(expected_figures),
        "artifact_count": len(hashes),
    }


def promote_run(layout: RunLayout) -> dict:
    """Publish a validated formal run to legacy paths and update the formal pointer."""
    if layout.profile != "formal":
        raise ValueError("Only formal runs can be promoted")
    summary = validate_run_artifacts(
        layout.run_root,
        require_formal=True,
        expected_plant_sha256=compute_plant_sha256(layout.project_root),
    )
    _, manifest = _load_manifest(layout.run_root)
    for raw_relative in manifest["artifact_sha256"]:
        relative = _safe_relative_path(raw_relative)
        destination = layout.project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(layout.run_root / relative, destination)

    manifest["status"] = "promoted"
    manifest["promoted_at_utc"] = datetime.now(timezone.utc).isoformat()
    layout.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    canonical_manifest = layout.project_root / "results" / "logs" / "run_manifest.json"
    canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(layout.manifest_path, canonical_manifest)

    latest_dir = layout.project_root / "artifacts" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pointer = {
        "run_id": layout.run_id,
        "profile": layout.profile,
        "run_root": layout.run_root.relative_to(layout.project_root).as_posix(),
    }
    (latest_dir / "formal.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return summary


def update_latest_pointer(layout: RunLayout) -> Path:
    """Point to the latest unpromoted run without changing canonical results."""
    latest_dir = layout.project_root / "artifacts" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = latest_dir / f"{layout.profile}.json"
    pointer_path.write_text(
        json.dumps(
            {
                "run_id": layout.run_id,
                "profile": layout.profile,
                "run_root": layout.run_root.relative_to(layout.project_root).as_posix(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pointer_path


def resolve_latest_run(project_root: str | Path, profile: str = "formal") -> Path:
    """Resolve the latest run pointer while preventing path traversal."""
    root = Path(project_root).resolve()
    profile = _validate_profile(profile)
    pointer_path = root / "artifacts" / "latest" / f"{profile}.json"
    if not pointer_path.is_file():
        raise ArtifactValidationError(f"No latest {profile} run pointer exists")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("profile") != profile:
        raise ArtifactValidationError("Latest-run pointer profile mismatch")
    relative = _safe_relative_path(pointer.get("run_root", ""))
    run_root = (root / relative).resolve()
    runs_root = (root / "artifacts" / "runs").resolve()
    if runs_root not in run_root.parents:
        raise ArtifactValidationError("Latest-run pointer escapes the runs directory")
    return run_root
