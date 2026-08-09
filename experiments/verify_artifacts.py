"""Verify provenance, hashes, completeness, and numerical validity of a formal run."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from ev_thermal.artifacts import (
    ArtifactValidationError,
    compute_plant_sha256,
    resolve_latest_run,
    validate_run_artifacts,
)


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
    except ArtifactValidationError as exc:
        raise SystemExit(f"Artifact verification failed: {exc}") from exc
    print(
        "Verified formal run {run_id}: {episode_count} episodes, "
        "{comparison_rows} comparison rows, {figure_count} figures, "
        "and {artifact_count} hashed artifacts.".format(**summary)
    )


if __name__ == "__main__":
    main()
