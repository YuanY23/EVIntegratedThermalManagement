"""Fail fast when formal experiment artifacts are missing or numerically invalid."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT


def main() -> None:
    required = [
        ROOT / "data" / "processed" / "thermal_load_episodes.csv",
        ROOT / "models" / "thermal_load_lstm.pt",
        ROOT / "models" / "feature_scaler.joblib",
        ROOT / "models" / "target_scaler.joblib",
        ROOT / "models" / "test_metrics.json",
        ROOT / "results" / "tables" / "strategy_comparison.csv",
        ROOT / "results" / "tables" / "robustness_checks.csv",
        ROOT / "results" / "figures" / "strategy_comparison.png",
        ROOT / "results" / "logs" / "run_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing/empty artifacts:\n" + "\n".join(missing))

    comparison = pd.read_csv(required[5])
    numeric = comparison.select_dtypes(include=[np.number])
    if len(comparison) != 12 or not np.isfinite(numeric.to_numpy()).all():
        raise SystemExit("Strategy comparison must have 12 finite rows")
    if comparison["thermal_balance_error_pct"].max() >= 2.0:
        raise SystemExit("Thermal balance acceptance criterion failed")
    metrics = json.loads((ROOT / "models" / "test_metrics.json").read_text(encoding="utf-8"))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise SystemExit("Forecast metrics contain non-finite values")
    figures = list((ROOT / "results" / "figures").glob("*.png"))
    if len(figures) < 14 or min(path.stat().st_size for path in figures) < 1000:
        raise SystemExit("Expected at least 14 non-empty result figures")
    print(f"Verified {len(required)} core artifacts, {len(comparison)} comparison rows, and {len(figures)} figures.")


if __name__ == "__main__":
    main()

