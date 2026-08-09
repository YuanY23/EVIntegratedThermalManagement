"""Compare baseline and prediction-enhanced integrated thermal management."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from ev_thermal.config import load_config
from ev_thermal.pipeline import run_strategy_comparison
from ev_thermal.prediction.predictor import ThermalLoadPredictor
from ev_thermal.simulation.scenarios import scenario_names
from ev_thermal.visualization import plot_strategy_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Existing isolated run directory used for model and outputs")
    parser.add_argument("--without-model", action="store_true",
                        help="Use deterministic route-preview proxy instead of the saved LSTM")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default_config.yaml")
    run_root = args.run_root.resolve()
    predictor = None if args.without_model else ThermalLoadPredictor(run_root / "models")
    table, _ = run_strategy_comparison(cfg, scenario_names(), args.duration, predictor)
    output = run_root / "results" / "tables" / "strategy_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    plot_strategy_comparison(table, run_root / "results" / "figures" / "strategy_comparison.png")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
