"""Train the 300-second multi-output LSTM from generated episodes."""

import argparse
import pandas as pd

from _bootstrap import ROOT
from ev_thermal.config import load_config
from ev_thermal.pipeline import train_predictor_from_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default_config.yaml")
    frame = pd.read_csv(ROOT / "data" / "processed" / "thermal_load_episodes.csv")
    result = train_predictor_from_frame(cfg, frame, ROOT / "models", args.quick)
    print(f"Best epoch: {result.best_epoch}; validation MSE: {result.best_validation_loss:.6f}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()

