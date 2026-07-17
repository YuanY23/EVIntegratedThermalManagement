"""Generate physics-simulation episodes for LSTM training."""

import argparse

from _bootstrap import ROOT
from ev_thermal.config import load_config
from ev_thermal.pipeline import generate_simulation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--duration", type=int, default=1200, help="Episode duration in seconds")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default_config.yaml")
    path = ROOT / "data" / "processed" / "thermal_load_episodes.csv"
    frame = generate_simulation_dataset(cfg, path, args.episodes, args.duration, cfg.seed)
    print(f"Saved {len(frame):,} rows from {args.episodes} episodes to {path}")


if __name__ == "__main__":
    main()

