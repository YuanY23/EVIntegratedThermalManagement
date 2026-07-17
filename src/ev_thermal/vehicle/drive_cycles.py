"""Built-in analytical cycles and external CSV cycle import."""

from pathlib import Path

import numpy as np
import pandas as pd


def built_in_speed(name: str, time_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate deterministic representative cycles in m/s, not legal standards."""
    t = np.asarray(time_s, dtype=float)
    if name == "urban":
        base = 10 + 6 * np.sin(2 * np.pi * t / 110) + 3 * np.sin(2 * np.pi * t / 37)
        stops = (np.sin(2 * np.pi * t / 180) < -0.75)
        speed = np.where(stops, 0.0, base)
    elif name == "highway":
        speed = 27 + 3 * np.sin(2 * np.pi * t / 240) + 1.2 * np.sin(2 * np.pi * t / 43)
    elif name == "aggressive":
        speed = 18 + 11 * np.sin(2 * np.pi * t / 65) + 4 * np.sin(2 * np.pi * t / 17)
    elif name == "hill":
        speed = 17 + 5 * np.sin(2 * np.pi * t / 150)
    else:
        speed = 16 + 8 * np.sin(2 * np.pi * t / 130) + 2 * np.sin(2 * np.pi * t / 29)
    # Small deterministic perturbation avoids perfectly periodic training data.
    speed = speed + rng.normal(0, 0.15, size=t.size)
    return np.clip(speed, 0.0, 36.0)


def load_cycle_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"time_s", "speed_mps"}
    if not required.issubset(frame.columns):
        raise ValueError(f"cycle CSV must contain {sorted(required)}")
    return frame.sort_values("time_s").reset_index(drop=True)

