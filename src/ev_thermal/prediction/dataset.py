"""Episode-safe sequence construction and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceArrays:
    features: np.ndarray
    targets: np.ndarray
    episode_ids: tuple[int, ...]


@dataclass(frozen=True)
class DatasetSplit:
    train: SequenceArrays
    validation: SequenceArrays
    test: SequenceArrays
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


class ThermalSequenceDataset(Dataset):
    def __init__(self, arrays: SequenceArrays):
        self.features = torch.as_tensor(arrays.features, dtype=torch.float32)
        self.targets = torch.as_tensor(arrays.targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class SequenceDatasetBuilder:
    """Build history-to-future-load windows without crossing episode borders."""

    feature_columns = (
        "speed_mps", "ambient_temp_c", "battery_temp_c", "motor_temp_c",
        "cabin_temp_c", "battery_power_w", "pump_fraction", "time_s",
        # These loads are available online from the same physical estimators used
        # by the energy manager. Their histories provide a physically meaningful
        # decoder initial state without leaking any future target values.
        "battery_heat_w", "powertrain_heat_w", "cabin_load_w",
    )
    target_columns = ("battery_heat_w", "powertrain_heat_w", "cabin_load_w")

    def __init__(self, history_steps: int = 60, horizon_steps: int = 60, stride: int = 3):
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.stride = stride

    def _episode_split(self, ids: list[int]) -> tuple[set[int], set[int], set[int]]:
        unique = sorted(ids)
        if len(unique) < 3:
            raise ValueError("at least three episodes are required for leakage-safe splits")
        n_train = max(1, int(round(0.6 * len(unique))))
        n_val = max(1, int(round(0.2 * len(unique))))
        if n_train + n_val >= len(unique):
            n_train = len(unique) - 2
            n_val = 1
        return set(unique[:n_train]), set(unique[n_train:n_train + n_val]), set(unique[n_train + n_val:])

    def _windows(self, frame: pd.DataFrame, ids: set[int], feature_scaler: StandardScaler,
                 target_scaler: StandardScaler) -> SequenceArrays:
        xs, ys, used_ids = [], [], []
        for episode_id in sorted(ids):
            episode = frame.loc[frame["episode_id"] == episode_id].sort_values("time_s")
            raw_x = episode.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
            raw_y = episode.loc[:, self.target_columns].to_numpy(dtype=np.float32)
            x = feature_scaler.transform(raw_x).astype(np.float32)
            y = target_scaler.transform(raw_y).astype(np.float32)
            last_start = len(episode) - self.history_steps - self.horizon_steps
            for start in range(0, last_start + 1, self.stride):
                split = start + self.history_steps
                xs.append(x[start:split])
                ys.append(y[split:split + self.horizon_steps])
                used_ids.append(int(episode_id))
        if not xs:
            return SequenceArrays(
                np.empty((0, self.history_steps, len(self.feature_columns)), dtype=np.float32),
                np.empty((0, self.horizon_steps, len(self.target_columns)), dtype=np.float32), tuple())
        return SequenceArrays(np.stack(xs), np.stack(ys), tuple(sorted(set(used_ids))))

    def build(self, frame: pd.DataFrame) -> DatasetSplit:
        required = {"episode_id", *self.feature_columns, *self.target_columns}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"dataset missing columns: {sorted(missing)}")
        train_ids, val_ids, test_ids = self._episode_split(frame["episode_id"].unique().tolist())
        train_rows = frame[frame["episode_id"].isin(train_ids)]
        # Flattening time rows is correct for per-channel affine scaling; fitting
        # exclusively on train episodes prevents future/test distribution leakage.
        feature_scaler = StandardScaler().fit(train_rows.loc[:, self.feature_columns].to_numpy())
        target_scaler = StandardScaler().fit(train_rows.loc[:, self.target_columns].to_numpy())
        return DatasetSplit(
            self._windows(frame, train_ids, feature_scaler, target_scaler),
            self._windows(frame, val_ids, feature_scaler, target_scaler),
            self._windows(frame, test_ids, feature_scaler, target_scaler),
            feature_scaler, target_scaler,
        )
