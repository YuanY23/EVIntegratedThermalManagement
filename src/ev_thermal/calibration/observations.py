"""Validated tabular observation contract for battery thermal calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "dataset_id",
    "maturity",
    "episode_id",
    "time_s",
    "heat_generation_w",
    "coolant_temp_c",
    "coolant_ua_w_k",
    "core_temp_c",
    "surface_temp_c",
    "core_temp_std_c",
    "surface_temp_std_c",
)
NUMERIC_COLUMNS = tuple(column for column in REQUIRED_COLUMNS if column not in {"dataset_id", "maturity"})
MATURITY_LABELS = {"synthetic", "measured"}


class ObservationValidationError(ValueError):
    """Raised when calibration observations violate the public data contract."""


@dataclass(frozen=True)
class ObservationDataset:
    frame: pd.DataFrame
    dataset_id: str
    maturity: str

    @classmethod
    def from_csv(cls, path: str | Path) -> "ObservationDataset":
        return cls.from_frame(pd.read_csv(path))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ObservationDataset":
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ObservationValidationError(f"Missing observation columns: {missing}")
        if frame.empty:
            raise ObservationValidationError("Observation dataset is empty")
        dataset_ids = frame["dataset_id"].dropna().astype(str).unique()
        if len(dataset_ids) != 1 or not dataset_ids[0]:
            raise ObservationValidationError("Each file must contain exactly one dataset_id")
        maturities = frame["maturity"].dropna().astype(str).unique()
        if len(maturities) != 1:
            raise ObservationValidationError("Each dataset must use one maturity label")
        if maturities[0] not in MATURITY_LABELS:
            raise ObservationValidationError(f"Unsupported maturity label: {maturities[0]}")

        checked = frame.copy()
        try:
            checked.loc[:, NUMERIC_COLUMNS] = checked.loc[:, NUMERIC_COLUMNS].apply(pd.to_numeric)
        except (TypeError, ValueError) as exc:
            raise ObservationValidationError("Observation numeric fields contain invalid values") from exc
        if not np.isfinite(checked.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=float)).all():
            raise ObservationValidationError("Observation numeric fields must be finite")
        if (checked[["core_temp_std_c", "surface_temp_std_c"]] <= 0).any().any():
            raise ObservationValidationError("Measurement standard deviations must be positive")
        if (checked["coolant_ua_w_k"] < 0).any():
            raise ObservationValidationError("coolant_ua_w_k must be non-negative")
        if checked.duplicated(["episode_id", "time_s"]).any():
            raise ObservationValidationError("Duplicate episode_id/time_s observations")
        for _, episode in checked.groupby("episode_id", sort=False):
            times = episode["time_s"].to_numpy(dtype=float)
            if len(times) > 1 and np.any(np.diff(times) <= 0):
                raise ObservationValidationError("time_s must increase strictly within each episode")
        if "external_heat_w" not in checked:
            checked["external_heat_w"] = 0.0
        elif not np.isfinite(pd.to_numeric(checked["external_heat_w"]).to_numpy(dtype=float)).all():
            raise ObservationValidationError("external_heat_w must be finite")
        checked["episode_id"] = checked["episode_id"].astype(int)
        return cls(checked.reset_index(drop=True), dataset_ids[0], maturities[0])

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in self.frame["episode_id"].unique()))

    def subset(self, episode_ids: list[int] | tuple[int, ...]) -> "ObservationDataset":
        selected = self.frame[self.frame["episode_id"].isin(episode_ids)].copy()
        if selected.empty:
            raise ObservationValidationError("Episode subset is empty")
        return ObservationDataset.from_frame(selected)

    def to_csv(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(output, index=False)
