"""Single-source parameter metadata shared by simulation and calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import yaml

from ..components.battery import BatteryParameters


CONFIDENCE_LEVELS = {"low", "medium", "high"}
MATURITY_LEVELS = {"synthetic_recovery", "measured_calibration"}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    default_value: float
    lower_bound: float
    upper_bound: float
    source: str
    confidence: str
    model: str
    calibratable: bool = True
    config_path: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.source or not self.model:
            raise ValueError("Parameter name, unit, source, and model are required")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"Unsupported confidence level: {self.confidence}")
        if not self.lower_bound < self.upper_bound:
            raise ValueError(f"Invalid bounds for {self.name}")
        if not self.lower_bound <= self.default_value <= self.upper_bound:
            raise ValueError(f"Default value is outside bounds for {self.name}")


class ParameterRegistry:
    def __init__(self, specs: list[ParameterSpec]):
        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate parameter names: {duplicates}")
        self._specs = tuple(specs)
        self._by_name = {spec.name: spec for spec in specs}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ParameterRegistry":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        entries = raw.get("parameters")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Parameter registry requires a non-empty 'parameters' list")
        specs = []
        for entry in entries:
            try:
                specs.append(ParameterSpec(
                    name=str(entry["name"]),
                    unit=str(entry["unit"]),
                    default_value=float(entry["default"]),
                    lower_bound=float(entry["lower"]),
                    upper_bound=float(entry["upper"]),
                    source=str(entry["source"]),
                    confidence=str(entry["confidence"]),
                    model=str(entry["model"]),
                    calibratable=bool(entry.get("calibratable", True)),
                    config_path=entry.get("config_path"),
                ))
            except KeyError as exc:
                raise ValueError(f"Missing registry field: {exc.args[0]}") from exc
        return cls(specs)

    @property
    def specs(self) -> tuple[ParameterSpec, ...]:
        return self._specs

    def __getitem__(self, name: str) -> ParameterSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown registered parameter: {name}") from exc

    def select(self, model: str | None = None,
               calibratable: bool | None = None) -> list[ParameterSpec]:
        return [
            spec for spec in self._specs
            if (model is None or spec.model == model)
            and (calibratable is None or spec.calibratable == calibratable)
        ]

    def defaults(self, names: list[str] | tuple[str, ...]) -> dict[str, float]:
        return {name: self[name].default_value for name in names}

    def validate_values(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            spec = self[name]
            numeric = float(value)
            if not spec.lower_bound <= numeric <= spec.upper_bound:
                raise ValueError(
                    f"{name}={numeric} is outside [{spec.lower_bound}, {spec.upper_bound}] {spec.unit}"
                )

    def validate_project_defaults(self, config: Any) -> None:
        for spec in self._specs:
            if not spec.config_path:
                continue
            value = config
            for part in spec.config_path.split("."):
                if not hasattr(value, part):
                    raise ValueError(f"Config path does not exist for {spec.name}: {spec.config_path}")
                value = getattr(value, part)
            if abs(float(value) - spec.default_value) > 1e-9 * max(abs(spec.default_value), 1.0):
                raise ValueError(
                    f"Registry default for {spec.name} ({spec.default_value}) "
                    f"does not match config ({value})"
                )


@dataclass(frozen=True)
class CalibratedParameterSet:
    dataset_id: str
    maturity: str
    values: dict[str, float]
    units: dict[str, str]
    method: str

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.method:
            raise ValueError("dataset_id and method are required")
        if self.maturity not in MATURITY_LEVELS:
            raise ValueError(f"Unsupported calibration maturity: {self.maturity}")
        if set(self.values) != set(self.units):
            raise ValueError("Every calibrated value must have exactly one unit")

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibratedParameterSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset_id=payload["dataset_id"],
            maturity=payload["maturity"],
            values={name: float(value) for name, value in payload["values"].items()},
            units=dict(payload["units"]),
            method=payload["method"],
        )


def battery_parameters_from_config(config: Any,
                                   calibrated_values: dict[str, float] | None = None) -> BatteryParameters:
    """Adapt project config plus optional registered overrides to the battery model."""
    battery = config.battery
    parameters = BatteryParameters(
        capacity_kwh=battery.capacity_kwh,
        nominal_voltage_v=battery.nominal_voltage_v,
        nominal_resistance_ohm=battery.nominal_resistance_ohm,
        core_heat_capacity_j_k=battery.core_heat_capacity_j_k,
        surface_heat_capacity_j_k=battery.surface_heat_capacity_j_k,
        core_surface_conductance_w_k=battery.core_surface_conductance_w_k,
    )
    values = calibrated_values or {}
    mapping = {
        "battery.core_heat_capacity_j_k": "core_heat_capacity_j_k",
        "battery.surface_heat_capacity_j_k": "surface_heat_capacity_j_k",
        "battery.core_surface_conductance_w_k": "core_surface_conductance_w_k",
        "battery.nominal_resistance_ohm": "nominal_resistance_ohm",
    }
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported battery parameter overrides: {unknown}")
    replacements = {mapping[name]: float(value) for name, value in values.items()}
    return replace(parameters, **replacements)
