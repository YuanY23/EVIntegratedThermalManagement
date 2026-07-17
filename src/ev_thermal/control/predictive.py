"""Prediction-enhanced rule scheduling; forecasts never command actuators."""

from dataclasses import dataclass, replace

from .state_machine import SupervisorThresholds, ThermalSupervisor


@dataclass(frozen=True)
class ForecastSummary:
    valid: bool
    battery_heat_peak_w: float = 0.0
    powertrain_heat_mean_w: float = 0.0
    cabin_load_mean_w: float = 0.0
    reason: str = "ok"

    @classmethod
    def invalid(cls, reason: str) -> "ForecastSummary":
        return cls(False, reason=reason)


class PredictiveSupervisor:
    """Adjust supervisor thresholds from forecast summaries within safe bounds."""

    def __init__(self, baseline: ThermalSupervisor):
        self.baseline = baseline
        # Thresholds are immutable dataclasses; retaining the original value
        # prevents a scheduled threshold from becoming tomorrow's new baseline.
        self.base_thresholds = baseline.thresholds

    def adjust_thresholds(self, forecast: ForecastSummary) -> SupervisorThresholds:
        base = self.base_thresholds
        if not forecast.valid:
            return base
        # Forecast heat affects only scheduling thresholds. Local PID and all
        # actuator interlocks remain exactly those of the interpretable baseline.
        # Even a 2-3 kW future pack heat peak is material over five minutes. A
        # bounded 6 degC scheduling shift can start circulation early while the
        # physical temperature feedback still determines actual PID intensity.
        heat_severity = max(0.0, min(1.0, (forecast.battery_heat_peak_w - 500.0) / 3500.0))
        cooling_on = max(28.0, base.battery_cooling_on_c - 6.0 * heat_severity)
        high_cooling = max(36.0, base.battery_high_cooling_c - 2.0 * heat_severity)
        waste_min = max(700.0, base.waste_heat_min_w - 400.0
                        if forecast.cabin_load_mean_w > 1000 else base.waste_heat_min_w)
        return replace(base, battery_cooling_on_c=cooling_on,
                       battery_high_cooling_c=high_cooling, waste_heat_min_w=waste_min)
