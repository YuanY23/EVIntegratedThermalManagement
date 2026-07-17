"""Quasi-steady reversible heat-pump and electric PTC models."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeatPumpResult:
    thermal_output_w: float
    electrical_power_w: float
    cop: float
    capacity_limited: bool


class HeatPumpModel:
    """System-level map, intentionally excluding refrigerant transient states.

    It captures source/sink temperature lift, part load, frosting degradation, and
    finite capacity. A two-phase refrigerant model would require compressor and
    heat-exchanger calibration unavailable in a vehicle-level conceptual study.
    """

    def __init__(self, max_heating_w: float = 7000.0, max_cooling_w: float = 6500.0,
                 carnot_efficiency: float = 0.42):
        self.max_heating_w = max_heating_w
        self.max_cooling_w = max_cooling_w
        self.carnot_efficiency = carnot_efficiency

    def operate(self, thermal_request_w: float, ambient_temp_c: float, sink_temp_c: float) -> HeatPumpResult:
        if abs(thermal_request_w) < 1e-9:
            return HeatPumpResult(0, 0, 1, False)
        heating = thermal_request_w > 0
        if heating:
            hot_k = max(sink_temp_c + 273.15 + 5.0, 275.0)
            cold_k = min(ambient_temp_c + 273.15 - 3.0, hot_k - 3.0)
            carnot = hot_k / max(hot_k - cold_k, 3.0)
            frost = 0.68 if ambient_temp_c < -10 else (0.82 if ambient_temp_c < 2 else 1.0)
            cop = max(1.0, min(4.5, self.carnot_efficiency * carnot * frost))
            capacity = self.max_heating_w * max(0.45, min(1.0, (ambient_temp_c + 30.0) / 40.0))
        else:
            cold_k = max(sink_temp_c + 273.15 - 7.0, 268.0)
            hot_k = max(ambient_temp_c + 273.15 + 8.0, cold_k + 3.0)
            carnot = cold_k / max(hot_k - cold_k, 3.0)
            cop = max(1.2, min(4.2, self.carnot_efficiency * carnot))
            capacity = self.max_cooling_w
        delivered = math.copysign(min(abs(thermal_request_w), capacity), thermal_request_w)
        return HeatPumpResult(delivered, abs(delivered) / cop, cop, abs(thermal_request_w) > capacity)


def ptc_heating(electrical_power_w: float, efficiency: float = 0.97) -> float:
    """PTC thermal output; all unconverted electricity remains local heat."""
    return max(electrical_power_w, 0.0) * min(max(efficiency, 0.0), 1.0)

