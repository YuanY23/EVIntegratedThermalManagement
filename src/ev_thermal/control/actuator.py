"""Simple physical response dynamics for thermal-management actuators."""

from dataclasses import dataclass
import math


@dataclass
class FirstOrderActuator:
    """Bounded first-order lag using an exact discrete-time update.

    ``dx/dt=(u-x)/tau`` represents motor, fluid, valve, and compressor inertia at
    system level. The exponential update is stable for any positive time step.
    """

    time_constant_s: float
    value: float = 0.0

    def update(self, command: float, dt_s: float) -> float:
        target = max(0.0, min(1.0, float(command)))
        if self.time_constant_s <= 0:
            self.value = target
        else:
            alpha = 1.0 - math.exp(-max(dt_s, 0.0) / self.time_constant_s)
            self.value += alpha * (target - self.value)
        self.value = max(0.0, min(1.0, self.value))
        return self.value

    def reset(self, value: float = 0.0) -> None:
        self.value = max(0.0, min(1.0, value))

