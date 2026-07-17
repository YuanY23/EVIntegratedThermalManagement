"""Bounded PID controller with conditional integration anti-windup."""

from dataclasses import dataclass


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    output_min: float = 0.0
    output_max: float = 1.0
    integral_limit: float = 10.0

    def __post_init__(self):
        self.integral = 0.0
        self.previous_error = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, setpoint: float, measurement: float, dt_s: float, reverse: bool = False) -> float:
        """Update output; reverse=True makes output rise above a cooling setpoint."""
        raw_error = measurement - setpoint if reverse else setpoint - measurement
        derivative = 0.0 if self.previous_error is None else (raw_error - self.previous_error) / max(dt_s, 1e-9)
        candidate_integral = max(-self.integral_limit, min(self.integral_limit,
                                                          self.integral + raw_error * dt_s))
        unconstrained = self.kp * raw_error + self.ki * candidate_integral + self.kd * derivative
        output = max(self.output_min, min(self.output_max, unconstrained))
        # Do not integrate further if the error would push an already saturated
        # actuator farther into saturation. This prevents delayed recovery.
        drives_high = output >= self.output_max and raw_error > 0
        drives_low = output <= self.output_min and raw_error < 0
        if not (drives_high or drives_low):
            self.integral = candidate_integral
        self.previous_error = raw_error
        return output

