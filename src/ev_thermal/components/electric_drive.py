"""Motor/inverter loss maps and lumped thermal dynamics."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElectricDriveState:
    motor_temp_c: float
    inverter_temp_c: float


@dataclass(frozen=True)
class ElectricDriveLosses:
    efficiency: float
    motor_loss_w: float
    inverter_loss_w: float
    motor_coolant_heat_w: float
    inverter_coolant_heat_w: float

    @property
    def total_loss_w(self) -> float:
        return self.motor_loss_w + self.inverter_loss_w

    @property
    def coolant_heat_w(self) -> float:
        """Positive heat leaves devices and enters the powertrain coolant."""
        return self.motor_coolant_heat_w + self.inverter_coolant_heat_w


@dataclass(frozen=True)
class ElectricDriveStep:
    state: ElectricDriveState
    dc_power_w: float
    losses: ElectricDriveLosses


class ElectricDriveModel:
    """System-level efficiency surface with one thermal state per device."""

    motor_heat_capacity_j_k = 85_000.0
    inverter_heat_capacity_j_k = 28_000.0

    def efficiency(self, mechanical_power_w: float, speed_rpm: float) -> float:
        load = np.clip(abs(mechanical_power_w) / 180_000.0, 0.0, 1.0)
        speed = np.clip(abs(speed_rpm) / 14_000.0, 0.0, 1.0)
        # Peak efficiency lies at medium-high load and speed. This analytic map
        # stands in for a calibrated 2-D table while preserving its shape.
        eta = 0.965 - 0.07 * (load - 0.65) ** 2 - 0.05 * (speed - 0.55) ** 2 - 0.04 * np.exp(-8 * load)
        return float(np.clip(eta, 0.78, 0.97))

    def step(self, state: ElectricDriveState, mechanical_power_w: float, speed_rpm: float,
             coolant_temp_c: float, coolant_ua_w_per_k: float, dt_s: float) -> ElectricDriveStep:
        eta = self.efficiency(mechanical_power_w, speed_rpm)
        if mechanical_power_w >= 0:
            dc_power = mechanical_power_w / eta
            total_loss = dc_power - mechanical_power_w
        else:
            dc_power = mechanical_power_w * eta
            total_loss = abs(mechanical_power_w - dc_power)
        # Motor carries copper/iron loss; inverter carries semiconductor loss.
        motor_loss = 0.72 * total_loss
        inverter_loss = 0.28 * total_loss
        ua = max(coolant_ua_w_per_k, 0.0)
        motor_cooling = 0.72 * ua * (state.motor_temp_c - coolant_temp_c)
        inverter_cooling = 0.28 * ua * (state.inverter_temp_c - coolant_temp_c)
        motor_temp = state.motor_temp_c + dt_s * (motor_loss - motor_cooling) / self.motor_heat_capacity_j_k
        inverter_temp = state.inverter_temp_c + dt_s * (inverter_loss - inverter_cooling) / self.inverter_heat_capacity_j_k
        return ElectricDriveStep(
            ElectricDriveState(float(motor_temp), float(inverter_temp)),
            float(dc_power),
            ElectricDriveLosses(eta, float(motor_loss), float(inverter_loss),
                                float(motor_cooling), float(inverter_cooling)),
        )
