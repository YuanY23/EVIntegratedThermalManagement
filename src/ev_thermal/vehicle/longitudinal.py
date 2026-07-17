"""One-dimensional vehicle force and power balance."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VehicleParameters:
    mass_kg: float = 1950.0
    rolling_resistance: float = 0.0105
    drag_coefficient: float = 0.27
    frontal_area_m2: float = 2.35
    air_density_kg_m3: float = 1.225
    wheel_radius_m: float = 0.34
    final_drive_ratio: float = 9.1
    drivetrain_efficiency: float = 0.97
    max_traction_power_w: float = 180_000.0
    max_regen_power_w: float = 80_000.0


@dataclass(frozen=True)
class VehicleOperatingPoint:
    traction_force_n: float
    wheel_power_w: float
    battery_mechanical_power_w: float
    motor_speed_rpm: float
    motor_torque_nm: float


class LongitudinalVehicle:
    """Convert speed/acceleration/grade into the electric-drive operating point."""

    def __init__(self, params: VehicleParameters | None = None):
        self.params = params or VehicleParameters()

    def step(self, speed_mps: float, acceleration_mps2: float, grade_rad: float) -> VehicleOperatingPoint:
        p = self.params
        v = max(float(speed_mps), 0.0)
        # Positive force acts in the driving direction. Rolling resistance is kept
        # at zero at standstill to avoid fictitious energy use while the car waits.
        rolling = p.mass_kg * 9.81 * p.rolling_resistance * math.cos(grade_rad) if v > 0.05 else 0.0
        aerodynamic = 0.5 * p.air_density_kg_m3 * p.drag_coefficient * p.frontal_area_m2 * v**2
        grade = p.mass_kg * 9.81 * math.sin(grade_rad)
        inertia = p.mass_kg * acceleration_mps2
        force = rolling + aerodynamic + grade + inertia
        wheel_power = force * v
        if wheel_power >= 0:
            # Mechanical power before motor losses; traction clipping represents
            # the vehicle's continuous power capability rather than a controller.
            dc_equivalent = min(wheel_power / p.drivetrain_efficiency, p.max_traction_power_w)
        else:
            # Negative means regenerative mechanical power available to the DC bus.
            dc_equivalent = max(wheel_power * p.drivetrain_efficiency, -p.max_regen_power_w)
        wheel_omega = v / max(p.wheel_radius_m, 1e-6)
        motor_omega = wheel_omega * p.final_drive_ratio
        rpm = motor_omega * 60.0 / (2.0 * math.pi)
        torque = wheel_power / max(abs(motor_omega), 1.0)
        return VehicleOperatingPoint(force, wheel_power, dc_equivalent, rpm, torque)

