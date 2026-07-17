"""Rule supervisor for integrated battery, powertrain, and cabin thermal paths."""

from dataclasses import dataclass, replace

from .pid import PID


@dataclass(frozen=True)
class SupervisorThresholds:
    battery_cooling_on_c: float = 34.0
    battery_high_cooling_c: float = 39.0
    battery_heating_on_c: float = 12.0
    motor_cooling_on_c: float = 70.0
    inverter_cooling_on_c: float = 65.0
    cabin_deadband_c: float = 0.8
    waste_heat_min_w: float = 1200.0


@dataclass(frozen=True)
class SupervisorInputs:
    battery_temp_c: float
    motor_temp_c: float
    inverter_temp_c: float
    cabin_temp_c: float
    cabin_setpoint_c: float
    ambient_temp_c: float
    powertrain_waste_heat_w: float
    dt_s: float


@dataclass(frozen=True)
class ThermalAction:
    mode: str
    battery_pump: float = 0.0
    powertrain_pump: float = 0.0
    fan: float = 0.0
    compressor: float = 0.0
    ptc: float = 0.0
    waste_heat_valve: float = 0.0
    cabin_thermal_request_w: float = 0.0
    battery_heater_w: float = 0.0
    battery_chiller: float = 0.0
    cabin_ptc: float = 0.0
    reason: str = "baseline"

    def normalized_actuators(self) -> tuple[float, ...]:
        return (self.battery_pump, self.powertrain_pump, self.fan,
                self.compressor, self.ptc, self.waste_heat_valve,
                self.battery_chiller, self.cabin_ptc)


class ThermalSupervisor:
    """Topology/state selection above independent local feedback loops."""

    def __init__(self, thresholds: SupervisorThresholds | None = None):
        self.thresholds = thresholds or SupervisorThresholds()
        self.battery_cooling_pid = PID(0.12, 0.004, 0.0, integral_limit=40)
        self.powertrain_pid = PID(0.07, 0.002, 0.0, integral_limit=80)
        self.cabin_pid = PID(0.20, 0.003, 0.0, output_min=-1, output_max=1, integral_limit=120)
        self._cooling_latched = False

    def clone_with_thresholds(self, thresholds: SupervisorThresholds) -> "ThermalSupervisor":
        return ThermalSupervisor(thresholds)

    def reset(self) -> None:
        self.battery_cooling_pid.reset()
        self.powertrain_pid.reset()
        self.cabin_pid.reset()
        self._cooling_latched = False

    def command(self, inputs: SupervisorInputs) -> ThermalAction:
        t = self.thresholds
        # Hysteresis prevents rapid mode toggling around battery cooling onset.
        if inputs.battery_temp_c >= t.battery_cooling_on_c:
            self._cooling_latched = True
        elif inputs.battery_temp_c <= t.battery_cooling_on_c - 2.0:
            self._cooling_latched = False

        cabin_error = inputs.cabin_setpoint_c - inputs.cabin_temp_c
        cabin_fraction = self.cabin_pid.update(inputs.cabin_setpoint_c, inputs.cabin_temp_c, inputs.dt_s)
        cabin_request = 0.0
        if abs(cabin_error) > t.cabin_deadband_c:
            cabin_request = 6500.0 * max(-1.0, min(1.0, cabin_fraction))

        if inputs.battery_temp_c < t.battery_heating_on_c:
            heater_fraction = min((t.battery_heating_on_c - inputs.battery_temp_c) / 10.0 + 0.25, 1.0)
            # At severe cold, heat-pump capacity is insufficient for comfort and
            # demisting. A separate cabin PTC supplements the battery heater.
            cold_severity = max(0.0, min(1.0, (5.0 - inputs.ambient_temp_c) / 25.0))
            cabin_ptc = cold_severity * max(0.0, min(cabin_request / 6500.0, 1.0))
            return ThermalAction("battery_warmup", battery_pump=0.25, ptc=heater_fraction,
                                 compressor=max(0.0, min(cabin_request / 6500.0, 1.0)),
                                 battery_heater_w=4500 * heater_fraction,
                                 cabin_ptc=cabin_ptc,
                                 cabin_thermal_request_w=max(cabin_request, 0.0))

        battery_cooling = self.battery_cooling_pid.update(t.battery_cooling_on_c,
                                                          inputs.battery_temp_c,
                                                          inputs.dt_s, reverse=True)
        powertrain_temp = max(inputs.motor_temp_c - t.motor_cooling_on_c,
                              inputs.inverter_temp_c - t.inverter_cooling_on_c)
        powertrain_cooling = self.powertrain_pid.update(0.0, powertrain_temp, inputs.dt_s, reverse=True)

        if inputs.battery_temp_c >= t.battery_high_cooling_c:
            cabin_compressor = min(abs(cabin_request) / 6500.0, 1.0)
            return ThermalAction("high_cooling", 1.0, max(0.7, powertrain_cooling), 1.0,
                                 compressor=cabin_compressor,
                                 cabin_thermal_request_w=cabin_request,
                                 battery_chiller=1.0)
        if self._cooling_latched or powertrain_temp > 0:
            pump = max(0.3, battery_cooling)
            fan = max(0.15, 0.75 * max(battery_cooling, powertrain_cooling))
            use_waste = cabin_request > 0 and inputs.powertrain_waste_heat_w > t.waste_heat_min_w
            recovered_request = min(inputs.powertrain_waste_heat_w, cabin_request) if use_waste else 0.0
            cabin_compressor = min(abs(cabin_request - recovered_request) / 6500.0, 1.0)
            # Refrigerant chiller is useful when ambient air cannot reject pack
            # heat. At mild ambient the liquid radiator remains the first choice.
            ambient_chiller_need = max(0.0, min(1.0, (inputs.ambient_temp_c - 22.0) / 12.0))
            return ThermalAction("cooling", pump, max(0.3, powertrain_cooling), fan,
                                 compressor=cabin_compressor,
                                 waste_heat_valve=1.0 if use_waste else 0.0,
                                 cabin_thermal_request_w=cabin_request,
                                 battery_chiller=battery_cooling * ambient_chiller_need)
        if cabin_error > t.cabin_deadband_c:
            use_waste = inputs.powertrain_waste_heat_w > t.waste_heat_min_w
            return ThermalAction("waste_heat_recovery" if use_waste else "cabin_heating",
                                 battery_pump=0.1, powertrain_pump=0.35 if use_waste else 0.15,
                                 compressor=0.2 if use_waste else min(cabin_request / 6500, 1),
                                 ptc=0.0 if use_waste else max(0.0, min(cabin_request / 6500 - 0.5, 0.5)),
                                 waste_heat_valve=1.0 if use_waste else 0.0,
                                 cabin_thermal_request_w=cabin_request)
        if cabin_error < -t.cabin_deadband_c:
            return ThermalAction("cabin_cooling", 0.1, 0.15, 0.2,
                                 compressor=min(abs(cabin_request) / 6500, 1.0),
                                 cabin_thermal_request_w=cabin_request)
        return ThermalAction("standby", battery_pump=0.08, powertrain_pump=0.08)
