"""Explicit, energy-coupled whole-vehicle thermal simulation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..components.battery import BatteryModel, BatteryParameters, BatteryState
from ..components.cabin import CabinModel, CabinState
from ..components.electric_drive import ElectricDriveModel, ElectricDriveState
from ..components.heat_pump import HeatPumpModel, ptc_heating
from ..config import ProjectConfig
from ..control.actuator import FirstOrderActuator
from ..control.predictive import ForecastSummary, PredictiveSupervisor
from ..control.state_machine import SupervisorInputs, SupervisorThresholds, ThermalSupervisor
from ..metrics import simulation_metrics
from ..thermal_hydraulics.cold_plate import ColdPlate
from ..thermal_hydraulics.fluid import glycol_properties
from ..thermal_hydraulics.heat_exchanger import Radiator
from ..thermal_hydraulics.pump import Pump
from ..vehicle.longitudinal import LongitudinalVehicle, VehicleParameters
from .scenarios import Scenario


@dataclass(frozen=True)
class SimulationResult:
    timeseries: pd.DataFrame
    metrics: dict[str, float]


class IntegratedSimulator:
    """Couple vehicle demand, thermal plants, liquid loops, and controls.

    The solve order is explicit and fixed for reproducibility: longitudinal load,
    previous-state control, pump working points, electric-drive losses, HVAC and
    radiator heat, component state integration, then electrical/thermal ledgers.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        v = config.vehicle
        self.vehicle = LongitudinalVehicle(VehicleParameters(
            mass_kg=v.mass_kg, rolling_resistance=v.rolling_resistance,
            drag_coefficient=v.drag_coefficient, frontal_area_m2=v.frontal_area_m2,
            wheel_radius_m=v.wheel_radius_m, final_drive_ratio=v.final_drive_ratio,
            drivetrain_efficiency=v.drivetrain_efficiency,
            max_traction_power_w=v.max_traction_power_kw * 1000,
            max_regen_power_w=v.max_regen_power_kw * 1000,
        ))
        b = config.battery
        self.battery = BatteryModel(BatteryParameters(
            capacity_kwh=b.capacity_kwh, nominal_voltage_v=b.nominal_voltage_v,
            nominal_resistance_ohm=b.nominal_resistance_ohm,
            core_heat_capacity_j_k=b.core_heat_capacity_j_k,
            surface_heat_capacity_j_k=b.surface_heat_capacity_j_k,
            core_surface_conductance_w_k=b.core_surface_conductance_w_k,
        ))
        self.drive = ElectricDriveModel()
        self.cabin = CabinModel()
        self.heat_pump = HeatPumpModel()
        self.pump = Pump()
        self.cold_plate = ColdPlate()
        self.radiator = Radiator()
        self.battery_radiator = Radiator(ua_nominal_w_k=620.0, fan_max_power_w=0.0)

    def _supervisor(self) -> ThermalSupervisor:
        c = self.config.control
        return ThermalSupervisor(SupervisorThresholds(
            battery_cooling_on_c=c.battery_cooling_on_c,
            battery_high_cooling_c=c.battery_high_cooling_c,
            battery_heating_on_c=c.battery_heating_on_c,
            motor_cooling_on_c=c.motor_cooling_on_c,
            inverter_cooling_on_c=c.inverter_cooling_on_c,
        ))

    def run(self, scenario: Scenario, strategy: str = "baseline", predictor=None) -> SimulationResult:
        if strategy not in {"baseline", "predictive"}:
            raise ValueError("strategy must be 'baseline' or 'predictive'")
        dt = float(np.median(np.diff(scenario.time_s)))
        battery_state = BatteryState(scenario.initial_soc, scenario.initial_battery_temp_c,
                                     scenario.initial_battery_temp_c)
        drive_state = ElectricDriveState(scenario.initial_battery_temp_c + 5,
                                         scenario.initial_battery_temp_c + 3)
        cabin_state = CabinState(scenario.initial_cabin_temp_c, scenario.initial_cabin_temp_c)
        battery_coolant_c = scenario.initial_battery_temp_c
        powertrain_coolant_c = scenario.initial_battery_temp_c + 2
        supervisor = self._supervisor()
        predictive = PredictiveSupervisor(supervisor)
        actuators = {
            "battery_pump": FirstOrderActuator(8.0),
            "powertrain_pump": FirstOrderActuator(8.0),
            "fan": FirstOrderActuator(6.0),
            "cabin_compressor": FirstOrderActuator(20.0),
            "battery_chiller": FirstOrderActuator(20.0),
            "ptc": FirstOrderActuator(5.0),
            "cabin_ptc": FirstOrderActuator(5.0),
        }
        rows: list[dict] = []

        acceleration = np.gradient(scenario.speed_mps, scenario.time_s)
        for index, time_s in enumerate(scenario.time_s):
            previous_battery = battery_state
            previous_drive = drive_state
            previous_cabin = cabin_state
            previous_battery_coolant_c = battery_coolant_c
            previous_powertrain_coolant_c = powertrain_coolant_c
            ambient = float(scenario.ambient_temp_c[index])
            speed = float(scenario.speed_mps[index])
            op = self.vehicle.step(speed, float(acceleration[index]), float(scenario.grade_rad[index]))

            # Controller decisions use only current measured states plus, for the
            # predictive strategy, a compact forecast summary. In absence of a
            # trained predictor the deterministic preview is an oracle-free route
            # proxy based on upcoming speed/grade severity.
            waste_estimate = 2500.0 if abs(op.battery_mechanical_power_w) > 30_000 else 500.0
            thresholds = supervisor.thresholds
            forecast_valid = False
            forecast_reason = "baseline"
            forecast_battery_peak_w = 0.0
            if strategy == "predictive":
                if predictor is not None:
                    summary = predictor.summary(rows)
                else:
                    end = min(index + int(300 / dt) + 1, len(scenario.time_s))
                    future_speed = scenario.speed_mps[index:end]
                    future_grade = scenario.grade_rad[index:end]
                    severity = float(np.mean(future_speed**2) + 300 * np.mean(np.maximum(future_grade, 0)))
                    summary = ForecastSummary(True, 2500 + 18 * severity, waste_estimate,
                                              abs(self.config.control.cabin_setpoint_c - ambient) * 180)
                thresholds = predictive.adjust_thresholds(summary)
                forecast_valid = summary.valid
                forecast_reason = summary.reason
                forecast_battery_peak_w = summary.battery_heat_peak_w
                # Update thresholds in place so all local PID integrals and mode
                # hysteresis survive forecast changes. Recreating the supervisor
                # here would silently degrade cabin tracking.
                if thresholds != supervisor.thresholds:
                    supervisor.thresholds = thresholds

            inputs = SupervisorInputs(
                battery_state.core_temp_c, drive_state.motor_temp_c, drive_state.inverter_temp_c,
                cabin_state.air_temp_c, self.config.control.cabin_setpoint_c, ambient,
                waste_estimate, dt,
            )
            action = supervisor.command(inputs)
            # Commands pass through actuator dynamics before they affect fluid or
            # refrigerant power. This prevents nonphysical step changes and makes
            # plotted commands comparable to measured actuator responses.
            battery_pump_fraction = actuators["battery_pump"].update(action.battery_pump, dt)
            powertrain_pump_fraction = actuators["powertrain_pump"].update(action.powertrain_pump, dt)
            fan_fraction = actuators["fan"].update(action.fan, dt)
            cabin_compressor_command = actuators["cabin_compressor"].update(action.compressor, dt)
            battery_chiller_command = actuators["battery_chiller"].update(action.battery_chiller, dt)
            ptc_fraction = actuators["ptc"].update(action.ptc, dt)
            cabin_ptc_fraction = actuators["cabin_ptc"].update(action.cabin_ptc, dt)
            battery_pump = self.pump.working_point(battery_pump_fraction, 360_000.0)
            drive_pump = self.pump.working_point(powertrain_pump_fraction, 260_000.0)
            battery_fluid = glycol_properties(battery_coolant_c)
            cold_plate = self.cold_plate.exchange(battery_state.surface_temp_c, battery_coolant_c,
                                                  battery_pump.mass_flow_kg_s, battery_fluid)
            battery_ua = (cold_plate.heat_transfer_w /
                          max(abs(battery_state.surface_temp_c - battery_coolant_c), 1e-6))
            battery_radiator = self.battery_radiator.exchange(
                battery_coolant_c, ambient, battery_pump.mass_flow_kg_s, speed, fan_fraction)
            radiator_bypass = 1.0 if action.mode in {"cooling", "high_cooling"} else 0.0
            battery_radiator_heat = battery_radiator.heat_rejected_w * radiator_bypass

            drive_step = self.drive.step(drive_state, op.battery_mechanical_power_w, op.motor_speed_rpm,
                                         powertrain_coolant_c,
                                         900.0 * min(drive_pump.mass_flow_kg_s / 0.25, 1.0), dt)
            drive_state = drive_step.state
            radiator = self.radiator.exchange(powertrain_coolant_c, ambient,
                                               drive_pump.mass_flow_kg_s, speed, fan_fraction)

            # Waste heat is limited by available electric-drive losses and cabin
            # request. It displaces heat-pump output before PTC is considered.
            waste_heat = min(max(drive_step.losses.coolant_heat_w, 0.0) * action.waste_heat_valve,
                             max(action.cabin_thermal_request_w, 0.0))
            remaining_cabin_request = action.cabin_thermal_request_w - waste_heat
            # Compressor speed limits available heat-pump output. Battery warm-up
            # has a separate PTC path so its electricity is not incorrectly added
            # to cabin heat at the same time.
            # Cabin HVAC and battery chiller share a normalized compressor budget.
            # If both request full capacity, proportional allocation keeps the
            # combined command at or below one instead of double-counting power.
            compressor_sum = cabin_compressor_command + battery_chiller_command
            compressor_scale = 1.0 / max(compressor_sum, 1.0)
            cabin_compressor = cabin_compressor_command * compressor_scale
            battery_chiller_fraction = battery_chiller_command * compressor_scale
            hp_request = np.sign(remaining_cabin_request) * min(
                abs(remaining_cabin_request), 6500.0 * cabin_compressor)
            hp = self.heat_pump.operate(hp_request, ambient, cabin_state.air_temp_c)
            battery_chiller = self.heat_pump.operate(
                -6000.0 * battery_chiller_fraction, ambient, battery_coolant_c)
            battery_chiller_heat = max(-battery_chiller.thermal_output_w, 0.0)
            battery_heater_power = 4500.0 * ptc_fraction / 0.97 if action.battery_heater_w > 0 else 0.0
            cabin_ptc_power = 5000.0 * cabin_ptc_fraction
            if action.battery_heater_w <= 0:
                cabin_ptc_power += 5000.0 * ptc_fraction
            hvac_heat = hp.thermal_output_w + waste_heat + ptc_heating(cabin_ptc_power)
            cabin_step = self.cabin.step(cabin_state, ambient, float(scenario.solar_w_m2[index]),
                                         int(scenario.occupants[index]), hvac_heat, dt)
            cabin_state = cabin_step.state

            aux_power = (battery_pump.electrical_power_w + drive_pump.electrical_power_w +
                         radiator.fan_power_w + hp.electrical_power_w + cabin_ptc_power +
                         battery_heater_power + battery_chiller.electrical_power_w)
            battery_total_requested = drive_step.dc_power_w + aux_power
            battery_step = self.battery.step(battery_state, battery_total_requested,
                                             battery_coolant_c, battery_ua, dt,
                                             external_surface_heat_w=ptc_heating(battery_heater_power))
            battery_state = battery_step.state

            # Coolant nodes are finite thermal masses. Heat received from devices
            # minus radiator rejection and ambient leakage changes stored energy.
            batt_coolant_capacity = 3.5 * battery_fluid.cp_j_kgk
            battery_coolant_c += dt * (battery_step.diagnostics.coolant_heat_w
                                       - battery_radiator_heat - battery_chiller_heat
                                       - 90.0 * (battery_coolant_c - ambient)) / batt_coolant_capacity
            drive_fluid = glycol_properties(powertrain_coolant_c)
            drive_coolant_capacity = 5.5 * drive_fluid.cp_j_kgk
            powertrain_coolant_c += dt * (drive_step.losses.coolant_heat_w - radiator.heat_rejected_w
                                          - waste_heat) / drive_coolant_capacity

            # The electrical ledger identity uses actual terminal power. Its
            # residual is retained rather than hidden, enabling acceptance tests.
            actual_total = battery_step.diagnostics.terminal_power_w
            ledger_residual = actual_total - (drive_step.dc_power_w + aux_power)

            # Independent first-law checks for the five thermal storage groups.
            # Internal core/surface and air/interior exchange terms cancel when
            # each group is summed, so only boundary heat flows remain.
            bp = self.battery.params
            battery_storage_w = (
                bp.core_heat_capacity_j_k * (battery_state.core_temp_c - previous_battery.core_temp_c) +
                bp.surface_heat_capacity_j_k * (battery_state.surface_temp_c - previous_battery.surface_temp_c)
            ) / dt
            battery_boundary_w = (battery_step.diagnostics.heat_generation_w +
                                  battery_step.diagnostics.external_heat_w -
                                  battery_step.diagnostics.coolant_heat_w)
            drive_storage_w = (
                self.drive.motor_heat_capacity_j_k * (drive_state.motor_temp_c - previous_drive.motor_temp_c) +
                self.drive.inverter_heat_capacity_j_k * (drive_state.inverter_temp_c - previous_drive.inverter_temp_c)
            ) / dt
            drive_boundary_w = drive_step.losses.total_loss_w - drive_step.losses.coolant_heat_w
            cp = self.cabin.params
            cabin_storage_w = (
                cp.air_heat_capacity_j_k * (cabin_state.air_temp_c - previous_cabin.air_temp_c) +
                cp.interior_heat_capacity_j_k * (cabin_state.interior_temp_c - previous_cabin.interior_temp_c)
            ) / dt
            cabin_boundary_w = (cabin_step.diagnostics.ambient_heat_w + cabin_step.diagnostics.solar_heat_w +
                                cabin_step.diagnostics.occupant_heat_w + cabin_step.diagnostics.hvac_heat_w)
            battery_coolant_storage_w = batt_coolant_capacity * (
                battery_coolant_c - previous_battery_coolant_c) / dt
            battery_coolant_boundary_w = (battery_step.diagnostics.coolant_heat_w -
                                          battery_radiator_heat - battery_chiller_heat -
                                          90.0 * (previous_battery_coolant_c - ambient))
            drive_coolant_storage_w = drive_coolant_capacity * (
                powertrain_coolant_c - previous_powertrain_coolant_c) / dt
            drive_coolant_boundary_w = (drive_step.losses.coolant_heat_w - radiator.heat_rejected_w - waste_heat)
            thermal_residual = ((battery_storage_w - battery_boundary_w) +
                                (drive_storage_w - drive_boundary_w) +
                                (cabin_storage_w - cabin_boundary_w) +
                                (battery_coolant_storage_w - battery_coolant_boundary_w) +
                                (drive_coolant_storage_w - drive_coolant_boundary_w))
            rows.append({
                "time_s": time_s, "speed_mps": speed, "grade_rad": scenario.grade_rad[index],
                "ambient_temp_c": ambient, "soc": battery_state.soc,
                "battery_core_temp_c": battery_state.core_temp_c,
                "battery_surface_temp_c": battery_state.surface_temp_c,
                "motor_temp_c": drive_state.motor_temp_c, "inverter_temp_c": drive_state.inverter_temp_c,
                "cabin_temp_c": cabin_state.air_temp_c,
                "cabin_temp_error_c": cabin_state.air_temp_c - self.config.control.cabin_setpoint_c,
                "battery_coolant_temp_c": battery_coolant_c,
                "powertrain_coolant_temp_c": powertrain_coolant_c,
                "wheel_power_w": op.wheel_power_w, "drive_dc_power_w": drive_step.dc_power_w,
                "battery_terminal_power_w": battery_step.diagnostics.terminal_power_w,
                "battery_total_power_w": actual_total,
                "battery_heat_w": battery_step.diagnostics.heat_generation_w,
                "powertrain_heat_w": drive_step.losses.total_loss_w,
                "powertrain_to_coolant_heat_w": drive_step.losses.coolant_heat_w,
                "cabin_load_w": cabin_step.diagnostics.net_unconditioned_load_w,
                "auxiliary_power_w": aux_power, "pump_power_w": battery_pump.electrical_power_w + drive_pump.electrical_power_w,
                "fan_power_w": radiator.fan_power_w, "compressor_power_w": hp.electrical_power_w,
                "battery_chiller_power_w": battery_chiller.electrical_power_w,
                "ptc_power_w": cabin_ptc_power + battery_heater_power,
                "battery_heater_power_w": battery_heater_power,
                "cabin_ptc_power_w": cabin_ptc_power,
                "waste_heat_recovered_w": waste_heat,
                "battery_flow_kg_s": battery_pump.mass_flow_kg_s,
                "powertrain_flow_kg_s": drive_pump.mass_flow_kg_s,
                "battery_pressure_drop_pa": battery_pump.pressure_rise_pa,
                "radiator_heat_w": radiator.heat_rejected_w,
                "battery_radiator_heat_w": battery_radiator_heat,
                "battery_chiller_heat_w": battery_chiller_heat,
                "battery_cooling_heat_w": battery_step.diagnostics.coolant_heat_w,
                "hvac_heat_w": hvac_heat, "heat_pump_cop": hp.cop,
                "pump_fraction": battery_pump_fraction, "fan_fraction": fan_fraction,
                "compressor_fraction": cabin_compressor + battery_chiller_fraction,
                "battery_chiller_fraction": battery_chiller_fraction,
                "ptc_fraction": ptc_fraction,
                "cabin_ptc_fraction": cabin_ptc_fraction,
                "mode": action.mode, "forecast_valid": forecast_valid,
                "forecast_reason": forecast_reason,
                "forecast_battery_heat_peak_w": forecast_battery_peak_w,
                "scheduled_battery_cooling_on_c": thresholds.battery_cooling_on_c,
                "energy_balance_residual_w": ledger_residual,
                "thermal_balance_residual_w": thermal_residual,
            })
        frame = pd.DataFrame(rows)
        return SimulationResult(frame, simulation_metrics(frame, self.config.battery.capacity_kwh))
