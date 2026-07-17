import numpy as np

from ev_thermal.config import load_config
from ev_thermal.control.pid import PID
from ev_thermal.control.actuator import FirstOrderActuator
from ev_thermal.control.predictive import ForecastSummary, PredictiveSupervisor
from ev_thermal.control.state_machine import SupervisorInputs, ThermalSupervisor
from ev_thermal.simulation.integrated import IntegratedSimulator
from ev_thermal.simulation.scenarios import make_scenario


def test_pid_and_supervisor_outputs_are_bounded():
    pid = PID(kp=0.2, ki=0.1, kd=0.0, output_min=0, output_max=1)
    for _ in range(100):
        output = pid.update(setpoint=25, measurement=50, dt_s=1)
    assert 0 <= output <= 1
    assert abs(pid.integral) <= pid.integral_limit

    supervisor = ThermalSupervisor()
    action = supervisor.command(SupervisorInputs(
        battery_temp_c=42, motor_temp_c=85, inverter_temp_c=75,
        cabin_temp_c=30, cabin_setpoint_c=24, ambient_temp_c=40,
        powertrain_waste_heat_w=5000, dt_s=1,
    ))
    assert action.mode in {"cooling", "high_cooling", "cabin_cooling"}
    assert all(0 <= value <= 1 for value in action.normalized_actuators())
    assert action.battery_chiller > 0


def test_actuator_first_order_response_is_bounded_and_monotonic():
    actuator = FirstOrderActuator(time_constant_s=10.0)
    values = [actuator.update(1.0, dt_s=2.0) for _ in range(8)]
    assert 0 < values[0] < values[-1] < 1
    assert np.all(np.diff(values) > 0)
    falling = actuator.update(0.0, dt_s=2.0)
    assert 0 <= falling < values[-1]


def test_predictive_layer_adjusts_setpoints_and_falls_back():
    baseline = ThermalSupervisor()
    predictive = PredictiveSupervisor(baseline)
    hot = ForecastSummary(valid=True, battery_heat_peak_w=9000, powertrain_heat_mean_w=3000, cabin_load_mean_w=2000)
    adjusted = predictive.adjust_thresholds(hot)
    assert adjusted.battery_cooling_on_c <= 29.0
    assert adjusted.battery_high_cooling_c < baseline.thresholds.battery_high_cooling_c
    invalid = predictive.adjust_thresholds(ForecastSummary.invalid("missing"))
    assert invalid == baseline.thresholds


def test_predictive_fallback_uses_immutable_original_thresholds():
    baseline = ThermalSupervisor()
    original = baseline.thresholds
    predictive = PredictiveSupervisor(baseline)
    baseline.thresholds = predictive.adjust_thresholds(ForecastSummary(True, 8000, 2000, 1000))
    assert predictive.adjust_thresholds(ForecastSummary.invalid("fault")) == original


def test_integrated_simulation_is_finite_and_balanced():
    cfg = load_config()
    scenario = make_scenario("urban_hot", duration_s=300, dt_s=cfg.simulation.dt_s, seed=7)
    result = IntegratedSimulator(cfg).run(scenario, strategy="baseline")
    numeric = result.timeseries.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert result.timeseries["soc"].between(0, 1).all()
    assert result.metrics["energy_balance_error_pct"] < 2.0
    assert result.metrics["thermal_balance_error_pct"] < 2.0
    assert result.metrics["distance_km"] > 0


def test_cold_battery_warmup_can_heat_cabin_concurrently():
    cfg = load_config()
    scenario = make_scenario("cold_start", duration_s=300, dt_s=cfg.simulation.dt_s, seed=3)
    result = IntegratedSimulator(cfg).run(scenario, strategy="baseline")
    assert result.timeseries["battery_heater_power_w"].max() > 0
    assert result.timeseries["compressor_power_w"].max() > 0
    assert result.timeseries["cabin_ptc_power_w"].max() > 0
    assert result.timeseries["cabin_temp_c"].iloc[-1] > scenario.initial_cabin_temp_c


def test_invalid_forecast_reproduces_baseline_physical_trajectory():
    class InvalidPredictor:
        def summary(self, rows):
            return ForecastSummary.invalid("sensor_fault")

    cfg = load_config()
    scenario = make_scenario("aggressive", duration_s=300, dt_s=cfg.simulation.dt_s, seed=9)
    simulator = IntegratedSimulator(cfg)
    baseline = simulator.run(scenario, "baseline").timeseries
    fallback = simulator.run(scenario, "predictive", InvalidPredictor()).timeseries
    columns = ["soc", "battery_core_temp_c", "motor_temp_c", "cabin_temp_c", "auxiliary_power_w"]
    assert np.allclose(baseline[columns], fallback[columns])
    assert set(fallback["forecast_reason"]) == {"sensor_fault"}


def test_high_heat_forecast_preconditioning_does_not_raise_peak_battery_temperature():
    class HotForecast:
        def summary(self, rows):
            return ForecastSummary(True, 8000, 2500, 1000)

    cfg = load_config()
    scenario = make_scenario("aggressive", duration_s=600, dt_s=cfg.simulation.dt_s, seed=12)
    simulator = IntegratedSimulator(cfg)
    baseline = simulator.run(scenario, "baseline")
    predictive = simulator.run(scenario, "predictive", HotForecast())
    assert predictive.metrics["max_battery_temp_c"] <= baseline.metrics["max_battery_temp_c"]
    assert predictive.timeseries["battery_chiller_heat_w"].max() > 0
