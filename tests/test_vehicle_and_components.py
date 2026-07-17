import numpy as np

from ev_thermal.components.battery import BatteryModel, BatteryParameters, BatteryState
from ev_thermal.components.cabin import CabinModel, CabinParameters, CabinState
from ev_thermal.components.electric_drive import ElectricDriveModel, ElectricDriveState
from ev_thermal.vehicle.longitudinal import LongitudinalVehicle, VehicleParameters


def test_longitudinal_force_and_regen_are_physical():
    model = LongitudinalVehicle(VehicleParameters())
    level = model.step(speed_mps=20.0, acceleration_mps2=0.0, grade_rad=0.0)
    uphill = model.step(speed_mps=20.0, acceleration_mps2=0.0, grade_rad=np.deg2rad(5.0))
    braking = model.step(speed_mps=20.0, acceleration_mps2=-2.0, grade_rad=0.0)
    assert level.traction_force_n > 0
    assert uphill.wheel_power_w > level.wheel_power_w
    assert braking.battery_mechanical_power_w <= 0
    assert abs(braking.battery_mechanical_power_w) <= model.params.max_regen_power_w


def test_battery_heat_soc_and_cooling_directions():
    model = BatteryModel(BatteryParameters())
    state = BatteryState(soc=0.8, core_temp_c=30.0, surface_temp_c=29.0)
    hot = model.step(state, terminal_power_w=70_000.0, coolant_temp_c=35.0, coolant_ua_w_per_k=0.0, dt_s=1.0)
    cooled = model.step(state, terminal_power_w=70_000.0, coolant_temp_c=20.0, coolant_ua_w_per_k=500.0, dt_s=1.0)
    assert hot.diagnostics.heat_generation_w > 0
    assert hot.state.soc < state.soc
    assert cooled.state.surface_temp_c < hot.state.surface_temp_c
    assert model.resistance_ohm(0.5, -10.0) > model.resistance_ohm(0.5, 25.0)


def test_external_battery_heater_enters_surface_energy_balance():
    model = BatteryModel(BatteryParameters())
    state = BatteryState(soc=0.8, core_temp_c=-15.0, surface_temp_c=-15.0)
    passive = model.step(state, 0, -15, 0, 10)
    heated = model.step(state, 0, -15, 0, 10, external_surface_heat_w=4000)
    assert heated.state.surface_temp_c > passive.state.surface_temp_c
    assert heated.diagnostics.external_heat_w == 4000


def test_electric_drive_losses_and_cooling():
    model = ElectricDriveModel()
    state = ElectricDriveState(motor_temp_c=45.0, inverter_temp_c=40.0)
    warm = model.step(state, mechanical_power_w=60_000, speed_rpm=5000, coolant_temp_c=55, coolant_ua_w_per_k=0, dt_s=2)
    cool = model.step(state, mechanical_power_w=60_000, speed_rpm=5000, coolant_temp_c=20, coolant_ua_w_per_k=800, dt_s=2)
    assert warm.losses.total_loss_w > 0
    assert 0.7 < warm.losses.efficiency < 1.0
    assert cool.losses.coolant_heat_w > 0
    assert cool.losses.coolant_heat_w <= cool.losses.total_loss_w + 800 * (45 - 20)
    assert cool.state.motor_temp_c < warm.state.motor_temp_c


def test_cabin_2r2c_load_directions():
    model = CabinModel(CabinParameters())
    initial = CabinState(air_temp_c=20.0, interior_temp_c=20.0)
    passive = model.step(initial, ambient_temp_c=0, solar_w_m2=0, occupants=1, hvac_heat_w=0, dt_s=10)
    heated = model.step(initial, ambient_temp_c=0, solar_w_m2=0, occupants=1, hvac_heat_w=3000, dt_s=10)
    assert passive.state.air_temp_c < initial.air_temp_c
    assert heated.state.air_temp_c > passive.state.air_temp_c
