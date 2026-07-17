import numpy as np

from ev_thermal.components.heat_pump import HeatPumpModel
from ev_thermal.thermal_hydraulics.cold_plate import ColdPlate
from ev_thermal.thermal_hydraulics.fluid import glycol_properties
from ev_thermal.thermal_hydraulics.heat_exchanger import epsilon_ntu, Radiator
from ev_thermal.thermal_hydraulics.pipe import pipe_pressure_drop
from ev_thermal.thermal_hydraulics.pump import Pump


def test_fluid_pipe_and_pump_working_point():
    fluid = glycol_properties(25.0)
    assert 1000 < fluid.density_kg_m3 < 1100
    assert 3000 < fluid.cp_j_kgk < 4000
    low = pipe_pressure_drop(0.03, 0.02, 3.0, 4.0, fluid)
    high = pipe_pressure_drop(0.12, 0.02, 3.0, 4.0, fluid)
    assert high.pressure_drop_pa > low.pressure_drop_pa > 0
    point = Pump().working_point(speed_fraction=0.7, system_resistance_pa_per_kg2_s2=200_000)
    assert point.mass_flow_kg_s > 0
    assert point.hydraulic_power_w <= point.electrical_power_w


def test_cold_plate_and_heat_exchanger_conserve_direction():
    fluid = glycol_properties(25.0)
    result = ColdPlate().exchange(surface_temp_c=40, coolant_in_temp_c=25, mass_flow_kg_s=0.12, fluid=fluid)
    assert 0 < result.effectiveness <= 1
    assert result.heat_transfer_w > 0
    assert 25 < result.coolant_out_temp_c < 40
    eff = epsilon_ntu(ua_w_per_k=500, hot_capacity_w_per_k=800, cold_capacity_w_per_k=1000)
    assert 0 < eff < 1


def test_radiator_and_heat_pump_limits():
    radiator = Radiator()
    low = radiator.exchange(60, 25, coolant_mass_flow_kg_s=0.15, vehicle_speed_mps=0, fan_fraction=0.2)
    high = radiator.exchange(60, 25, coolant_mass_flow_kg_s=0.15, vehicle_speed_mps=20, fan_fraction=0.8)
    assert high.heat_rejected_w > low.heat_rejected_w
    assert high.fan_power_w > low.fan_power_w
    hp = HeatPumpModel()
    mild = hp.operate(thermal_request_w=4000, ambient_temp_c=10, sink_temp_c=35)
    cold = hp.operate(thermal_request_w=4000, ambient_temp_c=-20, sink_temp_c=35)
    assert mild.thermal_output_w > 0
    assert mild.cop > cold.cop >= 1.0
    assert np.isclose(mild.thermal_output_w, mild.electrical_power_w * mild.cop)

