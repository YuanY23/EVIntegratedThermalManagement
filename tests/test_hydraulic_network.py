import numpy as np
import pytest

from ev_thermal.thermal_hydraulics.fluid import glycol_properties
from ev_thermal.thermal_hydraulics.network import (
    ColdPlateElement,
    PipeElement,
    QuadraticLoss,
    SeriesHydraulicNetwork,
)
from ev_thermal.thermal_hydraulics.pump import Pump
from ev_thermal.thermal_hydraulics.valve import ValveElement


def test_series_network_closes_pump_and_component_pressure_budget():
    network = SeriesHydraulicNetwork(
        (
            PipeElement("supply_return", diameter_m=0.019, length_m=7.0, local_loss_coefficient=8.0),
            ColdPlateElement("battery_cold_plate"),
            QuadraticLoss("radiator_and_manifold", 120_000.0),
        )
    )

    stopped = network.solve(Pump(), speed_fraction=0.0, coolant_temp_c=25.0)
    point = network.solve(Pump(), speed_fraction=0.75, coolant_temp_c=25.0)

    assert stopped.status == "stopped"
    assert stopped.point.mass_flow_kg_s == stopped.point.electrical_power_w == 0
    assert point.converged
    assert point.point.mass_flow_kg_s > 0
    assert np.isclose(point.point.pressure_rise_pa, sum(point.component_pressure_drop_pa.values()), rtol=1e-7)
    assert abs(point.closure_residual_pa) < 1e-4
    assert 0 < point.point.efficiency <= 1
    assert point.point.hydraulic_power_w <= point.point.electrical_power_w


def test_more_valve_resistance_reduces_flow_and_closed_valve_is_diagnostic():
    open_network = SeriesHydraulicNetwork((
        QuadraticLoss("base", 120_000.0),
        ValveElement("three_way_valve", full_open_resistance_pa_per_kg2_s2=30_000.0, opening_fraction=1.0),
    ))
    throttled_network = SeriesHydraulicNetwork((
        QuadraticLoss("base", 120_000.0),
        ValveElement("three_way_valve", full_open_resistance_pa_per_kg2_s2=30_000.0, opening_fraction=0.25),
    ))
    blocked_network = SeriesHydraulicNetwork((
        ValveElement("closed_valve", full_open_resistance_pa_per_kg2_s2=30_000.0, opening_fraction=0.0),
    ))

    open_result = open_network.solve(Pump(), 0.7, 25.0)
    throttled_result = throttled_network.solve(Pump(), 0.7, 25.0)
    blocked_result = blocked_network.solve(Pump(), 0.7, 25.0)

    assert throttled_result.point.mass_flow_kg_s < open_result.point.mass_flow_kg_s
    assert blocked_result.status == "blocked"
    assert not blocked_result.converged
    assert blocked_result.failed_component == "closed_valve"


def test_quadratic_network_matches_legacy_working_point_and_rejects_invalid_topology():
    resistance = 200_000.0
    pump = Pump()
    legacy = pump.working_point(0.7, resistance)
    network = SeriesHydraulicNetwork((QuadraticLoss("legacy_equivalent", resistance),))
    result = network.solve(pump, 0.7, 25.0)

    assert np.isclose(result.point.mass_flow_kg_s, legacy.mass_flow_kg_s, rtol=1e-8)
    assert np.isclose(result.point.pressure_rise_pa, legacy.pressure_rise_pa, rtol=1e-8)
    with pytest.raises(ValueError, match="unique"):
        SeriesHydraulicNetwork((QuadraticLoss("same", 1.0), QuadraticLoss("same", 2.0)))


def test_pipe_and_cold_plate_elements_have_positive_finite_losses():
    fluid = glycol_properties(25.0)
    elements = (
        PipeElement("pipe", diameter_m=0.02, length_m=4.0, local_loss_coefficient=3.0),
        ColdPlateElement("plate"),
    )
    losses = [element.pressure_drop_pa(0.12, fluid) for element in elements]
    assert all(np.isfinite(loss) and loss > 0 for loss in losses)
