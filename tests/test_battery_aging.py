import pytest

from ev_thermal.charging.aging import AgingParameters, incremental_aging


def test_high_temperature_and_c_rate_increase_relative_cycle_damage():
    parameters = AgingParameters()
    mild = incremental_aging(200.0, 25.0, 0.55, 600.0, 197.0, parameters)
    hot = incremental_aging(200.0, 45.0, 0.55, 600.0, 197.0, parameters)
    high_rate = incremental_aging(400.0, 25.0, 0.55, 300.0, 197.0, parameters)

    assert hot.throughput_ah == pytest.approx(mild.throughput_ah)
    assert hot.cycle_damage > mild.cycle_damage
    assert high_rate.throughput_ah == pytest.approx(mild.throughput_ah)
    assert high_rate.cycle_damage > mild.cycle_damage


def test_zero_current_has_no_cycle_damage_and_invalid_parameters_are_rejected():
    result = incremental_aging(0.0, 35.0, 0.8, 600.0, 197.0, AgingParameters())
    assert result.cycle_damage == 0
    assert result.calendar_damage > 0
    assert incremental_aging(200.0, 35.0, 0.8, 0.0, 197.0, AgingParameters()).total_damage == 0
    with pytest.raises(ValueError):
        AgingParameters(reference_temperature_c=-300.0)
