"""Compare no-preconditioning and rule-based arrival conditioning before fast charge."""

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from ev_thermal.artifacts import write_upgrade_suite_manifest
from ev_thermal.simulation.charging_scenarios import (
    charging_scenario_names,
    make_charging_scenario,
    simulate_trip_charge,
)
from ev_thermal.visualization import plot_preconditioning_comparison


def _row(result) -> dict:
    payload = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name not in {"route_timeseries", "charge_timeseries"}
    }
    payload["charge_time_min"] = payload["charge_time_s"] / 60.0
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "charging")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    maximum_power_ledger_residual_w = 0.0
    for name in charging_scenario_names():
        scenario = make_charging_scenario(name)
        for strategy in ("none", "rule"):
            result = simulate_trip_charge(scenario, strategy)
            rows.append(_row(result))
            maximum_power_ledger_residual_w = max(
                maximum_power_ledger_residual_w,
                float(result.charge_timeseries["power_ledger_residual_w"].abs().max()),
            )
            result.route_timeseries.to_csv(
                output / f"route_{name}_{strategy}.csv", index=False
            )
            result.charge_timeseries.to_csv(
                output / f"charge_{name}_{strategy}.csv", index=False
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "preconditioning_comparison.csv", index=False)
    plot_preconditioning_comparison(comparison, output / "preconditioning_comparison.png")

    cold = make_charging_scenario("cold_arrival")
    robustness_cases = {
        "preview_unavailable": replace(cold, route_preview_valid=False),
        "route_cancelled": replace(cold, route_active=False),
        "station_derated": replace(cold, station_power_w=100_000.0),
        "late_station_selection": replace(cold, route_time_s=600),
    }
    robustness = []
    for case, scenario in robustness_cases.items():
        result = simulate_trip_charge(scenario, "rule")
        robustness.append({"case": case, **_row(result)})
    robustness_table = pd.DataFrame(robustness)
    robustness_table.to_csv(output / "preconditioning_robustness.csv", index=False)

    summary = {
        "scenario_count": len(charging_scenario_names()),
        "baseline_strategies": ["none", "rule"],
        "comparison_rows": len(comparison),
        "all_charge_events_completed": bool((comparison["charge_status"] == "charge_complete").all()),
        "maximum_power_ledger_residual_w": maximum_power_ledger_residual_w,
        "aging_claim_scope": "relative strategy comparison only",
    }
    (output / "preconditioning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_upgrade_suite_manifest(ROOT, "charging", output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
