"""Compare three coolant-loop architectures and scan component size envelopes."""

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from ev_thermal.config import load_config
from ev_thermal.pipeline import run_architecture_comparison, run_architecture_sizing
from ev_thermal.visualization import plot_architecture_comparison, plot_sizing_feasibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "architecture")
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--sizing-duration", type=int, default=600)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs" / "default_config.yaml")
    comparison = run_architecture_comparison(config, duration_s=args.duration)
    sizing = run_architecture_sizing(config, duration_s=args.sizing_duration)
    comparison.to_csv(output / "architecture_comparison.csv", index=False)
    sizing.to_csv(output / "component_sizing_feasibility.csv", index=False)

    ranked = comparison.copy()
    for metric in ("max_battery_temp_c", "max_motor_temp_c", "pump_energy_kwh", "auxiliary_energy_kwh"):
        ranked[f"rank_{metric}"] = ranked.groupby("scenario")[metric].rank(method="average")
    rank_columns = [column for column in ranked if column.startswith("rank_")]
    ranking = ranked.groupby("architecture", as_index=False)[rank_columns].mean()
    ranking["engineering_rank_score"] = ranking[rank_columns].sum(axis=1)
    ranking = ranking.sort_values(["engineering_rank_score", "architecture"]).reset_index(drop=True)
    ranking.to_csv(output / "architecture_ranking.csv", index=False)

    plot_architecture_comparison(comparison, output / "architecture_comparison.png")
    plot_sizing_feasibility(sizing, output / "component_sizing_feasibility.png")
    summary = {
        "architectures": sorted(comparison["architecture"].unique().tolist()),
        "scenario_count": int(comparison["scenario"].nunique()),
        "comparison_rows": int(len(comparison)),
        "sizing_rows": int(len(sizing)),
        "feasible_sizing_points": int(sizing["feasible"].sum()),
        "recommended_by_equal_rank": ranking.iloc[0]["architecture"],
        "ranking_method": "equal sum of within-scenario ranks; lower is better",
    }
    (output / "architecture_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
