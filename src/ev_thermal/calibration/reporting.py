"""Compact engineering figures for calibration and V&V evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_calibration_summary(parameter_table: pd.DataFrame, truth: dict[str, float],
                             validation_metrics: pd.DataFrame,
                             local_table: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    names = parameter_table["parameter"].tolist()
    short_names = [name.split(".")[-1].replace("_", "\n") for name in names]
    normalized = np.asarray([
        estimate / truth[name] for name, estimate in zip(names, parameter_table["estimate"])
    ])
    axes[0].bar(short_names, normalized, color="#287271")
    axes[0].axhline(1.0, color="#d97706", linestyle="--", label="Hidden truth")
    axes[0].set_ylabel("Estimate / truth (-)")
    axes[0].set_title("Synthetic parameter recovery")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        validation_metrics["parameter_set"],
        validation_metrics["combined_rmse_c"],
        color=["#9ca3af", "#287271"],
    )
    axes[1].set_ylabel("Holdout RMSE (degC)")
    axes[1].set_title("Independent validation")
    axes[1].grid(axis="y", alpha=0.25)

    heatmap = local_table.pivot(
        index="parameter", columns="metric", values="normalized_sensitivity"
    ).loc[names]
    image = axes[2].imshow(heatmap.to_numpy(), cmap="coolwarm", aspect="auto")
    axes[2].set_yticks(range(len(names)), [name.split(".")[-1] for name in names], fontsize=8)
    axes[2].set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=35, ha="right", fontsize=8)
    axes[2].set_title("Local normalized sensitivity")
    fig.colorbar(image, ax=axes[2], shrink=0.75)

    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
