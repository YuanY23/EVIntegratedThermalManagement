"""Publication-oriented thermal-management figures with explicit engineering units."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_scenario_overview(frame: pd.DataFrame, path: str | Path) -> Path:
    """Plot temperatures, heat loads, liquid flow, and actuator commands."""
    output = _prepare_path(path)
    minutes = frame["time_s"] / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    axes[0].plot(minutes, frame["battery_core_temp_c"], label="Battery core")
    axes[0].plot(minutes, frame["motor_temp_c"], label="Motor")
    axes[0].plot(minutes, frame["inverter_temp_c"], label="Inverter")
    axes[0].plot(minutes, frame["cabin_temp_c"], label="Cabin")
    axes[0].plot(minutes, frame["ambient_temp_c"], label="Ambient", linestyle="--")
    axes[0].set_ylabel("Temperature (degC)")
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(minutes, frame["battery_heat_w"] / 1000, label="Battery heat")
    axes[1].plot(minutes, frame["powertrain_heat_w"] / 1000, label="Powertrain heat")
    axes[1].plot(minutes, frame["cabin_load_w"] / 1000, label="Cabin load")
    axes[1].set_ylabel("Thermal load (kW)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(minutes, frame["battery_flow_kg_s"], label="Battery loop")
    axes[2].plot(minutes, frame["powertrain_flow_kg_s"], label="Powertrain loop")
    axes[2].set_ylabel("Coolant flow (kg/s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    axes[3].plot(minutes, frame["pump_fraction"], label="Battery pump")
    axes[3].plot(minutes, frame["fan_fraction"], label="Fan")
    axes[3].plot(minutes, frame["compressor_fraction"], label="Compressor")
    axes[3].plot(minutes, frame["ptc_fraction"], label="PTC")
    axes[3].set_ylabel("Command (-)")
    axes[3].set_xlabel("Time (min)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(ncol=4, fontsize=8)
    axes[3].grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_strategy_comparison(table: pd.DataFrame, path: str | Path) -> Path:
    output = _prepare_path(path)
    metrics = ["max_battery_temp_c", "cabin_comfort_rmse_c", "auxiliary_energy_kwh",
               "consumption_kwh_100km"]
    labels = ["Max battery temp (degC)", "Cabin RMSE (degC)",
              "Auxiliary energy (kWh)", "Consumption (kWh/100km)"]
    grouped = table.groupby("strategy")[metrics].mean()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    colors = ["#287271", "#d97706"]
    for axis, metric, label in zip(axes, metrics, labels):
        grouped[metric].plot.bar(ax=axis, color=colors, rot=0)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_training_history(history: list[dict], path: str | Path) -> Path:
    output = _prepare_path(path)
    frame = pd.DataFrame(history)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(frame["epoch"], frame["train_loss"], label="Train")
    axis.plot(frame["epoch"], frame["validation_loss"], label="Validation")
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

