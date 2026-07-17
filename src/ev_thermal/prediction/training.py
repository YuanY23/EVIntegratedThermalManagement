"""Deterministic LSTM training, early stopping, and physical-unit evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import DatasetSplit, ThermalSequenceDataset
from .model import ThermalLoadLSTM


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 35
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 6
    seed: int = 42
    hidden_size: int = 48
    num_layers: int = 2


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]
    metrics: dict[str, float]
    model_path: str


def _loader(arrays, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ThermalSequenceDataset(arrays), batch_size=batch_size,
                      shuffle=shuffle, generator=generator)


def _loss_on(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features, targets.shape[1])
            losses.append(nn.functional.mse_loss(predictions, targets).item())
    return float(np.mean(losses)) if losses else float("nan")


def forecast_metrics(predicted: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Return physical-unit target metrics and 60/180/300 s checkpoints."""
    if predicted.shape != true.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("predicted and true must have matching shape (samples, horizon, 3)")
    names = ("battery_heat_w", "powertrain_heat_w", "cabin_load_w")
    metrics: dict[str, float] = {}
    for index, name in enumerate(names):
        error = predicted[:, :, index] - true[:, :, index]
        metrics[f"mae_{name}"] = float(np.mean(np.abs(error)))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(error**2)))
        denominator = float(np.sum((true[:, :, index] - np.mean(true[:, :, index])) ** 2))
        metrics[f"r2_{name}"] = float(1.0 - np.sum(error**2) / denominator) if denominator > 1e-12 else 0.0
    # The configured 60-step horizon uses a 5 s sample period. Checkpoint errors
    # expose long-horizon drift that one aggregate MAE can conceal.
    for seconds, step in ((60, 11), (180, 35), (300, 59)):
        index = min(step, predicted.shape[1] - 1)
        metrics[f"mae_all_targets_at_{seconds}s"] = float(
            np.mean(np.abs(predicted[:, index, :] - true[:, index, :])))
    return metrics


def evaluate_physical(model: nn.Module, arrays, target_scaler) -> dict[str, float]:
    model.eval()
    if len(arrays.features) == 0:
        return {"mae_battery_heat_w": float("nan")}
    with torch.no_grad():
        normalized = model(torch.as_tensor(arrays.features, dtype=torch.float32),
                           arrays.targets.shape[1]).numpy()
    pred = target_scaler.inverse_transform(normalized.reshape(-1, 3)).reshape(normalized.shape)
    true = target_scaler.inverse_transform(arrays.targets.reshape(-1, 3)).reshape(arrays.targets.shape)
    return forecast_metrics(pred, true)


def train_model(split: DatasetSplit, output_dir: str | Path,
                options: TrainingOptions | None = None) -> TrainingResult:
    options = options or TrainingOptions()
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = ThermalLoadLSTM(split.train.features.shape[-1], 3, options.hidden_size, options.num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=1e-5)
    train_loader = _loader(split.train, options.batch_size, True, options.seed)
    val_loader = _loader(split.validation, options.batch_size, False, options.seed)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    checkpoint = output / "thermal_load_lstm.pt"
    for epoch in range(1, options.epochs + 1):
        model.train()
        epoch_losses = []
        teacher_ratio = max(0.0, 0.5 * (1.0 - epoch / max(options.epochs, 1)))
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features, targets.shape[1], targets, teacher_ratio)
            loss = nn.functional.mse_loss(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        validation_loss = _loss_on(model, val_loader)
        train_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "input_size": split.train.features.shape[-1],
                        "output_size": 3, "hidden_size": options.hidden_size,
                        "num_layers": options.num_layers}, checkpoint)
        else:
            stale += 1
            if stale >= options.patience:
                break
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    metrics = evaluate_physical(model, split.test, split.target_scaler)
    joblib.dump(split.feature_scaler, output / "feature_scaler.joblib")
    joblib.dump(split.target_scaler, output / "target_scaler.joblib")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainingResult(best_epoch, best_loss, history, metrics, str(checkpoint))
