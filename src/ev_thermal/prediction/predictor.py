"""Saved-model inference, physical-unit conversion, and health checks."""

from pathlib import Path

import joblib
import numpy as np
import torch

from ..control.predictive import ForecastSummary
from .model import ThermalLoadLSTM


class ThermalLoadPredictor:
    def __init__(self, model_dir: str | Path):
        root = Path(model_dir)
        payload = torch.load(root / "thermal_load_lstm.pt", map_location="cpu", weights_only=True)
        self.model = ThermalLoadLSTM(payload["input_size"], payload["output_size"],
                                     payload["hidden_size"], payload["num_layers"])
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.feature_scaler = joblib.load(root / "feature_scaler.joblib")
        self.target_scaler = joblib.load(root / "target_scaler.joblib")

    def predict(self, history: np.ndarray, horizon_steps: int = 60) -> np.ndarray:
        if history.ndim != 2 or history.shape[1] != self.feature_scaler.n_features_in_:
            raise ValueError("history must have shape (history_steps, feature_count)")
        normalized = self.feature_scaler.transform(history).astype(np.float32)
        with torch.no_grad():
            forecast = self.model(torch.from_numpy(normalized).unsqueeze(0), horizon_steps)[0].numpy()
        physical = self.target_scaler.inverse_transform(forecast)
        return physical

    def health(self, forecast: np.ndarray) -> tuple[bool, str]:
        if forecast.shape != (60, 3):
            return False, "shape"
        if not np.isfinite(forecast).all():
            return False, "nonfinite"
        if np.max(np.abs(forecast)) > 100_000:
            return False, "out_of_range"
        return True, "ok"

    def summarize_forecast(self, forecast: np.ndarray) -> ForecastSummary:
        valid, reason = self.health(forecast)
        if not valid:
            return ForecastSummary.invalid(reason)
        return ForecastSummary(True, float(np.max(forecast[:, 0])),
                               float(np.mean(forecast[:, 1])),
                               float(np.mean(forecast[:, 2])))

    def summary(self, rows: list[dict]) -> ForecastSummary:
        """Adapt integrated-simulator records to the model's history contract."""
        required_history = 60
        if len(rows) < required_history:
            return ForecastSummary.invalid("history_warmup")
        recent = rows[-required_history:]
        history = np.asarray([[row["speed_mps"], row["ambient_temp_c"],
                               row["battery_core_temp_c"], row["motor_temp_c"],
                               row["cabin_temp_c"], row["battery_total_power_w"],
                               row["pump_fraction"], row["time_s"],
                               row["battery_heat_w"], row["powertrain_heat_w"],
                               row["cabin_load_w"]]
                              for row in recent], dtype=np.float32)
        try:
            return self.summarize_forecast(self.predict(history, horizon_steps=60))
        except (ValueError, RuntimeError) as exc:
            return ForecastSummary.invalid(type(exc).__name__)
