"""Compact sequence-to-sequence LSTM for three thermal-load trajectories."""

import torch
from torch import nn


class ThermalLoadLSTM(nn.Module):
    """Encode measured history and autoregressively decode future loads.

    Input shape is ``(batch, history_steps, features)``. The decoder receives its
    previous normalized three-load prediction plus an encoded context vector and
    returns ``(batch, horizon_steps, 3)``. It predicts information for the rule
    supervisor; no output has actuator semantics.
    """

    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 48,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=effective_dropout)
        self.decoder = nn.LSTM(output_size + hidden_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=effective_dropout)
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                        nn.Linear(hidden_size // 2, output_size))

    def forward(self, history: torch.Tensor, horizon_steps: int,
                teacher_targets: torch.Tensor | None = None,
                teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        encoded, (hidden, cell) = self.encoder(history)
        context = encoded[:, -1, :]
        if self.input_size >= self.output_size + 8:
            # Dataset features end with normalized histories of the same three
            # targets. Feature and target scalers see identical raw channels, so
            # the final history point is in the decoder's normalized units.
            previous = history[:, -1, -self.output_size:]
        else:
            previous = torch.zeros(history.size(0), self.output_size,
                                   device=history.device, dtype=history.dtype)
        outputs = []
        for step in range(horizon_steps):
            decoder_input = torch.cat([previous, context], dim=-1).unsqueeze(1)
            decoded, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            # Residual decoding starts from persistence, a strong short-horizon
            # thermal baseline, and learns the future change rather than level.
            prediction = previous + 0.2 * self.projection(decoded[:, 0, :])
            outputs.append(prediction)
            if teacher_targets is not None and torch.rand((), device=history.device) < teacher_forcing_ratio:
                previous = teacher_targets[:, step, :]
            else:
                previous = prediction
        return torch.stack(outputs, dim=1)
