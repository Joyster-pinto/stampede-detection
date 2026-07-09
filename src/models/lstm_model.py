"""
lstm_model.py - Step 6: Baseline LSTM Classifier
==================================================
Reproduces the base paper's LSTM architecture for comparison.
Architecture: Feature Sequence → LSTM → Dense → Sigmoid
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Baseline LSTM classifier (base paper architecture)."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.model_name = "LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output: (batch, 1) logits
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden)
        # Take last timestep output
        last_out = lstm_out[:, -1, :]     # (batch, hidden)
        last_out = self.bn(last_out)
        last_out = self.dropout(last_out)
        output = self.fc(last_out)        # (batch, 1)
        return output
