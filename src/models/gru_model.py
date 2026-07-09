"""
gru_model.py - GRU Classifier
===============================
GRU variant for comparison. Fewer parameters than LSTM.
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """GRU classifier for comparison."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.model_name = "GRU"
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
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
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        last_out = self.bn(last_out)
        last_out = self.dropout(last_out)
        return self.fc(last_out)
