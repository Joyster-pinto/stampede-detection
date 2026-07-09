"""
bilstm_model.py - Bidirectional LSTM Classifier
=================================================
BiLSTM captures both forward and backward temporal dependencies.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.model_name = "BiLSTM"
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        # BiLSTM output is 2*hidden_size (forward + backward)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden*2)
        last_out = lstm_out[:, -1, :]     # (batch, hidden*2)
        last_out = self.bn(last_out)
        last_out = self.dropout(last_out)
        return self.fc(last_out)
