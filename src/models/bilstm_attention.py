"""
bilstm_attention.py - Step 7: Proposed BiLSTM + Attention Model
================================================================
OUR NOVEL ARCHITECTURE (main contribution):
  Feature Sequence → BiLSTM → Attention Layer → Dense → Output

The attention mechanism learns which timesteps in the sequence
are most important for stampede detection, providing:
  1. Better performance (focuses on critical moments)
  2. Explainability (attention weights show what the model focuses on)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention mechanism.

    Learns to assign importance weights to each timestep in the sequence.
    This is key for explainability - we can visualize which temporal
    windows contributed most to the stampede detection.
    """

    def __init__(self, hidden_size, attention_units=64):
        super().__init__()
        self.W = nn.Linear(hidden_size, attention_units, bias=False)
        self.V = nn.Linear(attention_units, 1, bias=False)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) - weighted sum of timesteps
            attention_weights: (batch, seq_len) - importance of each timestep
        """
        # Score each timestep
        energy = torch.tanh(self.W(lstm_output))   # (batch, seq_len, attn_units)
        scores = self.V(energy).squeeze(-1)         # (batch, seq_len)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum (context vector)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                       # (batch, seq_len, hidden)
        ).squeeze(1)                          # (batch, hidden)

        return context, attention_weights


class BiLSTMAttentionClassifier(nn.Module):
    """BiLSTM + Attention classifier (our proposed model).

    Architecture:
        Input → BiLSTM → Attention → BatchNorm → Dropout → Dense → Output

    This model addresses two limitations of the base paper:
        1. BiLSTM captures bidirectional temporal context (vs unidirectional LSTM)
        2. Attention focuses on critical timesteps (vs equal weighting)
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, attention_units=64):
        super().__init__()
        self.model_name = "BiLSTM_Attention"
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention over BiLSTM output (hidden*2 because bidirectional)
        self.attention = AdditiveAttention(hidden_size * 2, attention_units)

        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, input_size)
            return_attention: if True, also return attention weights

        Returns:
            output: (batch, 1) logits
            attention_weights: (batch, seq_len) - only if return_attention=True
        """
        lstm_out, _ = self.lstm(x)                    # (batch, seq_len, hidden*2)
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*2)
        context = self.bn(context)
        context = self.dropout(context)
        output = self.fc(context)                     # (batch, 1)

        if return_attention:
            return output, attn_weights
        return output
