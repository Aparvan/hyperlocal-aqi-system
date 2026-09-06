"""
Temporal LSTM Feature Extraction Branch.
Processes historical and forecasted meteorological and pollution sequences
using a multi-layer Bidirectional LSTM with temporal self-attention pooling
to capture diurnal cycles, meteorological trends, and atmospheric stagnation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTMBranch(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, num_layers: int = 2, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Temporal Attention Pooling
        # Computes importance weight for each of the 24 time steps
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len=24, In_Features=8)
        Returns:
            Temporal context representation of shape (Batch, out_dim)
        """
        # lstm_out shape: (Batch, Seq_Len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights across time dimension
        attn_scores = self.attention_net(lstm_out) # (Batch, Seq_Len, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (Batch, Seq_Len, 1)
        
        # Context vector: weighted sum over time steps
        context = torch.sum(lstm_out * attn_weights, dim=1) # (Batch, hidden_dim * 2)
        
        return self.proj(context)
