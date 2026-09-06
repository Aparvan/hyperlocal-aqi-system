"""
Multimodal Deep Learning Architecture for Hyperlocal Air Quality Prediction.
Fuses three distinct heterogeneous modalities:
1. Spatial Branch: 2D Convolutional network processing satellite & geographic rasters.
2. Temporal Branch: Bidirectional LSTM with attention over weather & pollution time-series.
3. Tabular Branch: Dense projection of point-level meteorological & geo-spatial attributes.
Combines all representations via Cross-Modal Multi-Head Self-Attention
to predict fine-grained pollutant concentrations (PM2.5, PM10, NO2) and overall AQI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spatial_cnn import SpatialCNNBranch
from models.temporal_lstm import TemporalLSTMBranch


class TabularBranch(nn.Module):
    def __init__(self, in_features: int = 12, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusionNetwork(nn.Module):
    def __init__(
        self,
        spatial_channels: int = 4,
        temporal_features: int = 8,
        tabular_features: int = 12,
        embedding_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Modality Encoders
        self.spatial_encoder = SpatialCNNBranch(in_channels=spatial_channels, out_dim=embedding_dim)
        self.temporal_encoder = TemporalLSTMBranch(in_features=temporal_features, out_dim=embedding_dim, dropout=dropout)
        self.tabular_encoder = TabularBranch(in_features=tabular_features, out_dim=embedding_dim)
        
        # Cross-Modal Multi-Head Attention Fusion
        # Treats the 3 representations as tokens in a sequence: [Spatial, Temporal, Tabular]
        self.cross_modal_mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mha_norm = nn.LayerNorm(embedding_dim)
        
        # Fusion Multi-Layer Perceptron (MLP)
        # Fuses concatenated attended representations (3 * embedding_dim = 96)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Multi-Target Regression Prediction Heads
        # Outputs: [PM2.5, PM10, NO2, AQI]
        self.head_pm25 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_pm10 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_no2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_aqi = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(
        self,
        spatial_patch: torch.Tensor,
        temporal_seq: torch.Tensor,
        tabular_vec: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            spatial_patch: (Batch, 4, 5, 5)
            temporal_seq:  (Batch, 24, 8)
            tabular_vec:   (Batch, 12)
        Returns:
            Dictionary with predicted 'pm25', 'pm10', 'no2', 'aqi' and fused representations.
        """
        # 1. Encode individual modalities
        h_spatial = self.spatial_encoder(spatial_patch)   # (Batch, embedding_dim)
        h_temporal = self.temporal_encoder(temporal_seq) # (Batch, embedding_dim)
        h_tabular = self.tabular_encoder(tabular_vec)     # (Batch, embedding_dim)
        
        # 2. Form modality token matrix: (Batch, 3, embedding_dim)
        tokens = torch.stack([h_spatial, h_temporal, h_tabular], dim=1)
        
        # 3. Cross-Modal Attention
        attn_out, attn_weights = self.cross_modal_mha(tokens, tokens, tokens)
        tokens_refined = self.mha_norm(tokens + attn_out) # Residual + Norm
        
        # 4. Flatten and Project through Fusion MLP
        batch_size = tokens_refined.size(0)
        fused_vector = tokens_refined.reshape(batch_size, -1) # (Batch, 3 * embedding_dim)
        fused_repr = self.fusion_mlp(fused_vector)            # (Batch, 64)
        
        # 5. Output Heads (Enforce non-negativity with ReLU)
        pred_pm25 = F.relu(self.head_pm25(fused_repr))
        pred_pm10 = F.relu(self.head_pm10(fused_repr))
        pred_no2 = F.relu(self.head_no2(fused_repr))
        pred_aqi = F.relu(self.head_aqi(fused_repr))
        
        # Concatenate predictions: [Batch, 4]
        preds = torch.cat([pred_pm25, pred_pm10, pred_no2, pred_aqi], dim=1)
        
        res = {
            "preds": preds,
            "pm25": pred_pm25.squeeze(-1),
            "pm10": pred_pm10.squeeze(-1),
            "no2": pred_no2.squeeze(-1),
            "aqi": pred_aqi.squeeze(-1),
            "fused_embedding": fused_repr,
        }
        if return_attention:
            res["cross_modal_attention"] = attn_weights
            
        return res
