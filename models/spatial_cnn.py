"""
Spatial CNN Feature Extraction Branch.
Processes 2D multi-channel spatial rasters (satellite AOD, land use, road density, elevation)
using convolutional filters with batch normalization and leaky ReLU activations
to extract neighborhood spatial dispersion context representations.
"""

import torch
import torch.nn as nn


class SpatialCNNBranch(nn.Module):
    def __init__(self, in_channels: int = 4, out_dim: int = 32):
        super().__init__()
        # Input shape: (Batch, in_channels=4, H=5, W=5)
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=0), # Output size: (Batch, 32, 3, 3)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling -> (Batch, 32, 1, 1)
            nn.Flatten(),                 # (Batch, 32)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 4, 5, 5)
        Returns:
            Spatial feature representation of shape (Batch, out_dim)
        """
        feats = self.conv_net(x)
        return self.proj(feats)
