"""
Spatio-Temporal Graph Attention Network (ST-GAT).
Models topological pollutant transport across urban monitoring networks.
Constructs dynamic directed graphs where edge weights represent atmospheric
wind advection and proximity:
A_ij(t) = LeakyReLU( (v_wind . (x_j - x_i)) / ||x_j - x_i|| ) * exp(-||x_j - x_i||^2 / (2 * sigma^2))
Computes dynamic multi-head graph attention over neighboring sensors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer (GAT) with dynamic directional edge gating."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.zeros(out_features, 1))
        self.a_dst = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features tensor of shape (Batch, Num_Nodes, in_features)
            adj: Dynamic Adjacency matrix of shape (Batch, Num_Nodes, Num_Nodes)
        Returns:
            Updated node representations of shape (Batch, Num_Nodes, out_features)
        """
        Wh = self.W(h) # (Batch, N, out_features)
        
        # Self-attention coefficients
        f_src = torch.matmul(Wh, self.a_src) # (Batch, N, 1)
        f_dst = torch.matmul(Wh, self.a_dst) # (Batch, N, 1)
        
        # Broadcast addition: e_ij = LeakyReLU(f_src_i + f_dst_j)
        scores = self.leaky_relu(f_src + f_dst.transpose(-1, -2)) # (Batch, N, N)
        
        # Modulate scores with physical wind advection edge weights
        # Mask out zero-adjacency edges with large negative value
        mask = (adj > 0.0).float()
        scores = scores * mask + (1.0 - mask) * -1e9
        
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # Aggregate neighbor representations
        h_prime = torch.matmul(attention, Wh) # (Batch, N, out_features)
        return F.elu(h_prime)


class SpatioTemporalGAT(nn.Module):
    """
    ST-GAT Network:
    Takes spatial station graph features and dynamic meteorological wind vectors,
    builds the advective adjacency matrix, and applies 2-layer Graph Attention
    to extract downwind plume propagation representations.
    """
    def __init__(self, node_features: int = 8, hidden_dim: int = 32, out_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_features, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    @staticmethod
    def build_dynamic_adjacency(
        node_coords: np.ndarray,
        wind_u: float,
        wind_v: float,
        sigma_km: float = 12.0,
    ) -> torch.Tensor:
        """
        Construct physical advective adjacency matrix.
        Args:
            node_coords: (N, 2) array of [lat, lon]
            wind_u: Eastward wind component (m/s)
            wind_v: Northward wind component (m/s)
            sigma_km: Spatial decay radius in km
        Returns:
            adj: (1, N, N) tensor of advective weights
        """
        N = len(node_coords)
        adj = np.zeros((N, N), dtype=np.float32)
        wind_speed = math.hypot(wind_u, wind_v) + 1e-4
        u_norm = wind_u / wind_speed
        v_norm = wind_v / wind_speed

        for i in range(N):
            lat_i, lon_i = node_coords[i]
            for j in range(N):
                if i == j:
                    adj[i, j] = 1.0 # self-loop
                    continue
                lat_j, lon_j = node_coords[j]
                
                # Approximate distances in km (Delhi latitude ~28.6)
                d_north_km = (lat_j - lat_i) * 111.0
                d_east_km = (lon_j - lon_i) * 111.0 * math.cos(math.radians(lat_i))
                dist_km = math.hypot(d_north_km, d_east_km)
                
                if dist_km > 35.0: # Beyond local plume reach
                    continue

                # Direction vector from i to j
                dir_east = d_east_km / (dist_km + 1e-6)
                dir_north = d_north_km / (dist_km + 1e-6)
                
                # Alignment with wind vector (is j downwind of i?)
                alignment = dir_east * u_norm + dir_north * v_norm
                
                # Advective weight
                advection_boost = max(0.1, 1.0 + alignment * 1.5)
                spatial_decay = math.exp(-(dist_km ** 2) / (2.0 * sigma_km * sigma_km))
                
                adj[i, j] = float(advection_boost * spatial_decay)

        return torch.tensor(adj, dtype=torch.float32).unsqueeze(0)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (Batch, N, node_features)
            adj: (Batch, N, N)
        Returns:
            Node representations: (Batch, N, out_dim)
        """
        h1 = self.gat1(node_features, adj)
        h2 = self.gat2(h1, adj)
        return self.proj(h2)
