"""
Physics-Informed Neural Network (PINN) Loss Module.
Enforces the 2D Atmospheric Advection-Diffusion Partial Differential Equation:
dC/dt + u*(dC/dx) + v*(dC/dy) = D * (d^2C/dx^2 + d^2C/dy^2) + S - k*C
Where:
- C: Pollutant concentration field (ug/m3)
- (u, v): Wind vector field (m/s)
- D: Turbulent horizontal atmospheric eddy diffusivity (~50-200 m2/s)
- S: Surface emission source proxy
- k: Chemical loss / dry deposition rate
Computes finite-difference PDE residuals across the spatial prediction grid to
penalize unphysical artifacts and ensure physical mass consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    def __init__(self, dx_meters: float = 2000.0, dy_meters: float = 2000.0, diffusivity: float = 120.0, decay_rate: float = 1e-5):
        super().__init__()
        self.dx = dx_meters
        self.dy = dy_meters
        self.D = diffusivity
        self.k = decay_rate

        # 2D Finite-Difference Sobel/Central Difference Kernels for 1st and 2nd derivatives
        # Central difference: dC/dx = (C[i, j+1] - C[i, j-1]) / (2*dx)
        kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], dtype=torch.float32)
        ky = torch.tensor([[[[-0.5], [0.0], [0.5]]]], dtype=torch.float32)
        
        # 5-point Laplace operator for Laplacian: d2C/dx2 + d2C/dy2
        # [ 0,  1,  0]
        # [ 1, -4,  1] / dx^2
        # [ 0,  1,  0]
        laplace = torch.tensor([[[[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]]]], dtype=torch.float32)

        self.register_buffer("kernel_x", kx)
        self.register_buffer("kernel_y", ky)
        self.register_buffer("kernel_laplace", laplace)

    def forward(
        self,
        grid_concentration: torch.Tensor,
        wind_u: float,
        wind_v: float,
        dt_seconds: float = 3600.0,
        prev_concentration: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            grid_concentration: 2D or 4D Tensor (Batch, 1, Rows, Cols) of predicted concentrations
            wind_u: Eastward wind velocity (m/s)
            wind_v: Northward wind velocity (m/s)
            dt_seconds: Temporal step (1 hour = 3600s)
            prev_concentration: Optional previous hour concentration field
        Returns:
            Dictionary with 'pinn_loss', 'advection_residual', 'diffusion_residual'
        """
        if grid_concentration.dim() == 2:
            C = grid_concentration.unsqueeze(0).unsqueeze(0)
        elif grid_concentration.dim() == 3:
            C = grid_concentration.unsqueeze(1)
        else:
            C = grid_concentration

        device = C.device

        # 1. Spatial 1st derivatives (Advection terms)
        dC_dx = F.conv2d(C, self.kernel_x.to(device), padding=(0, 1)) / self.dx
        dC_dy = F.conv2d(C, self.kernel_y.to(device), padding=(1, 0)) / self.dy

        # Advective transport: u*(dC/dx) + v*(dC/dy)
        advection = wind_u * dC_dx + wind_v * dC_dy

        # 2. Spatial 2nd derivative (Turbulent diffusion term: D * Lap(C))
        laplace_C = F.conv2d(C, self.kernel_laplace.to(device), padding=1) / (self.dx ** 2)
        diffusion = self.D * laplace_C

        # 3. Temporal Rate of Change: dC/dt
        if prev_concentration is not None:
            if prev_concentration.dim() == 2:
                C_prev = prev_concentration.unsqueeze(0).unsqueeze(0)
            elif prev_concentration.dim() == 3:
                C_prev = prev_concentration.unsqueeze(1)
            else:
                C_prev = prev_concentration
            dC_dt = (C - C_prev.to(device)) / dt_seconds
        else:
            # Quasi-steady state assumption: dC/dt approx balance
            dC_dt = torch.zeros_like(advection)

        # 4. Decay and deposition
        loss_term = self.k * C

        # 5. PDE Residual: R = dC/dt + advection - diffusion + loss
        pde_residual = dC_dt + advection - diffusion + loss_term

        # Exclude border cells affected by boundary padding
        valid_residual = pde_residual[:, :, 1:-1, 1:-1]
        
        # Mean Squared Physics Loss
        loss_value = torch.mean(valid_residual ** 2)

        return {
            "pinn_loss": loss_value,
            "mean_residual": float(torch.mean(torch.abs(valid_residual)).item()),
            "max_residual": float(torch.max(torch.abs(valid_residual)).item()),
        }
