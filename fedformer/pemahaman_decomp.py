# demo_series_decomp_multi.py
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Building blocks
# ──────────────────────────────────────────────────────────────────────────────
class moving_avg(nn.Module):
    """
    Moving-average (1-D) yang menjaga panjang sekuen.
    """
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        left_pad  = self.kernel_size - 1 - (self.kernel_size - 1) // 2
        right_pad = (self.kernel_size - 1) // 2

        front = x[:, :1, :].repeat(1, left_pad, 1)
        back  = x[:, -1:, :].repeat(1, right_pad, 1)
        x_pad = torch.cat([front, x, back], dim=1)      # (B, T+pad, C)

        # AvgPool1d expects (B, C, T)
        out = self.avg(x_pad.permute(0, 2, 1))
        return out.permute(0, 2, 1)                     # back to (B, T, C)


class series_decomp_multi(nn.Module):
    """
    Multi-kernel series decomposition dengan soft attention.
    """
    def __init__(self, kernel_sizes):
        super().__init__()
        self.moving_avg = nn.ModuleList(
            [moving_avg(k) for k in kernel_sizes]
        )
        self.selector = nn.Linear(1, len(kernel_sizes))  # attention α_k

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        moving_means = []
        for ma in self.moving_avg:
            moving_means.append(ma(x).unsqueeze(-1))     # => (B, T, C, 1)
        moving_means = torch.cat(moving_means, dim=-1)    # (B, T, C, K)

        # soft-attention antar-kernel
        alpha = nn.functional.softmax(
            self.selector(x.unsqueeze(-1)), dim=-1
        )                                                 # (B, T, C, K)
        trend = (moving_means * alpha).sum(dim=-1)        # (B, T, C)
        residual = x - trend
        return residual, trend

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Demo data
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
B, T, C = 1, 24, 1
t = torch.arange(T, dtype=torch.float32)
signal = 0.5 * torch.sin(2 * np.pi * t / 12)     # musim 12-step
signal += 0.02 * t                               # tren linier ringan
signal += 0.05 * torch.randn(T)                  # noise
signal = signal.view(1, T, 1)                    # (B, T, C)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Decompose & plot
# ──────────────────────────────────────────────────────────────────────────────
decomp = series_decomp_multi([24])               # kernel_size = 24
residual, trend = decomp(signal)

plt.figure(figsize=(8, 4))
plt.plot(signal.squeeze().numpy(),  label='Original')
plt.plot(trend.squeeze().detach().numpy(),     label='Trend (24-MA)')
plt.plot(residual.squeeze().detach().numpy(),  label='Residual')
plt.title('series_decomp_multi demo  (kernel_size=[24])')
plt.xlabel('Time-step'); plt.legend(); plt.tight_layout()
plt.show()
