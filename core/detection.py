# core/detection.py
"""Detection algorithms like CFAR and neighborhood filtering."""
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from .filters import generate_gamma_kernel

class CFAR(nn.Module):
    """Constant False Alarm Rate (CFAR) detector."""
    def __init__(self, p: Dict[str, Any]):
        super().__init__()
        self.T = p.get('T', 3.0)
        self.kernel_size = p.get('kernel_size', (21, 21))
        self.eps = p.get('eps', 1e-9)
        
        gr = p.get('gamma_radial_params', (1.0, 10.0))
        n, peak_radius = float(gr[0]), float(gr[1])
        mu = (n - 1) / peak_radius if n > 1 and peak_radius > 0 else 0
        self.bg_kernel = generate_gamma_kernel(2, n, mu, self.kernel_size).unsqueeze(0).unsqueeze(0)

    def set_threshold(self, T: float) -> None:
        self.T = float(T)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kernel = self.bg_kernel.to(x.device, x.dtype)
        pad_h, pad_w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        padding = (pad_w, pad_w, pad_h, pad_h)
        
        x_padded = F.pad(x, padding, mode='reflect')
        mean = F.conv2d(x_padded, kernel, padding=0)
        
        x_pow2_padded = F.pad(x.pow(2), padding, mode='reflect')
        mean_sq = F.conv2d(x_pow2_padded, kernel, padding=0)
        
        std = torch.sqrt(F.relu(mean_sq - mean.pow(2)))
        z_score = (x - mean) / (std + self.eps)
        mask = (z_score > self.T).to(torch.uint8)
        
        return z_score, mask, std, mean

@lru_cache(maxsize=None)
def _get_neigh_kernel(k_size, device, dtype):
    """Cached factory for neighborhood kernels."""
    return torch.ones((1, 1, k_size, k_size), device=device, dtype=dtype)

def apply_neighborhood_filter(mask: torch.Tensor, k: int, kernel_size: int = 3) -> torch.Tensor:
    """Applies a neighborhood filter to a binary mask."""
    if k <= 1:
        return mask
    
    kernel = _get_neigh_kernel(kernel_size, mask.device, torch.float32)
    neighborhood_sum = F.conv2d(mask.float().unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)
    
    return torch.logical_and(mask, neighborhood_sum >= k).to(torch.uint8)

