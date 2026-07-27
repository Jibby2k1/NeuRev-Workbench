"""Bounded learnable residuals for a fixed multi-hypothesis CFAR bank.

This component is intentionally separate from the v4-v6 screen.  It is ready
for the conditional C3 stage, but the experiment runner must not train it until
the C2 scientific gate passes.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def model_class():
    """Return a lazily imported torch module for multiplicative kernel tuning."""
    import torch
    import torch.nn as nn

    class BoundedKernelResidual(nn.Module):
        """Tune weights inside fixed supports without changing kernel topology."""

        def __init__(self, bank: dict[str, np.ndarray], max_log_gain: float = 0.05):
            super().__init__()
            if not 0 < max_log_gain <= 0.10:
                raise ValueError("max_log_gain must be in (0, 0.10]")
            self.max_log_gain = float(max_log_gain)
            for name in ("test", "reference", "sectors", "core"):
                base = torch.as_tensor(bank[name], dtype=torch.float32)
                self.register_buffer(f"base_{name}", base)
                self.register_parameter(f"raw_{name}", nn.Parameter(torch.zeros_like(base)))

        @staticmethod
        def _normalize_like_kernel(value):
            dims = tuple(range(value.ndim - 2, value.ndim))
            return value / value.sum(dims, keepdim=True).clamp_min(1e-12)

        def _bounded(self, name: str):
            import torch

            base = getattr(self, f"base_{name}")
            raw = getattr(self, f"raw_{name}")
            multiplier = torch.exp(self.max_log_gain * torch.tanh(raw))
            return self._normalize_like_kernel(base * multiplier)

        def kernels(self) -> dict[str, Any]:
            return {
                name: self._bounded(name)
                for name in ("test", "reference", "sectors", "core")
            }

        def residual_penalty(self):
            import torch

            values = [
                torch.tanh(getattr(self, f"raw_{name}")).square().mean()
                for name in ("test", "reference", "sectors", "core")
            ]
            return torch.stack(values).mean()

        def diagnostics(self) -> dict[str, float]:
            import torch

            with torch.no_grad():
                maximum = max(
                    float(torch.tanh(getattr(self, f"raw_{name}")).abs().max().item())
                    for name in ("test", "reference", "sectors", "core")
                )
            return {
                "max_log_gain": self.max_log_gain,
                "max_fraction_of_bound_used": maximum,
                "support_topology_fixed": True,
            }

    return BoundedKernelResidual
