"""
Root Mean Square Layer Normalization

    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = √(mean(x²) + ε)

Compared to LayerNorm:
  - No mean-centering → preserves representation diversity
  - ~15% faster (one fewer reduction operation)
  - Same training stabilization benefits
  - Used by LLaMA, Mistral, Gemma, Griffin

Reference:
    Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root-mean-square normalization with a learned scale parameter.

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Epsilon for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability, cast back
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x / rms).to(input_dtype)
        return x * self.weight
