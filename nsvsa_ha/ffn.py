"""
Feed-Forward Network sublayer

Transformers derive substantial expressive power from FFN layers, not solely
from attention. This module provides a standard 4× expansion FFN with:
  - SwiGLU gating (better than ReLU for language, same cost as 2-layer MLP)
  - Pre-norm residual (more stable than post-norm for deep models)
  - Configurable expansion ratio

SwiGLU: Shazeer (2020) "GLU Variants Improve Transformer"
    FFN_SwiGLU(x) = (xW₁ ⊙ SiLU(xW₂)) W₃

We implement it with a single fused linear for the gate+up projection
(same parameter count, one GEMM instead of two).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

        hidden = SiLU(x @ W_gate) ⊙ (x @ W_up)
        out    = hidden @ W_down

    With expansion_ratio r, intermediate dim = floor(r * d_model * 2/3)
    rounded to a multiple of 64 (hardware alignment).  This matches the
    LLaMA / Mistral convention and keeps params ≈ same as a plain 4× FFN.
    """

    def __init__(
        self,
        d_model: int,
        expansion_ratio: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        # Intermediate dim: 2/3 factor compensates for the two-branch structure
        intermediate = int(d_model * expansion_ratio * 2 / 3)
        # Round up to nearest multiple of 64
        intermediate = ((intermediate + 63) // 64) * 64

        self.d_model = d_model
        self.intermediate = intermediate

        # Fused gate+up projection in one matrix → one GEMM
        self.gate_up_proj = nn.Linear(d_model, 2 * intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., d_model]
        # Split into gate and up branches
        proj = self.gate_up_proj(x)                         # [..., 2*intermediate]
        gate, up = proj.chunk(2, dim=-1)                    # each [..., intermediate]
        hidden = F.silu(gate) * up                          # SwiGLU
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)                        # [..., d_model]


class GEGLUFFFN(nn.Module):
    """
    GEGLU variant: replaces SiLU with GELU.
    Slightly smoother gradient landscape, marginally slower.
    """

    def __init__(self, d_model: int, expansion_ratio: float = 4.0,
                 dropout: float = 0.0, bias: bool = False):
        super().__init__()
        intermediate = int(d_model * expansion_ratio * 2 / 3)
        intermediate = ((intermediate + 63) // 64) * 64
        self.gate_up_proj = nn.Linear(d_model, 2 * intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        hidden = F.gelu(gate) * up
        return self.down_proj(self.dropout(hidden))


def build_ffn(
    d_model: int,
    expansion_ratio: float = 4.0,
    variant: str = "swiglu",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Module:
    """Factory for FFN variants."""
    if variant == "swiglu":
        return SwiGLUFFN(d_model, expansion_ratio, dropout, bias)
    elif variant == "geglu":
        return GEGLUFFFN(d_model, expansion_ratio, dropout, bias)
    else:
        raise ValueError(f"Unknown FFN variant '{variant}'. Choose 'swiglu' or 'geglu'.")
