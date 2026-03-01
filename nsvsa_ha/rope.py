"""
Rotary Position Embeddings (RoPE)

Replaces the spec's frozen random bipolar position matrix with learned-friendly
RoPE that integrates cleanly with both:

  1. Local windowed attention (standard Q/K rotation)
  2. VSA binding  – the local position vectors p_k are just the RoPE rotation
     of a canonical unit vector at position k, staying on the unit hypersphere.

Key properties:
  - Relative-position invariant: dot(RoPE(q,m), RoPE(k,n)) depends only on m-n
  - Length extrapolation: works beyond training length (especially with YaRN / NTK)
  - Zero parameters for basic RoPE (frequencies computed analytically)
  - Compatible with bipolar vectors: rotation preserves ||x||₂

Reference: Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position
Embedding" https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cos/sin cache for RoPE.

    For head dimension d_h, pairs of dimensions (2i, 2i+1) are rotated by θᵢ·m:
        θᵢ = 1 / base^(2i / d_h)

    Args:
        seq_len:  Maximum sequence length to cache.
        head_dim: Dimension of a single attention head (must be even).
        base:     Frequency base (10000 standard; lower = faster variation).
        device:   Target device.
        dtype:    Target dtype.

    Returns:
        cos_cache: [seq_len, head_dim]
        sin_cache: [seq_len, head_dim]
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # θᵢ  shape [head_dim // 2]
    i = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    thetas = 1.0 / (base ** (i / head_dim))

    # positions  shape [seq_len]
    positions = torch.arange(seq_len, device=device, dtype=dtype)

    # outer product: [seq_len, head_dim // 2]
    freqs = torch.outer(positions, thetas)

    # interleave to get [seq_len, head_dim]
    cos_cache = freqs.cos().repeat_interleave(2, dim=-1)
    sin_cache = freqs.sin().repeat_interleave(2, dim=-1)

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate x by swapping and negating paired dimensions.

    For x = [x₀, x₁, x₂, x₃, ...]:
    returns [-x₁, x₀, -x₃, x₂, ...]

    This implements the 2D rotation matrix without materializing it explicitly.
    """
    # reshape to [..., d//2, 2], swap last dim, negate first, flatten back
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    # interleave: [-x2, x1]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return rotated


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE rotation to x.

    RoPE(x, m) = x ⊙ cos(m·Θ) + rotate_half(x) ⊙ sin(m·Θ)

    Args:
        x:   [..., seq_len, head_dim]
        cos: [seq_len, head_dim]  or broadcastable
        sin: [seq_len, head_dim]  or broadcastable

    Returns:
        Rotated tensor, same shape as x.
    """
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """
    RoPE module with cached cos/sin tables.

    Supports on-the-fly cache extension for sequences longer than
    the initial max_seq_len.

    Usage (attention):
        rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
        q, k = rope(q, k, seq_positions)

    Usage (VSA position vectors):
        pos_vecs = rope.get_position_vectors(positions, d)
        # pos_vecs are unit-norm and encode absolute position
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10_000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Register as non-persistent buffers (recomputed on device move)
        cos_cache, sin_cache = build_rope_cache(max_seq_len, head_dim, base)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _maybe_extend_cache(self, seq_len: int):
        if seq_len <= self.cos_cache.shape[0]:
            return
        # Rebuild for longer sequence
        cos_cache, sin_cache = build_rope_cache(
            seq_len, self.head_dim, self.base,
            device=self.cos_cache.device,
            dtype=self.cos_cache.dtype,
        )
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def forward(
        self,
        q: torch.Tensor,          # [B, heads, seq_len, head_dim]
        k: torch.Tensor,          # [B, heads, seq_len, head_dim]
        positions: Optional[torch.Tensor] = None,  # [seq_len] int positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        If positions is None, assumes contiguous 0, 1, ..., seq_len-1.
        """
        seq_len = q.shape[-2]
        self._maybe_extend_cache(seq_len if positions is None else positions.max().item() + 1)

        if positions is None:
            cos = self.cos_cache[:seq_len]              # [seq_len, head_dim]
            sin = self.sin_cache[:seq_len]
        else:
            cos = self.cos_cache[positions]             # [seq_len, head_dim]
            sin = self.sin_cache[positions]

        # Broadcast over batch and heads: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)

        return q_rot, k_rot

    def get_position_vectors(
        self,
        positions: torch.Tensor,  # [n] int positions
        d: int,
    ) -> torch.Tensor:             # [n, d]
        """
        Generate L2-normalized position vectors for use as VSA anchors.

        Each vector is the RoPE rotation of the all-ones vector at that position,
        projected / tiled to match full dimension d.  These replace the frozen
        random bipolar P matrix and carry the same quasi-orthogonality guarantees
        while being differentiable and length-generalizing.

        Similarity between two position vectors decays with |m - n| because
        RoPE frequencies decorrelate them – the same property we want for VSA.
        """
        head_dim_eff = min(self.head_dim, d)
        # Ensure even
        if head_dim_eff % 2 != 0:
            head_dim_eff -= 1

        self._maybe_extend_cache(positions.max().item() + 1)

        cos = self.cos_cache[positions, :head_dim_eff]  # [n, head_dim_eff]
        sin = self.sin_cache[positions, :head_dim_eff]

        # Rotate the canonical all-ones vector
        base_vec = torch.ones(
            len(positions), head_dim_eff,
            device=cos.device, dtype=cos.dtype
        )
        rotated = apply_rope(base_vec, cos, sin)         # [n, head_dim_eff]

        # Tile / truncate to full dimension d
        if head_dim_eff < d:
            repeats = (d + head_dim_eff - 1) // head_dim_eff
            rotated = rotated.repeat(1, repeats)[:, :d]

        return torch.nn.functional.normalize(rotated, p=2, dim=-1)
