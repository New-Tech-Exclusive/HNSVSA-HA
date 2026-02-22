"""
ARNL Utilities
==============

Shared primitives used across the architecture:
  - RMSNorm (pre-norm)
  - SwiGLU feed-forward network
  - LoRA adapter for Phase 2/3 training
  - Semantic hashing for Hyper-Adjacency Map keys
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# RMSNorm — used as the pre-norm throughout ARNL
# ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU-gated feed-forward network (Shazeer, 2020).

    Projects d_model → 2 * d_ffn (gate + value), applies SiLU
    gating, then projects back to d_model.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# ────────────────────────────────────────────────────────────────
# LoRA Adapter — lightweight fine-tuning for Phases 2 & 3
# ────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen nn.Linear.

    During Phase 2/3 training the base linear remains frozen while
    a low-rank ΔW = B @ A (rank r) is learned and added.

    Parameters
    ----------
    base_linear : nn.Linear
        The frozen weight to adapt.
    rank : int
        LoRA rank *r*.
    alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout applied to the LoRA path.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero init for B (start at identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the original weight
        for p in self.base_linear.parameters():
            p.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base_linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A) @ self.lora_B.T
        return base_out + lora_out * self.scaling


def apply_lora(
    module: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[set[str]] = None,
) -> nn.Module:
    """Replace targeted nn.Linear layers in *module* with LoRA-wrapped versions.

    Parameters
    ----------
    module : nn.Module
        The module (typically System 1 LLSU) to adapt.
    target_modules : set[str] | None
        Names of sub-modules to wrap.  If ``None``, wraps all Linear layers.

    Returns
    -------
    nn.Module
        The same module, modified in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            pass  # already wrapped — skip to keep apply_lora idempotent
        elif isinstance(child, nn.Linear):
            if target_modules is None or name in target_modules:
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            apply_lora(child, rank, alpha, dropout, target_modules)
    return module


# ────────────────────────────────────────────────────────────────
# Semantic Hashing — content-addressable keys for System 2
# ────────────────────────────────────────────────────────────────

def semantic_hash(anchor_ids: list[int]) -> int:
    """Compute a deterministic hash from a set of semantic anchor token IDs.

    The IDs are sorted to make the hash order-invariant.  We use a
    simple polynomial rolling hash for cross-session determinism.
    """
    sorted_ids = tuple(sorted(anchor_ids))
    h = 0
    for tok_id in sorted_ids:
        h = (h * 131 + tok_id) & 0xFFFF_FFFF_FFFF_FFFF
    return h


def semantic_hash_combinations(anchor_ids: list[int], min_k: int = 2) -> list[int]:
    """Generate hash keys for all subsets of *anchor_ids* of size ≥ min_k.

    This enables partial-match retrieval when only a subset of the
    original anchors is present in a new context.

    Returns a list of hash values ordered by subset size descending
    (full set first → best match priority).
    """
    from itertools import combinations

    keys: list[int] = []
    for k in range(len(anchor_ids), min_k - 1, -1):
        for combo in combinations(sorted(anchor_ids), k):
            keys.append(semantic_hash(list(combo)))
    return keys


# ────────────────────────────────────────────────────────────────
# Causal mask helper
# ────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a (seq_len, seq_len) boolean causal mask (True = masked)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
