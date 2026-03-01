"""
Inference cache for efficient autoregressive generation.

During generation, instead of recomputing the full forward pass for every token:
  - Local attention reuses cached K/V (only the last W entries)
  - Soft VSA maintains its recurrent state incrementally

Cost per generated token: O(W·d) attention + O(d) VSA update per layer,
versus O(L·d) for the naïve re-forward approach.

Usage:
    # Prefill — process prompt, build cache
    out = model(prompt_ids, use_cache=True)
    cache = out["cache"]

    # Decode — one token at a time, O(W) per step
    out = model(next_token_id, cache=cache, use_cache=True)
    cache = out["cache"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class AttentionCache:
    """KV cache for one local-attention layer.

    Stores the last ≤ W RoPE-rotated K and V tensors so that each
    new query token only needs to compute its own Q and attend to the
    cached window — no recomputation of past positions.
    """
    k: torch.Tensor   # [B, H, cache_len, Dh]  (cache_len ≤ W)
    v: torch.Tensor   # [B, H, cache_len, Dh]


@dataclass
class VSACache:
    """Incremental VSA state for one layer.

    Instead of recomputing all group vectors and the global EMA from
    scratch, we keep:
      - The global state S (EMA after the last completed group)
      - A running accumulator for the group currently being filled
      - Counts so we know when to commit the current group to S
    """
    global_state: torch.Tensor   # [B, d] — S after last completed group
    group_accum: torch.Tensor    # [B, d] — sum of bound tokens in current group
    group_count: int             # tokens accumulated in current group (0..K-1)
    num_completed_groups: int    # groups fully committed to global state


@dataclass
class LayerCache:
    """Combined cache for one HybridNSVSALayer."""
    attn: AttentionCache
    vsa: VSACache


@dataclass
class ModelCache:
    """Full generation cache — one LayerCache per model layer."""
    layers: List[LayerCache]
    seq_len: int   # total tokens processed so far (for RoPE position offsets)
