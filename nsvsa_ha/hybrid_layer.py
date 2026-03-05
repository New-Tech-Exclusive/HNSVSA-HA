"""
Hybrid NSVSA Layer

Per-layer architecture (pre-norm residual throughout):

    ┌─────────────────────────────────────────────┐
    │  x  (B, L, d)                               │
    │   │                                         │
    │   ├─ LayerNorm ─► LocalWindowedAttn ─► + ──►│ x₁
    │   │                                         │
    │   ├─ LayerNorm ─► SoftVSA query     ─► + ──►│ x₂
    │   │               (updates VSA state)       │
    │   │                                         │
    │   └─ LayerNorm ─► SwiGLU FFN        ─► + ──►│ x_out
    │                                             │
    │  also returns: updated VSA recurrent state  │
    └─────────────────────────────────────────────┘

Gradient flow notes:
  - Local attention has full real gradients (standard softmax).
  - Soft VSA has full real gradients (L2-normalize, no sgn/STE).
  - FFN has full real gradients.
  - Skip connections prevent vanishing gradients across depth.
  - VSA state is differentiable because SoftBundle is a normalized mean.

Training stability:
  - Warmup: freeze VSA decay parameter for first N steps to avoid
    degenerate state collapse to zero.
  - The attention and VSA components share the same embedding space,
    which helps them co-adapt rather than fight each other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .local_attention import LocalWindowedAttention
from .soft_vsa import SoftVSAStateUpdate
from .ffn import build_ffn
from .cache import LayerCache
from .rmsnorm import RMSNorm


class HybridNSVSALayer(nn.Module):
    """
    One layer of the hybrid Local-Attention + Soft-VSA + FFN architecture.

    Args:
        d_model:      Model dimension.
        num_heads:    Attention heads (d_model % num_heads == 0).
        window_size:  Local attention window W (e.g. 128 or 256).
        group_size:   VSA group size K (e.g. 64).
        max_groups:   Maximum number of groups (max_seq_len // K).
        ffn_ratio:    FFN expansion ratio (default 4.0).
        dropout:      Dropout probability (attention + FFN).
        ffn_variant:  "swiglu" (default) or "geglu".
        layer_norm_eps: RMSNorm epsilon.
        num_kv_heads: KV heads for GQA (0 = MHA, same as num_heads).
        qk_norm:      L2-normalize Q/K with learned temperature.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        group_size: int,
        max_groups: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        ffn_variant: str = "swiglu",
        layer_norm_eps: float = 1e-6,
        gate_init_bias: float = 0.0,
        num_kv_heads: int = 0,
        qk_norm: bool = False,
    ):
        super().__init__()

        # Sub-modules
        self.local_attn = LocalWindowedAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
        )

        self.soft_vsa = SoftVSAStateUpdate(
            d=d_model,
            group_size=group_size,
            max_groups=max_groups,
        )

        self.ffn = build_ffn(
            d_model=d_model,
            expansion_ratio=ffn_ratio,
            variant=ffn_variant,
            dropout=dropout,
        )

        # Projection to bring VSA queries into residual stream
        self.vsa_out_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-norm RMSNorm (one per sub-component)
        self.norm_attn = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm_vsa  = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm_ffn  = RMSNorm(d_model, eps=layer_norm_eps)

        # Input-dependent mixing gate: how much VSA contributes per token.
        # At init, weight=0 and bias=gate_init_bias, so the gate starts as
        # a static sigmoid(bias) ≈ 0.12 (for bias=-2); during training each
        # token learns to modulate how much long-range VSA context it uses.
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        self.gate_proj.bias.data.fill_(gate_init_bias)

    def forward(
        self,
        x: torch.Tensor,                           # [B, L, d]
        local_positions: torch.Tensor,             # [K, d]  RoPE-derived
        macro_positions: torch.Tensor,             # [max_groups, d]
        token_seq_positions: Optional[torch.Tensor] = None,  # [L] int
        layer_cache: Optional[LayerCache] = None,  # cache from previous step
        use_cache: bool = False,                   # build / update cache?
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[LayerCache]]:
        """
        Args:
            x:                    Input embeddings [B, L, d].
            local_positions:      Position vectors for intra-group binding [K, d].
            macro_positions:      Position vectors for group-level binding [max_groups, d].
            token_seq_positions:  Absolute int positions for RoPE (default 0..L-1).
            layer_cache:          Previous-step cache (None for prefill/training).
            use_cache:            Whether to return an updated cache.

        Returns:
            x_out:      Updated embeddings [B, L, d].
            vsa_state:  VSA recurrent state after this layer [B, d].
            new_cache:  LayerCache (if use_cache) or None.
        """
        # ── 1. Local windowed attention ──────────────────────────────
        attn_kv_cache = layer_cache.attn if layer_cache is not None else None
        attn_out, new_attn_cache = self.local_attn(
            self.norm_attn(x), positions=token_seq_positions,
            kv_cache=attn_kv_cache, use_cache=use_cache,
        )
        x = x + attn_out

        # ── 2. Soft VSA state update ─────────────────────────────────
        # Input is the current x (post-attention), RMSNorm'd only (no L2
        # normalization — magnitude information from the residual stream is
        # preserved for richer group bundling).
        x_norm = self.norm_vsa(x)

        vsa_cache = layer_cache.vsa if layer_cache is not None else None
        vsa_queries, vsa_state, new_vsa_cache = self.soft_vsa(
            x_norm, local_positions, macro_positions,
            vsa_cache=vsa_cache, use_cache=use_cache,
        )  # [B, L, d], [B, d], Optional[VSACache]

        # Input-dependent gate: each token decides per-dimension how much
        # long-range VSA context to incorporate.
        gate = torch.sigmoid(self.gate_proj(x_norm))  # [B, L, d]
        vsa_contrib = gate * self.vsa_out_proj(vsa_queries)
        x = x + vsa_contrib

        # ── 3. FFN ───────────────────────────────────────────────────
        x = x + self.ffn(self.norm_ffn(x))

        new_layer_cache: Optional[LayerCache] = None
        if use_cache and new_attn_cache is not None and new_vsa_cache is not None:
            new_layer_cache = LayerCache(attn=new_attn_cache, vsa=new_vsa_cache)

        return x, vsa_state, new_layer_cache


class HybridNSVSALayerStack(nn.Module):
    """
    Stack of HybridNSVSALayer with shared positional infrastructure.

    The VSA recurrent state is passed forward layer-to-layer:
    this gives each layer an increasingly refined compressed context.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        window_size: int,
        group_size: int,
        max_groups: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        ffn_variant: str = "swiglu",
        layer_norm_eps: float = 1e-6,
        gate_init_bias: float = 0.0,
        num_kv_heads: int = 0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridNSVSALayer(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                group_size=group_size,
                max_groups=max_groups,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                ffn_variant=ffn_variant,
                layer_norm_eps=layer_norm_eps,
                gate_init_bias=gate_init_bias,
                num_kv_heads=num_kv_heads,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,                # [B, L, d]
        local_positions: torch.Tensor,  # [K, d]
        macro_positions: torch.Tensor,  # [max_groups, d]
        token_seq_positions: Optional[torch.Tensor] = None,
        layer_caches: Optional[List[LayerCache]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, list, Optional[List[LayerCache]]]:
        """
        Returns:
            x:              Final hidden states [B, L, d] (post final norm).
            vsa_states:     List of per-layer VSA states [[B, d], ...].
            new_caches:     List of LayerCache (if use_cache) or None.
        """
        vsa_states = []
        new_layer_caches: Optional[List[LayerCache]] = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            cache_i = layer_caches[i] if layer_caches is not None else None
            x, vsa_state, new_cache_i = layer(
                x, local_positions, macro_positions, token_seq_positions,
                layer_cache=cache_i, use_cache=use_cache,
            )
            vsa_states.append(vsa_state)
            if use_cache:
                new_layer_caches.append(new_cache_i)

        return self.final_norm(x), vsa_states, new_layer_caches
    def forward_partial(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        local_positions: torch.Tensor,
        macro_positions: torch.Tensor,
        token_seq_positions: Optional[torch.Tensor] = None,
        layer_caches: Optional[List[LayerCache]] = None,
        use_cache: bool = False,
        apply_final_norm: bool = False,
    ) -> Tuple[torch.Tensor, list, Optional[List[LayerCache]]]:
        """
        Run a contiguous sub-range of layers [start, end).

        Used by the reasoning block to separately execute base layers
        and reasoning layers.

        Args:
            x:                 Input hidden states [B, L, d].
            start:             First layer index (inclusive).
            end:               Last layer index (exclusive).
            local_positions:   [K, d] RoPE-derived local position vectors.
            macro_positions:   [max_groups, d] macro position vectors.
            token_seq_positions: [L] absolute int positions for RoPE.
            layer_caches:      Caches for the layers in [start, end).
                               Length must be (end - start) if provided.
            use_cache:         Build/update cache.
            apply_final_norm:  Apply self.final_norm to the output.

        Returns:
            x, vsa_states, new_caches (same semantics as forward).
        """
        vsa_states = []
        new_layer_caches: Optional[List[LayerCache]] = [] if use_cache else None

        for idx, layer_idx in enumerate(range(start, end)):
            cache_i = layer_caches[idx] if layer_caches is not None else None
            x, vsa_state, new_cache_i = self.layers[layer_idx](
                x, local_positions, macro_positions, token_seq_positions,
                layer_cache=cache_i, use_cache=use_cache,
            )
            vsa_states.append(vsa_state)
            if use_cache:
                new_layer_caches.append(new_cache_i)

        if apply_final_norm:
            x = self.final_norm(x)

        return x, vsa_states, new_layer_caches