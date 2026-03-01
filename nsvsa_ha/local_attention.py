"""
Local Windowed Causal Attention

Precise short-range reasoning complement to the VSA global state.

Design decisions:
  - Window size W (default 128): tokens only attend to the last W positions.
    This is O(W·L) instead of O(L²), and at W=128 covers essentially all
    grammatical dependencies.
  - Causal masking: no information leaks from future tokens.
  - RoPE applied to Q and K: relative-position aware, length-generalizing.
  - Uses torch.nn.functional.scaled_dot_product_attention (FlashAttention
    kernel when available) for memory and speed efficiency.

Griffin insight (De et al. 2024): combining a local attention window with a
recurrent compressed state (here: soft VSA) matches or exceeds pure attention
quality while being sub-quadratic in sequence length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .rope import RotaryEmbedding
from .cache import AttentionCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for Grouped Query Attention.

    [B, Hkv, L, D] → [B, H, L, D] where H = Hkv * n_rep.
    No-op when n_rep == 1 (standard MHA).
    """
    if n_rep == 1:
        return x
    B, Hkv, L, D = x.shape
    x = x[:, :, None, :, :].expand(B, Hkv, n_rep, L, D)
    return x.reshape(B, Hkv * n_rep, L, D)


class LocalWindowedAttention(nn.Module):
    """
    Multi-head causal self-attention restricted to a sliding window.

    Tokens at position t can attend to positions max(0, t-W+1) … t only.
    This is implemented by building an additive bias mask rather than
    dropping tokens, which keeps the implementation simple while letting
    PyTorch / FlashAttention handle the CUDA kernel.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
        num_kv_heads: int = 0,
        qk_norm: bool = False,
    ):
        """
        Args:
            d_model:      Model / embedding dimension.
            num_heads:    Number of query attention heads. d_model must be divisible.
            window_size:  W – maximum number of past tokens to attend to.
            dropout:      Attention dropout probability.
            bias:         Whether to include bias in Q/K/V/O projections.
            num_kv_heads: Number of KV heads for GQA (0 = same as num_heads → MHA).
            qk_norm:      L2-normalize Q and K with learned per-head temperature.
        """
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        # GQA: fewer KV heads → smaller KV cache, same quality
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.kv_group_size = num_heads // self.num_kv_heads

        # Separate Q and KV projections (enables GQA with different head counts)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # QK-Norm: prevents attention logit growth, stabilizes deep layers
        self.qk_norm = qk_norm
        if qk_norm:
            # Per-head learned temperature, init ≈ sqrt(head_dim) so initial
            # logits match standard scaled dot-product range
            self.qk_temperature = nn.Parameter(
                torch.full((num_heads, 1, 1), math.sqrt(self.head_dim))
            )

        self.attn_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=4096,   # extend lazily if needed
        )

        self._attn_mask_cache: dict = {}  # seq_len -> mask tensor

    # ------------------------------------------------------------------
    # Mask building
    # ------------------------------------------------------------------

    def _get_causal_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Additive bias mask: 0 for allowed positions, -inf for forbidden.

        Shape: [1, 1, seq_len, seq_len]
        """
        key = (seq_len, str(device))
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]

        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # Causal positions (attend only to current and past)
        allowed = row_idx >= col_idx

        # Window restriction: only last window_size positions
        if self.window_size < seq_len:
            allowed = allowed & ((row_idx - col_idx) < self.window_size)

        # Convert to additive bias
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = mask.masked_fill(allowed, 0.0)
        mask = mask.unsqueeze(0).unsqueeze(0)   # [1, 1, L, L]

        self._attn_mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                          # [B, L, d_model]
        positions: Optional[torch.Tensor] = None, # [L] int, default 0..L-1
        kv_cache: Optional[AttentionCache] = None, # cached K/V from previous steps
        use_cache: bool = False,                   # return updated cache?
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
        """
        Returns:
            output:    [B, L, d_model]
            new_cache: AttentionCache if use_cache else None
        """
        B, L, _ = x.shape
        H = self.num_heads
        Hkv = self.num_kv_heads
        Dh = self.head_dim

        # Separate Q and KV projections (GQA: Hkv ≤ H)
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)          # [B, H,   L, Dh]
        kv = self.kv_proj(x).view(B, L, 2, Hkv, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                            # [B, Hkv, L, Dh]

        # Apply RoPE to Q and K (broadcasts over different head counts)
        if positions is None:
            positions = torch.arange(L, device=x.device)
        q, k = self.rope(q, k, positions)

        new_cache: Optional[AttentionCache] = None

        if kv_cache is not None:
            # ── Decode mode: prepend cached K/V (compact Hkv heads) ──
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)

            # Trim to window size (evict oldest)
            if k.shape[2] > self.window_size:
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]

            if use_cache:
                new_cache = AttentionCache(k=k, v=v)  # store compact

            # Expand KV heads to match Q heads
            k_attn = repeat_kv(k, self.kv_group_size)
            v_attn = repeat_kv(v, self.kv_group_size)

            # QK-Norm: L2 normalize Q/K, apply learned temperature
            if self.qk_norm:
                q = F.normalize(q, p=2, dim=-1) * self.qk_temperature
                k_attn = F.normalize(k_attn, p=2, dim=-1)

            # No causal mask needed: all cached entries are past positions,
            # and window trimming ensures we're within W.
            out = F.scaled_dot_product_attention(
                q, k_attn, v_attn, dropout_p=0.0,
                scale=1.0 if self.qk_norm else None,
            )
        else:
            # ── Normal forward (training / prefill) ──────────────────
            if use_cache:
                # Cache last window_size K/V — compact form (RoPE'd)
                new_cache = AttentionCache(
                    k=k[:, :, -self.window_size:].contiguous(),
                    v=v[:, :, -self.window_size:].contiguous(),
                )

            # Expand KV heads to match Q heads
            k_attn = repeat_kv(k, self.kv_group_size)
            v_attn = repeat_kv(v, self.kv_group_size)

            # QK-Norm: L2 normalize Q/K, apply learned temperature
            if self.qk_norm:
                q = F.normalize(q, p=2, dim=-1) * self.qk_temperature
                k_attn = F.normalize(k_attn, p=2, dim=-1)

            attn_mask = self._get_causal_window_mask(L, x.device)
            sdpa_scale = 1.0 if self.qk_norm else None

            try:
                out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    scale=sdpa_scale,
                )
            except Exception:
                # Fallback for older PyTorch
                if self.qk_norm:
                    scores = torch.matmul(q, k_attn.transpose(-2, -1)) + attn_mask
                else:
                    scores = torch.matmul(q, k_attn.transpose(-2, -1)) / self.scale + attn_mask
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.attn_dropout(attn_weights)
                out = torch.matmul(attn_weights, v_attn)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), new_cache
