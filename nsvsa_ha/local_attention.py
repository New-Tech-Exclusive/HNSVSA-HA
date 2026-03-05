"""
Local Windowed Causal Attention

Precise short-range reasoning complement to the VSA global state.

Design decisions:
  - Window size W (default 128): tokens only attend to the last W positions.
    This is O(W·L) instead of O(L²), and at W=128 covers essentially all
    grammatical dependencies.
  - Causal masking: no information leaks from future tokens.
  - RoPE applied to Q and K: relative-position aware, length-generalizing.
  - Uses FlashAttention-2 via torch SDPA with is_causal=True when possible.
    Falls back to a Triton fused sliding-window kernel, then to masked SDPA.

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

# --------------------------------------------------------------------------
# Triton fused sliding-window causal attention kernel
# --------------------------------------------------------------------------
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass

if _HAS_TRITON:
    @triton.jit
    def _fwd_sliding_window_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len,
        window_size: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        sm_scale,
    ):
        """Fused sliding-window causal attention forward kernel.

        Each program instance computes a BLOCK_M×HEAD_DIM tile of the output.
        K/V are iterated in BLOCK_N chunks, masked for causal + window.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // stride_qh  # not used directly; strides encode layout
        # Ignore batch/head decomposition — we rely on strides for indexing.

        m_start = pid_m * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)

        # Pointers into Q for this block
        q_ptrs = Q + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        o_ptrs = Out + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

        # Load Q tile  [BLOCK_M, HEAD_DIM]
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len), other=0.0)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Determine K/V iteration range for this Q block
        # Causal: can only attend to columns <= max(offs_m)
        # Window: can only attend to columns >= min(offs_m) - window_size + 1
        col_end = min((m_start + BLOCK_M), seq_len)
        col_start = max(0, m_start - window_size + 1)
        # Align col_start down to BLOCK_N boundary
        col_start = (col_start // BLOCK_N) * BLOCK_N

        for n_start in range(col_start, col_end, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            # Load K/V tiles
            k_ptrs = K + pid_bh * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
            v_ptrs = V + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

            k = tl.load(k_ptrs, mask=(offs_n[None, :] < seq_len), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len), other=0.0)

            # QK^T  [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, k) * sm_scale

            # Causal mask: row >= col
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            # Window mask: row - col < window_size
            window_mask = (offs_m[:, None] - offs_n[None, :]) < window_size
            # Validity mask (don't read past seq_len)
            valid_mask = offs_n[None, :] < seq_len

            mask = causal_mask & window_mask & valid_mask
            qk = tl.where(mask, qk, float("-inf"))

            # Online softmax update
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            l_i = l_i * alpha + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
            acc = acc * alpha[:, None] + tl.dot(tl.exp(qk - m_new[:, None]).to(v.dtype), v)
            m_i = m_new

        # Final normalization
        acc = acc / l_i[:, None]

        # Store output
        tl.store(o_ptrs, acc.to(q.dtype), mask=(offs_m[:, None] < seq_len))


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


def _triton_sliding_window_attention(
    q: torch.Tensor,   # [B, H, L, D]
    k: torch.Tensor,   # [B, H, L, D]
    v: torch.Tensor,   # [B, H, L, D]
    window_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Launch the Triton fused sliding-window causal attention kernel."""
    B, H, L, D = q.shape
    out = torch.empty_like(q)

    # Block sizes — tuned for SM 8.x (RTX 30xx / A100)
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (
        (L + BLOCK_M - 1) // BLOCK_M,  # tiles over sequence
        B * H,                           # one program per (batch, head)
    )

    _fwd_sliding_window_kernel[grid](
        q, k, v, out,
        q.stride(0) * q.stride(1),  # stride_qb  (unused — folded into pid_bh)
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(1), v.stride(2), v.stride(3),
        out.stride(1), out.stride(1), out.stride(2), out.stride(3),
        seq_len=L,
        window_size=window_size,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        sm_scale=sm_scale,
    )
    return out


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

    @torch._dynamo.disable
    def _get_causal_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Additive bias mask: 0 for allowed positions, -inf for forbidden.

        Shape: [1, 1, seq_len, seq_len]

        @torch._dynamo.disable makes this function opaque to AOT autograd:
        the returned tensor enters the compiled region as a plain constant,
        so its view chain (.unsqueeze/masked_fill) is never tracked by
        functionalization — which was causing corrupt view metadata replays.
        """
        key = (seq_len, str(device))
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]

        # Evict oldest entry when cache exceeds 8 entries (curriculum varies seq_len)
        if len(self._attn_mask_cache) >= 8:
            self._attn_mask_cache.pop(next(iter(self._attn_mask_cache)))

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

        Kernel selection (training / prefill):
          1. If seq_len <= window_size: pure causal SDPA with is_causal=True
             → routes to FlashAttention-2 (no explicit mask tensor needed).
          2. Else if Triton is available: fused sliding-window Triton kernel
             with causal + window masking in one pass — no mask allocation.
          3. Fallback: additive mask SDPA (math kernel).
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

            sdpa_scale = 1.0 if self.qk_norm else None
            dropout_p = self.attn_dropout.p if self.training else 0.0

            if L <= self.window_size:
                # ── Path 1: Full causal — FlashAttention via is_causal ──
                # No window restriction needed when the entire sequence fits
                # within the window.  is_causal=True avoids allocating a mask
                # tensor, enabling the Flash SDP kernel (2-4× faster, O(1) mem).
                out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn,
                    dropout_p=dropout_p,
                    is_causal=True,
                    scale=sdpa_scale,
                )
            elif (
                _HAS_TRITON
                and q.is_cuda
                and not torch.is_grad_enabled()   # forward-only kernel; use during inference
                and not self.qk_norm
                and dropout_p == 0.0
            ):
                # ── Path 2: Triton fused sliding-window kernel ──────────
                # Handles causal + window masking in a single fused kernel
                # with online softmax — no mask allocation, O(W·L) compute.
                # Only used during inference (no autograd backward).
                sm_scale = 1.0 / math.sqrt(Dh)
                out = _triton_sliding_window_attention(
                    q, k_attn, v_attn,
                    window_size=self.window_size,
                    sm_scale=sm_scale,
                )
            else:
                # ── Path 3: Fallback — additive mask SDPA ───────────────
                attn_mask = self._get_causal_window_mask(L, x.device)
                # Cast mask to match Q dtype (bf16/fp16 under DeepSpeed/AMP)
                attn_mask = attn_mask.to(q.dtype)
                try:
                    out = F.scaled_dot_product_attention(
                        q, k_attn, v_attn,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
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
