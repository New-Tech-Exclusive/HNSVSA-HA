"""
Root Mean Square Layer Normalization

    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = √(mean(x²) + ε)

Compared to LayerNorm:
  - No mean-centering → preserves representation diversity
  - ~15% faster (one fewer reduction operation)
  - Same training stabilization benefits
  - Used by LLaMA, Mistral, Gemma, Griffin

When Triton is available, a fused kernel replaces the three-pass PyTorch
implementation (cast → rms → scale) with a single GPU kernel launch.

Reference:
    Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Triton fused RMSNorm kernel
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
    def _rms_norm_fwd_kernel(
        X, W, Y,
        stride_x,   # stride between rows of X
        N: tl.constexpr,           # feature dimension (d_model)
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program per row: load x[row], compute RMS, multiply by weight, store."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

        # RMS(x) = sqrt(mean(x²) + eps)
        rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        y = (x / rms) * w

        tl.store(Y + row * stride_x + offs, y.to(tl.load(X + row * stride_x + offs, mask=mask, other=0.0).dtype), mask=mask)


def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Launch Triton fused RMSNorm on a flattened view of x."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    nrows, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_fwd_kernel[(nrows,)](
        x_2d, weight, y,
        stride_x=x_2d.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
    )
    return y.reshape(orig_shape)


class RMSNorm(nn.Module):
    """
    Root-mean-square normalization with a learned scale parameter.

    Uses a Triton fused kernel when available (single kernel launch
    instead of 3 separate ops).  Falls back to PyTorch otherwise.

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Epsilon for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton fused kernel during inference on CUDA when available.
        # The Triton kernel is forward-only (no autograd backward), so we
        # restrict it to inference (torch.is_grad_enabled() == False).
        # During training, the PyTorch path provides full autograd support.
        if (
            _HAS_TRITON
            and x.is_cuda
            and not torch.is_grad_enabled()
            and x.shape[-1] == self.d_model
        ):
            try:
                return _triton_rms_norm(x, self.weight, self.eps)
            except Exception:
                pass  # Fall through to PyTorch path

        # PyTorch fallback — compute in float32 for stability, cast back
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x / rms).to(input_dtype)
        return x * self.weight
