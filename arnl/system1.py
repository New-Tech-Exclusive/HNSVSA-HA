"""
System 1 — The Syntactic Linear-Latent Selective Unit (LLSU)
============================================================

ARNL V1.1 — SHADE Edition

Role: Fluidity Core — models transition probabilities of grammatical
structures without internalising factual content.  Also serves as the
gap detector: when System 1's output entropy over SAC tokens is high,
it signals SHADE to intervene.

Architecture: Gated Linear Attention (GLA) / Selective State-Space
Model (SSM) with data-dependent gating.

Key V1.1 Changes:
    • Injection ports REMOVED. SHADE writes to logits, not hidden state.
    • System 1 is the gap detector — H_SAC > H_gap triggers SHADE.
    • ~95% of total parameter budget (was ~80% in V1.0).

State Transition Formula:
    h_t = A_t ⊙ h_{t-1}  +  B_t ⊙ x_t
    A_t = selective forget gate (data-dependent)
    B_t = input gate (data-dependent)
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from arnl.config import ARNLConfig
from arnl.utils import RMSNorm, SwiGLUFFN


# ────────────────────────────────────────────────────────────────
# Gated Linear Attention Layer (single recurrent core)
# ────────────────────────────────────────────────────────────────

class GLALayer(nn.Module):
    """Multi-head Gated Linear Attention with data-dependent gating.

    Each head maintains an independent recurrent state of dimension
    ``d_state``.  Gates A_t (forget) and B_t (input) are computed from
    the current input, making the layer *selective*.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads

        # Data-dependent gating projections
        self.gate_proj = nn.Linear(d_model, 2 * n_heads * d_state, bias=True)

        # Value projection (input to the recurrence)
        self.value_proj = nn.Linear(d_model, n_heads * d_state, bias=False)

        # Output projection (recurrent state → residual stream)
        self.out_proj = nn.Linear(n_heads * d_state, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight)
        with torch.no_grad():
            # Bias the forget gate toward retaining state at init
            self.gate_proj.bias[:self.n_heads * self.d_state].fill_(1.0)
            self.gate_proj.bias[self.n_heads * self.d_state:].zero_()
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor[B, T, d_model]
        h_prev : Tensor[B, n_heads, d_state] | None

        Returns
        -------
        output : Tensor[B, T, d_model]
        h_last : Tensor[B, n_heads, d_state]
        """
        B, T, _ = x.shape
        device = x.device

        if h_prev is None:
            h_prev = torch.zeros(B, self.n_heads, self.d_state,
                                 device=device, dtype=x.dtype)

        gates = self.gate_proj(x)                          # (B, T, 2*H*S)
        gates = gates.view(B, T, 2, self.n_heads, self.d_state)
        A = torch.sigmoid(gates[:, :, 0])                  # (B, T, H, S)
        Bg = torch.sigmoid(gates[:, :, 1])

        V = self.value_proj(x).view(B, T, self.n_heads, self.d_state)

        # Sequential recurrence
        outputs = []
        h = h_prev
        for t in range(T):
            h = A[:, t] * h + Bg[:, t] * V[:, t]
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)                # (B, T, H, S)
        h_flat = h_seq.reshape(B, T, self.n_heads * self.d_state)
        y = self.dropout(self.out_proj(h_flat))

        return y, h

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token recurrent step (fast inference).

        Parameters
        ----------
        x_t : Tensor[B, d_model]
        h_prev : Tensor[B, n_heads, d_state]
        """
        B = x_t.size(0)

        gates = self.gate_proj(x_t)
        gates = gates.view(B, 2, self.n_heads, self.d_state)
        A_t = torch.sigmoid(gates[:, 0])
        B_t = torch.sigmoid(gates[:, 1])

        V_t = self.value_proj(x_t).view(B, self.n_heads, self.d_state)

        h_new = A_t * h_prev + B_t * V_t
        y_t = self.out_proj(h_new.reshape(B, -1))

        return y_t, h_new


# ────────────────────────────────────────────────────────────────
# LLSU Block — GLA + FFN (V1.1: no injection port)
# ────────────────────────────────────────────────────────────────

class LLSUBlock(nn.Module):
    """A single LLSU layer.

    V1.1 Architecture per block:
        1. RMSNorm → GLA recurrent layer → residual add
        2. RMSNorm → SwiGLU FFN → residual add

    The V1.0 injection port between GLA and FFN is removed.
    SHADE injects at the logit layer after the full forward pass.
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.norm_gla = RMSNorm(config.d_model)
        self.gla = GLALayer(
            d_model=config.d_model,
            d_state=config.d_state,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ffn, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor[B, T, d_model]
        h_prev : recurrent state or None

        Returns
        -------
        x : Tensor[B, T, d_model]
        h : recurrent state
        """
        # GLA with residual
        residual = x
        x = self.norm_gla(x)
        x, h = self.gla(x, h_prev)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = residual + x

        return x, h

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token step for autoregressive inference."""
        residual = x_t
        x_t = self.norm_gla(x_t)
        x_t, h = self.gla.step(x_t, h_prev)
        x_t = residual + x_t

        residual = x_t
        x_t = self.norm_ffn(x_t)
        x_t = self.ffn(x_t)
        x_t = residual + x_t

        return x_t, h


# ────────────────────────────────────────────────────────────────
# LLSU — Full System 1 Stack
# ────────────────────────────────────────────────────────────────

class LLSU(nn.Module):
    """System 1 — complete Syntactic Linear-Latent Selective Unit.

    V1.1: No injection ports.  SHADE injects directly into logits
    after the forward pass.  System 1 also serves as the gap detector
    via entropy over SAC tokens.
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # LLSU blocks (no injection ports in V1.1)
        self.layers = nn.ModuleList([LLSUBlock(config) for _ in range(config.n_layers)])

        # Output
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optional weight tying (embed ↔ lm_head)
        if config.weight_tying:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.token_embed.weight.device

    # ── Full-sequence forward (training) ────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        input_ids : LongTensor[B, T]
        states : list of per-layer recurrent states or None

        Returns
        -------
        logits : Tensor[B, T, vocab_size]
        new_states : list of per-layer recurrent states
        """
        B, T = input_ids.shape
        x = self.drop(self.token_embed(input_ids))

        if states is None:
            states = [None] * len(self.layers)

        new_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, h = layer(x, states[i])
            new_states.append(h)

        logits = self.lm_head(self.final_norm(x))
        return logits, new_states

    # ── Single-token step (autoregressive inference) ────────────

    def step(
        self,
        token_id: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Process a single token for autoregressive generation.

        Parameters
        ----------
        token_id : LongTensor[B] or [B, 1]
        states : per-layer recurrent states

        Returns
        -------
        logits : Tensor[B, vocab_size]
        new_states : updated per-layer states
        """
        if token_id.dim() == 2:
            token_id = token_id.squeeze(1)

        x = self.token_embed(token_id)

        new_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, h = layer.step(x, states[i])
            new_states.append(h)

        logits = self.lm_head(self.final_norm(x))
        return logits, new_states

    def init_states(self, batch_size: int) -> List[torch.Tensor]:
        """Create zero-initialised recurrent states for all layers."""
        return [
            torch.zeros(
                batch_size,
                self.config.n_heads,
                self.config.d_state,
                device=self.device,
                dtype=self.token_embed.weight.dtype,
            )
            for _ in range(len(self.layers))
        ]
