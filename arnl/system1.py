"""
System 1 — The Syntactic Linear-Latent Selective Unit (LLSU)
============================================================

Role: Fluidity Core — models transition probabilities of grammatical
structures without internalising factual content.

Architecture: Gated Linear Attention (GLA) / Selective State-Space
Model (SSM) with data-dependent gating.

Key Properties:
    • Linear complexity in sequence length — no quadratic memory cost.
    • Recurrent inference — processes tokens as a state machine.
    • Fact-blind by training (EMLM) — cannot independently recall entities.
    • Every block contains a Latent Summation Gate (injection port)
      positioned between the GLA output and the output FFN.

State Transition Formula:
    h_t = A_t ⊙ h_{t-1}  +  B_t ⊙ x_t
    A_t = selective forget gate (data-dependent)
    B_t = input gate (data-dependent)
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Parameters
    ----------
    d_model : int
        Model hidden dimension.
    d_state : int
        Per-head recurrent state width.
    n_heads : int
        Number of independent GLA heads.
    dropout : float
        Dropout rate on the output projection.
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
        # Produces A (forget) and B (input) gates — both sigmoid-activated
        self.gate_proj = nn.Linear(d_model, 2 * n_heads * d_state, bias=True)

        # Value projection (input to the recurrence)
        self.value_proj = nn.Linear(d_model, n_heads * d_state, bias=False)

        # Output projection (recurrent state → residual stream)
        self.out_proj = nn.Linear(n_heads * d_state, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight)
        # Bias the forget gate toward retaining state at init
        with torch.no_grad():
            # First half of bias (A gate) → positive ⇒ σ > 0.5 ⇒ retain
            self.gate_proj.bias[:self.n_heads * self.d_state].fill_(1.0)
            # Second half (B gate) → near zero
            self.gate_proj.bias[self.n_heads * self.d_state:].zero_()
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    # ── sequential (inference) mode ─────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor[B, T, d_model]
            Input token representations.
        h_prev : Tensor[B, n_heads, d_state] | None
            Previous recurrent state.  Zeros if None.

        Returns
        -------
        output : Tensor[B, T, d_model]
        h_last : Tensor[B, n_heads, d_state]
        """
        B, T, _ = x.shape
        device = x.device

        if h_prev is None:
            h_prev = torch.zeros(B, self.n_heads, self.d_state, device=device, dtype=x.dtype)

        # Pre-compute all projections for the sequence
        gates = self.gate_proj(x)                         # (B, T, 2*n_heads*d_state)
        gates = gates.view(B, T, 2, self.n_heads, self.d_state)
        A = torch.sigmoid(gates[:, :, 0])                 # (B, T, n_heads, d_state)
        Bg = torch.sigmoid(gates[:, :, 1])                # (B, T, n_heads, d_state)

        V = self.value_proj(x).view(B, T, self.n_heads, self.d_state)  # (B, T, H, S)

        # Sequential recurrence:  h_t = A_t ⊙ h_{t-1}  +  B_t ⊙ V_t
        outputs = []
        h = h_prev
        for t in range(T):
            h = A[:, t] * h + Bg[:, t] * V[:, t]         # (B, H, S)
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)               # (B, T, H, S)
        h_flat = h_seq.reshape(B, T, self.n_heads * self.d_state)
        y = self.dropout(self.out_proj(h_flat))            # (B, T, d_model)

        return y, h  # h is the last state

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

        Returns
        -------
        y_t : Tensor[B, d_model]
        h_new : Tensor[B, n_heads, d_state]
        """
        B = x_t.size(0)

        gates = self.gate_proj(x_t)                       # (B, 2*H*S)
        gates = gates.view(B, 2, self.n_heads, self.d_state)
        A_t = torch.sigmoid(gates[:, 0])                  # (B, H, S)
        B_t = torch.sigmoid(gates[:, 1])

        V_t = self.value_proj(x_t).view(B, self.n_heads, self.d_state)

        h_new = A_t * h_prev + B_t * V_t                  # (B, H, S)
        y_t = self.out_proj(h_new.reshape(B, -1))         # (B, d_model)

        return y_t, h_new


# ────────────────────────────────────────────────────────────────
# LLSU Block — GLA + Injection Port + FFN
# ────────────────────────────────────────────────────────────────

class LLSUBlock(nn.Module):
    """A single LLSU layer.

    Architecture per block:
        1. RMSNorm → GLA recurrent layer → residual add
        2. **Injection Port** (additive — biases h_t without overwriting)
        3. RMSNorm → SwiGLU FFN → residual add

    The injection port is the landing surface for W_proj injections from
    the Reasoning Head.  It sits between the recurrent output and the
    FFN so that System 1's FFN processes the biased state and produces a
    token that is both factually guided *and* syntactically coherent.
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

        # Flag: whether this block is an active injection target
        self.injection_enabled: bool = False

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        injection: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor[B, T, d_model]
        h_prev : recurrent state or None
        injection : Tensor[B, T, d_model] | None
            α · W_proj · v_target projected into System 1's space.

        Returns
        -------
        x : Tensor[B, T, d_model]
        h : recurrent state
        """
        # ── GLA with residual ──
        residual = x
        x = self.norm_gla(x)
        x, h = self.gla(x, h_prev)
        x = residual + x

        # ── Injection Port ──
        # Additive: h_t' = h_t + injection
        # Positioned between GLA output and FFN (per spec Section II.4)
        if injection is not None and self.injection_enabled:
            x = x + injection

        # ── FFN with residual ──
        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = residual + x

        return x, h

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        injection_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token step for autoregressive inference.

        Parameters
        ----------
        x_t : Tensor[B, d_model]
        h_prev : Tensor[B, n_heads, d_state]
        injection_t : Tensor[B, d_model] | None
        """
        residual = x_t
        x_t = self.norm_gla(x_t)
        x_t, h = self.gla.step(x_t, h_prev)
        x_t = residual + x_t

        if injection_t is not None and self.injection_enabled:
            x_t = x_t + injection_t

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

    Stacks ``n_layers`` LLSU blocks, adds token + (optional) position
    embeddings, and provides a language-modelling head.

    The model is *fact-blind by training* — during EMLM pretraining all
    factual entities are replaced by generic placeholder tokens so that
    weights optimise entirely for syntactic transitions.
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.config = config

        # ── Embeddings ──
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # ── LLSU blocks ──
        self.layers = nn.ModuleList([LLSUBlock(config) for _ in range(config.n_layers)])

        # Enable injection on designated layers (default: last only)
        injection_layers = config.injection_layers
        if injection_layers is None:
            injection_layers = [config.n_layers - 1]
        for idx in injection_layers:
            self.layers[idx].injection_enabled = True
        self._injection_layer_ids = set(injection_layers)

        # ── Output ──
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

    # ── Full-sequence forward (teacher-forced training) ─────────

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        injections: Optional[dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        input_ids : LongTensor[B, T]
        states : list of per-layer recurrent states or None
        injections : dict mapping layer index → Tensor[B, T, d_model]
            Pre-computed injection vectors for designated layers.

        Returns
        -------
        logits : Tensor[B, T, vocab_size]
        new_states : list of per-layer recurrent states
        """
        B, T = input_ids.shape
        x = self.drop(self.token_embed(input_ids))

        if states is None:
            states = [None] * len(self.layers)
        if injections is None:
            injections = {}

        new_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            inj = injections.get(i, None)
            x, h = layer(x, states[i], injection=inj)
            new_states.append(h)

        logits = self.lm_head(self.final_norm(x))
        return logits, new_states

    # ── Single-token step (autoregressive inference) ────────────

    def step(
        self,
        token_id: torch.Tensor,
        states: List[torch.Tensor],
        injection_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Process a single token for autoregressive generation.

        Parameters
        ----------
        token_id : LongTensor[B] or [B, 1]
        states : per-layer recurrent states
        injection_t : Tensor[B, d_model] | None

        Returns
        -------
        logits : Tensor[B, vocab_size]
        new_states : updated per-layer states
        """
        if token_id.dim() == 2:
            token_id = token_id.squeeze(1)

        x = self.token_embed(token_id)  # (B, d_model)

        new_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            inj = injection_t if i in self._injection_layer_ids else None
            x, h = layer.step(x, states[i], injection_t=inj)
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
