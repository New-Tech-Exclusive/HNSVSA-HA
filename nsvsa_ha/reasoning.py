"""
PonderNet-style Adaptive Reasoning for NSVSA-HA.

Adds variable-depth computation to the hybrid architecture by looping
the top R layers of the model multiple times with learned halting.

Key idea:  Arnold's Soft-VSA state is *already* recurrent — each
reasoning pass through the top layers refines the compressed global
state, giving the model more "thinking time" on hard problems while
spending fewer steps on easy ones.

Architecture:
    input → [base layers 0..B-1] → [reasoning layers B..L-1 × N steps] → LM head

    Each reasoning step:
      1. Add a step embedding so the model knows which pass it's on
      2. Run through the shared reasoning layers (with VSA state carry-over)
      3. Predict halting probability λ_t via ReasoningController
      4. Accumulate output weighted by effective probability

    Training loss: LM loss on weighted output + λ_ponder * KL(halt ∥ Geometric)

Reference:
    Banino et al., "PonderNet: Learning to Ponder" (2021)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .rmsnorm import RMSNorm


class ReasoningController(nn.Module):
    """
    Predicts per-step halting probability for adaptive computation.

    Takes mean-pooled hidden states → small MLP → sigmoid(·) ∈ (0, 1).
    One scalar per batch element (shared across all positions).
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, L, d] hidden states from reasoning layers.
        Returns:
            [B, 1] halting probability in (0, 1).
        """
        pooled = h.mean(dim=1)                      # [B, d]
        return torch.sigmoid(self.net(pooled))       # [B, 1]


def ponder_loss_fn(
    halt_probs: list[torch.Tensor],
    p_geometric: float = 0.5,
) -> torch.Tensor:
    """
    KL divergence between the learned halting distribution and a
    geometric prior Geom(p_geometric).

    This regularises the model to use a roughly geometric distribution
    of reasoning steps — preventing collapse to always-1 or always-max.

    Args:
        halt_probs:   List of N tensors, each [B, 1], the per-step λ_t.
        p_geometric:  Success probability of the geometric prior.

    Returns:
        Scalar mean KL divergence.
    """
    N = len(halt_probs)
    if N == 0:
        return torch.tensor(0.0, device=halt_probs[0].device if halt_probs else "cpu")

    device = halt_probs[0].device

    # Pre-compute log-prior: log P_geo(n) = (n-1)*log(1-p) + log(p)
    log_1mp = math.log(max(1.0 - p_geometric, 1e-10))
    log_p   = math.log(max(p_geometric, 1e-10))

    cum_remain = torch.ones_like(halt_probs[0])     # [B, 1]
    kl = torch.tensor(0.0, device=device)

    for n in range(N):
        # Effective probability of halting at step n
        if n == N - 1:
            p_n = cum_remain                         # flush remainder
        else:
            p_n = cum_remain * halt_probs[n]

        log_p_n   = torch.log(p_n + 1e-10)
        log_prior = n * log_1mp + log_p

        kl = kl + (p_n * (log_p_n - log_prior)).mean()
        cum_remain = cum_remain * (1.0 - halt_probs[n])

    return kl


class ReasoningBlock(nn.Module):
    """
    Wraps the reasoning (top-R) layers in a PonderNet loop.

    On each iteration:
      1. Add step embedding to hidden states
      2. Run through the reasoning layers (provided as a callable)
      3. Predict halt probability λ_t
      4. Accumulate hidden-state output weighted by effective probability

    The final output is a convex combination of all intermediate hidden
    states, ensuring differentiability and stable gradients.

    Args:
        d_model:        Model dimension.
        max_steps:      Maximum number of reasoning iterations.
        hidden_dim:     ReasoningController MLP hidden size.
        p_geometric:    Geometric prior parameter for ponder loss.
        epsilon:        Halt threshold for early-stop during inference.
    """

    def __init__(
        self,
        d_model: int,
        max_steps: int = 8,
        hidden_dim: int = 256,
        p_geometric: float = 0.5,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.p_geometric = p_geometric
        self.epsilon = epsilon

        self.controller = ReasoningController(d_model, hidden_dim)
        self.step_embed = nn.Embedding(max_steps, d_model)

        # Initialize step embeddings small so first pass ≈ identity
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.01)

    def forward(
        self,
        h: torch.Tensor,
        reasoning_fn,
        **reasoning_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h:               [B, L, d] hidden states from base layers.
            reasoning_fn:    Callable(h, **kwargs) -> (h_out, vsa_states, caches)
                             Runs the reasoning layers on h.
            **reasoning_kwargs: Forwarded to reasoning_fn.

        Returns:
            output:       [B, L, d] weighted combination of all steps.
            ponder_cost:  Scalar KL-divergence regularisation loss.
            mean_steps:   Scalar mean number of steps taken.
        """
        B, L, d = h.shape
        device = h.device

        output = torch.zeros_like(h)
        remainders = torch.ones(B, 1, 1, device=device)
        halt_probs: list[torch.Tensor] = []
        n_updates = torch.zeros(B, device=device)

        # Keep the original input for residual connection across steps
        h_input = h

        for step in range(self.max_steps):
            # Inject step information
            step_emb = self.step_embed.weight[step]            # [d]
            h_stepped = h_input + step_emb                     # [B, L, d]

            # Run through reasoning layers with gradient checkpointing
            # so intermediate activations are freed and recomputed on backward.
            if self.training:
                h_out, vsa_states, caches = grad_checkpoint(
                    reasoning_fn,
                    h_stepped,
                    use_reentrant=False,
                    **reasoning_kwargs,
                )
            else:
                h_out, vsa_states, caches = reasoning_fn(h_stepped, **reasoning_kwargs)

            # Predict halting probability
            lambda_t = self.controller(h_out)                  # [B, 1]
            halt_probs.append(lambda_t)

            # Compute effective weight for this step
            if step == self.max_steps - 1:
                weight = remainders                             # Use all remaining
            else:
                weight = remainders * lambda_t.unsqueeze(-1)    # [B, 1, 1]

            output = output + weight * h_out
            remainders = remainders * (1.0 - lambda_t.unsqueeze(-1))
            n_updates = n_updates + 1

            # Carry forward: use this step's output as next step's input
            h_input = h_out

            # Early stop during inference
            if not self.training and remainders.max().item() < self.epsilon:
                break

        ponder_cost = ponder_loss_fn(halt_probs, self.p_geometric)
        mean_steps = n_updates.float().mean()

        return output, ponder_cost, mean_steps
