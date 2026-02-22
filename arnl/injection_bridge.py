"""
Injection Bridge — W_proj
=========================

The learned linear projection that translates vectors from System 2's
frozen semantic space R^(d_semantic) into System 1's residual stream
space R^(d_model).

    h_t' = h_t  +  α · W_proj · v_target

W_proj ∈ R^(d_model × d_semantic)

Training procedure (3-phase):
    Phase 1: EMLM pretraining of System 1 (W_proj not involved)
    Phase 2: W_proj + lightweight LoRA adapters on System 1
             (System 1 core frozen)
    Phase 3: W_proj + LoRA only (end-to-end fine-tuning)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arnl.config import ARNLConfig


class InjectionBridge(nn.Module):
    """W_proj — the semantic-to-syntactic translator.

    A single learned linear projection that maps from the Reasoning Head's
    frozen hyperspherical space R^(d_semantic) to System 1's internal
    residual stream space R^(d_model).

    Parameters
    ----------
    config : ARNLConfig
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.config = config

        # W_proj ∈ R^(d_model × d_semantic)
        self.proj = nn.Linear(config.d_semantic, config.d_model, bias=False)

        # Layer norm on the projected output for training stability
        self.proj_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        v_target: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the injection vector.

        Parameters
        ----------
        v_target : Tensor[B, d_semantic] or Tensor[B, T, d_semantic]
            Target embedding from System 2, in the frozen semantic space.
        alpha : Tensor[B] or Tensor[B, T] or scalar
            Sigmoid-Ramped Influence Scaler.

        Returns
        -------
        injection : Tensor matching v_target's leading dims but last dim = d_model
            The vector α · W_proj · v_target to be added to System 1's h_t.
        """
        projected = self.proj_norm(self.proj(v_target))  # (..., d_model)

        # Broadcast α
        if isinstance(alpha, (int, float)):
            return alpha * projected

        if alpha.dim() == 1 and projected.dim() == 2:
            # (B,) → (B, 1)
            alpha = alpha.unsqueeze(-1)
        elif alpha.dim() == 1 and projected.dim() == 3:
            # (B,) → (B, 1, 1)
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        elif alpha.dim() == 2 and projected.dim() == 3:
            # (B, T) → (B, T, 1)
            alpha = alpha.unsqueeze(-1)

        return alpha * projected

    def project_only(self, v_target: torch.Tensor) -> torch.Tensor:
        """Project without scaling (for alignment loss computation)."""
        return self.proj_norm(self.proj(v_target))
