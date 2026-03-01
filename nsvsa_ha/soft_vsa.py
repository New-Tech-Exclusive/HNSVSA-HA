"""
Soft VSA - Continuous Weighted Superposition

The original NSVSA-HA spec uses hard sgn() bundling, which forces a STE
workaround for gradients and throws away magnitude information.

This module replaces hard binarization with a continuous representation:
- Bundling: running L2-normalized mean of bound vectors (Fréchet mean on sphere)
- Binding:  element-wise multiplication still works for continuous ±1-ish vectors
- Unbinding: same as binding (still self-inverse when vectors are near {-1,1})

Benefits:
1. Real gradients everywhere - no STE needed for internal operations
2. Better capacity: magnitude encodes confidence/recency weighting
3. Supports exponential decay for recency bias in the global state

The STE is still used once, at the Neural Frontend, to produce near-bipolar
token hypervectors. All subsequent VSA arithmetic stays continuous.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .cache import VSACache


class SoftBundle(nn.Module):
    """
    Continuous bundling via L2-normalized weighted mean.

    Instead of M = sgn(Σ bᵢ), we compute:
        M = Σ wᵢ bᵢ  / ||Σ wᵢ bᵢ||₂

    This lies on the unit hypersphere, preserves real gradients, and
    encodes the superposition of all bound vectors with controllable weights.

    Capacity: because we normalize, adding more vectors gradually moves M
    toward the centroid. The dot-product similarity to any single member
    decays as ~1/√n for uniform weights - same asymptotic as hard bundling,
    but with a smooth gradient landscape.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        vectors: torch.Tensor,           # [..., n, d]
        weights: Optional[torch.Tensor] = None,  # [..., n]  (will be softmax'd if provided)
        dim: int = -2,
    ) -> torch.Tensor:                    # [..., d]
        """
        Args:
            vectors: Tensors to bundle  [..., n, d]
            weights: Optional unnormalized weights [..., n].
                     If None, uniform mean is used.
            dim: Dimension to reduce over.

        Returns:
            Unit-norm bundle vector [..., d]
        """
        if weights is not None:
            # Softmax-normalize weights so they sum to 1
            w = F.softmax(weights, dim=dim)          # [..., n]
            summed = (vectors * w.unsqueeze(-1)).sum(dim=dim)
        else:
            summed = vectors.mean(dim=dim)           # [..., d]

        return F.normalize(summed, p=2, dim=-1, eps=self.eps)


class SoftVSAStateUpdate(nn.Module):
    """
    Continuous VSA state update combining local group bundling
    and global recurrent state maintenance.

    Per-group state:
        G_j = normalize( Σ_{k=1}^K  (x_{j,k} ⊙ p_k) )

    Global recurrent state (online EMA):
        S_t = normalize( α · S_{t-1}  +  (1-α) · (G_j ⊙ P_{macro,j}) )

    α (decay) is a learned scalar in [0, 1], initialized near 0.9.
    This is analogous to the gating mechanism in GRUs / Mamba.
    """

    def __init__(
        self,
        d: int,
        group_size: int,
        max_groups: int,
        init_decay: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d = d
        self.K = group_size
        self.max_groups = max_groups
        self.eps = eps

        self.bundle = SoftBundle(eps=eps)

        # Learned decay gate α ∈ (0,1) per dimension for richer state mixing
        # Initialized to logit(init_decay) so sigmoid gives init_decay
        init_logit = torch.log(torch.tensor(init_decay / (1.0 - init_decay)))
        self.decay_logit = nn.Parameter(init_logit.expand(d).clone())

        # Projection to mix local-attention output into VSA state query
        self.query_proj = nn.Linear(d, d, bias=False)

    @property
    def decay(self) -> torch.Tensor:
        """Per-dimension decay coefficients α ∈ (0, 1)."""
        return torch.sigmoid(self.decay_logit)

    # ------------------------------------------------------------------
    # Core algebraic primitives (continuous, real gradients throughout)
    # ------------------------------------------------------------------

    @staticmethod
    def bind(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Element-wise product – still invertible when ||p||₂ = 1."""
        return x * p

    @staticmethod
    def unbind(bound: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """When p is unit-norm bipolar, p ⊙ p = 1, so unbind == bind."""
        return bound * p

    # ------------------------------------------------------------------
    # Group and global state computation
    # ------------------------------------------------------------------

    def compute_groups(
        self,
        token_vectors: torch.Tensor,   # [B, seq_len, d]
        local_positions: torch.Tensor, # [K, d]  unit-norm
    ) -> torch.Tensor:                  # [B, n_groups, d]
        """Bundle overlapping windows of K tokens into group vectors."""
        B, L, d = token_vectors.shape
        K = self.K
        if L == 0:
            return torch.zeros(B, 0, d, device=token_vectors.device)

        n_full = L // K
        groups = []

        if n_full > 0:
            full = token_vectors[:, : n_full * K, :].reshape(B, n_full, K, d)  # [B, G, K, d]
            pos = local_positions[:K].view(1, 1, K, d)                           # [1, 1, K, d]
            bound = self.bind(full, pos)
            # Equivalent to SoftBundle over dim=2 for each group
            full_groups = F.normalize(bound.mean(dim=2), p=2, dim=-1, eps=self.eps)  # [B, G, d]
            groups.append(full_groups)

        # Partial trailing group
        remainder = L - n_full * K
        if remainder > 0:
            chunk = token_vectors[:, n_full * K :, :]           # [B, r, d]
            pos   = local_positions[:remainder]                  # [r, d]
            bound = self.bind(chunk, pos)
            groups.append(self.bundle(bound, dim=1).unsqueeze(1))

        if not groups:
            return torch.zeros(B, 0, d, device=token_vectors.device)

        return torch.cat(groups, dim=1)                          # [B, n_groups, d]

    def compute_causal_global_states(
        self,
        group_vectors: torch.Tensor,    # [B, n_groups, d]
        macro_positions: torch.Tensor,  # [n_groups, d]
    ) -> torch.Tensor:                  # [B, n_groups, d]
        """
        Causal EMA over bound group vectors.

        S_j = α ⊙ S_{j-1}  +  (1-α) ⊙ normalize(G_j ⊙ P_j)
        then re-normalize so ||S_j||₂ = 1.
        """
        B, n_groups, d = group_vectors.shape
        α = self.decay                                     # [d]
        states = []

        S = torch.zeros(B, d, device=group_vectors.device)

        for j in range(n_groups):
            bound_g = self.bind(group_vectors[:, j], macro_positions[j])  # [B, d]
            # EMA update
            S = α * S + (1.0 - α) * bound_g
            S = F.normalize(S, p=2, dim=-1, eps=self.eps)
            states.append(S)

        return torch.stack(states, dim=1)                  # [B, n_groups, d]

    def generate_query(
        self,
        global_state: torch.Tensor,    # [B, d]
        next_local_pos: torch.Tensor,  # [B, d]  or [d]
    ) -> torch.Tensor:                  # [B, d]
        """Unbind the next local position to retrieve a query vector."""
        q = self.unbind(global_state, next_local_pos)
        return self.query_proj(q)      # learned projection before similarity

    # ------------------------------------------------------------------
    # Incremental single-token update (for cached generation)
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        token_vec: torch.Tensor,        # [B, d]  L2-normalized
        local_positions: torch.Tensor,  # [K, d]
        macro_positions: torch.Tensor,  # [max_groups, d]
        cache: VSACache,
    ) -> Tuple[torch.Tensor, torch.Tensor, VSACache]:
        """
        Process a single new token using cached VSA state.

        Cost: O(d) — no recomputation of past groups.

        The query for this token uses the global state *before* the
        current group is committed (matching the causal semantics of
        the full forward pass).
        """
        l = cache.group_count  # local position within current group

        # Query from previous-group state (current group not yet committed)
        local_pos = local_positions[l]                      # [d]
        query = self.query_proj(
            self.unbind(cache.global_state, local_pos.unsqueeze(0))
        )  # [B, d]

        # Bind token to its local position and accumulate into current group
        bound = self.bind(token_vec, local_pos)             # [B, d]
        new_accum = cache.group_accum + bound
        new_count = l + 1
        new_state = cache.global_state
        new_n_groups = cache.num_completed_groups

        # Check if group is now complete
        if new_count == self.K:
            # Bundle: mean of K bound vectors, then normalize
            group_vec = F.normalize(
                new_accum / self.K, p=2, dim=-1, eps=self.eps
            )
            # Bind group vector to its macro position
            bound_group = self.bind(
                group_vec, macro_positions[new_n_groups]
            )
            # EMA update of global state
            α = self.decay                                  # [d]
            new_state = α * cache.global_state + (1.0 - α) * bound_group
            new_state = F.normalize(new_state, p=2, dim=-1, eps=self.eps)
            # Reset group accumulator
            new_accum = torch.zeros_like(new_accum)
            new_count = 0
            new_n_groups += 1

        new_cache = VSACache(
            global_state=new_state,
            group_accum=new_accum,
            group_count=new_count,
            num_completed_groups=new_n_groups,
        )

        return query.unsqueeze(1), new_state, new_cache     # [B, 1, d], [B, d]

    # ------------------------------------------------------------------
    # Full forward (training / prefill)
    # ------------------------------------------------------------------

    def forward(
        self,
        token_vectors: torch.Tensor,   # [B, L, d]  L2-normalized
        local_positions: torch.Tensor, # [K, d]
        macro_positions: torch.Tensor, # [max_groups, d]
        vsa_cache: Optional[VSACache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[VSACache]]:
        """
        Returns:
            queries:      [B, L, d]  – one query per input position
            final_state:  [B, d]     – state after last group
            new_cache:    VSACache (if use_cache) or None
        """
        # ── Incremental decode path ──────────────────────────────────
        if vsa_cache is not None:
            assert token_vectors.shape[1] == 1, \
                "VSA cache only supports single-token decode"
            return self._forward_step(
                token_vectors[:, 0], local_positions, macro_positions, vsa_cache
            )

        # ── Full forward (training / prefill) ────────────────────────
        B, L, d = token_vectors.shape
        K = self.K

        # Build group vectors and causal global states
        group_vecs = self.compute_groups(token_vectors, local_positions)
        n_groups = group_vecs.shape[1]

        if n_groups == 0:
            empty_q = torch.zeros(B, L, d, device=token_vectors.device)
            empty_s = torch.zeros(B, d, device=token_vectors.device)
            new_cache: Optional[VSACache] = None
            if use_cache:
                new_cache = VSACache(
                    global_state=empty_s.clone(),
                    group_accum=torch.zeros(B, d, device=token_vectors.device),
                    group_count=0,
                    num_completed_groups=0,
                )
            return empty_q, empty_s, new_cache

        macro_pos = macro_positions[:n_groups]
        global_states = self.compute_causal_global_states(group_vecs, macro_pos)
        # global_states: [B, n_groups, d]

        # Assign each token position a query derived from previous-group state
        t_idx = torch.arange(L, device=token_vectors.device)
        group_idx = t_idx // K                                  # [L]
        local_idx = t_idx % K                                   # [L]

        prev_group = group_idx - 1                              # [L]
        prev_clamped = prev_group.clamp(min=0)                  # [L]

        # Gather states for each token position: [B, L, d]
        state = global_states[:, prev_clamped, :]

        # Positions before first group have zero prior state
        no_prior = (prev_group < 0).view(1, L, 1)
        state = state.masked_fill(no_prior, 0.0)

        local_pos = local_positions[local_idx]                  # [L, d]
        queries = self.query_proj(self.unbind(state, local_pos.unsqueeze(0)))
        final_state = global_states[:, -1]                     # [B, d]

        # ── Build cache if requested ─────────────────────────────────
        new_cache = None
        if use_cache:
            n_full = L // K
            remainder = L % K

            # State after last *full* group (not including partial group)
            if n_full > 0:
                cache_state = global_states[:, n_full - 1].clone()
            else:
                cache_state = torch.zeros(B, d, device=token_vectors.device)

            # Reconstruct partial group accumulator
            if remainder > 0:
                partial = token_vectors[:, n_full * K:, :]      # [B, r, d]
                partial_pos = local_positions[:remainder]        # [r, d]
                partial_bound = self.bind(partial, partial_pos)  # [B, r, d]
                cache_accum = partial_bound.sum(dim=1)           # [B, d]
            else:
                cache_accum = torch.zeros(B, d, device=token_vectors.device)

            new_cache = VSACache(
                global_state=cache_state,
                group_accum=cache_accum,
                group_count=remainder,
                num_completed_groups=n_full,
            )

        return queries, final_state, new_cache
