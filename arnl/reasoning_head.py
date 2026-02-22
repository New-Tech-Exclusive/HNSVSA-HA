"""
Reasoning Head — Control Logic & Synchroniser
==============================================

A non-generative controller that orchestrates the interaction between
System 1 and System 2.  It performs four sequential operations at each
token step:

    Phase 1 — Semantic Saliency Filter
        Projects C into R^d, identifies C_salient (semantic anchors).

    Phase 2 — Tiered Consistency Gate (τ)
        Validates novel hyperedge candidates against existing Axioms.

    Phase 3 — Active Conflict & Inversion Detection
        Detects contradictions; applies inversion decay with dead-zone.

    Phase 4 — Injection Coordination
        Computes Sigmoid-Ramped Influence Scaler α and dispatches
        injection to the W_proj bridge.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arnl.config import ARNLConfig
from arnl.utils import RMSNorm, SwiGLUFFN, make_causal_mask
from arnl.system2 import Hyperedge, HyperAdjacencyMap
from arnl.decay_engine import DecayEngine


# ────────────────────────────────────────────────────────────────
# Data containers
# ────────────────────────────────────────────────────────────────

class SaliencyResult(NamedTuple):
    """Output of the Semantic Saliency Filter."""
    anchor_ids: List[int]           # Top k semantic anchor token IDs
    anchor_mask: torch.Tensor       # Boolean mask over C
    is_vacuum: bool                 # True if fewer than min_k anchors found


class InjectionPlan(NamedTuple):
    """Output of the full Reasoning Head pipeline for one token step."""
    v_target: Optional[torch.Tensor]   # Target embedding (d_semantic) or None
    alpha: float                       # Injection strength [0, 1]
    edge: Optional[Hyperedge]          # Retrieved hyperedge or None
    is_function_word: bool             # Whether current position is a function word
    is_vacuum: bool                    # Whether this was a semantic vacuum event
    conflict_actions: List[str]        # Conflict scan results


# ────────────────────────────────────────────────────────────────
# Context Encoder — compact transformer for the Reasoning Head
# ────────────────────────────────────────────────────────────────

class _ReasoningAttnBlock(nn.Module):
    """Lightweight self-attention block for reasoning context encoding."""

    def __init__(self, d: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, d_ffn, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=mask, is_causal=(mask is None))
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class ContextEncoder(nn.Module):
    """Small causal transformer that builds contextualised representations
    in the frozen semantic space for saliency scoring and classification."""

    def __init__(self, config: ARNLConfig):
        super().__init__()
        d = config.d_reasoning
        self.proj_in = nn.Linear(config.d_semantic, d, bias=False)
        self.layers = nn.ModuleList([
            _ReasoningAttnBlock(
                d=d,
                n_heads=config.n_reasoning_heads,
                d_ffn=config.d_reasoning_ffn,
                dropout=config.dropout,
            )
            for _ in range(config.n_reasoning_layers)
        ])
        self.proj_out = nn.Linear(d, config.d_semantic, bias=False)
        self.norm_out = RMSNorm(config.d_semantic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[B, T, d_semantic]
            Frozen semantic embeddings of context tokens.

        Returns
        -------
        Tensor[B, T, d_semantic]
            Contextualised representations.
        """
        B, T, _ = x.shape
        h = self.proj_in(x)
        mask = make_causal_mask(T, x.device)
        for layer in self.layers:
            h = layer(h, mask=mask)
        return self.norm_out(self.proj_out(h))


# ────────────────────────────────────────────────────────────────
# Reasoning Head (full controller)
# ────────────────────────────────────────────────────────────────

class ReasoningHead(nn.Module):
    """System 3 controller that coordinates System 1 and System 2.

    Contains:
        - Frozen semantic embedding space (R^d_semantic, L2-normalised)
        - Context encoder (compact causal transformer)
        - Saliency scoring head
        - Function-word classifier
        - Syntactic centroid vectors (for filtering)
        - All rule-based gating, conflict, and α logic

    Parameters
    ----------
    config : ARNLConfig
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.config = config
        d_sem = config.d_semantic

        # ── Frozen semantic embeddings ──
        # In production, initialise from a pretrained embedding model.
        self.semantic_embed = nn.Embedding(config.vocab_size, d_sem)
        # Frozen — no gradients
        self.semantic_embed.weight.requires_grad = False

        # ── Syntactic centroid vectors ──
        # Learnable centroids representing function-word clusters
        # (articles, prepositions, conjunctions, determiners, aux verbs)
        n_centroids = 32  # number of syntactic cluster centres
        self.syntactic_centroids = nn.Parameter(
            torch.randn(n_centroids, d_sem) * 0.1,
            requires_grad=False,
        )

        # ── Context encoder ──
        self.context_encoder = ContextEncoder(config)

        # ── Saliency scoring MLP ──
        self.saliency_head = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.SiLU(),
            nn.Linear(d_sem // 2, 1),
        )

        # ── Function-word classifier ──
        # Predicts whether the current generation position expects a function word
        self.function_word_head = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.SiLU(),
            nn.Linear(d_sem // 2, 1),
            nn.Sigmoid(),
        )

        # ── Tier classifier (for new hyperedge insertion) ──
        self.tier_classifier = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.SiLU(),
            nn.Linear(d_sem // 2, 3),  # axiom / domain / user
        )

        self._tier_labels = ["axiom", "domain", "user"]

    @torch.no_grad()
    def _embed_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Project token IDs into the frozen hyperspherical space.

        Returns L2-normalised embeddings.
        """
        emb = self.semantic_embed(token_ids)
        return F.normalize(emb, dim=-1)

    # ════════════════════════════════════════════════════════════
    # Phase 1 — Semantic Saliency Filter
    # ════════════════════════════════════════════════════════════

    def saliency_filter(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[SaliencyResult, torch.Tensor]:
        """Identify the top-k semantically dense tokens in C.

        Steps:
            1. Project all tokens into frozen R^d.
            2. Compute norms (semantic density proxy).
            3. Filter tokens close to syntactic centroids.
            4. Pick top k remaining tokens as C_salient.

        Parameters
        ----------
        token_ids : LongTensor[T]  (single example, no batch dim)

        Returns
        -------
        result : SaliencyResult
        sem_embeddings : Tensor[T, d_semantic]
        """
        T = token_ids.size(0)
        k = self.config.k_anchors
        threshold = self.config.syntactic_filter_cosine

        # 1. Frozen semantic embeddings (L2-normalised)
        sem = self._embed_semantic(token_ids)  # (T, d_sem)

        # 2. Semantic density = vector norm (before normalisation)
        raw_emb = self.semantic_embed(token_ids)
        norms = raw_emb.norm(dim=-1)  # (T,)

        # 3. Filter syntactic centroids
        # cos_sim of each token with each centroid ⇒ (T, n_centroids)
        centroid_norm = F.normalize(self.syntactic_centroids, dim=-1)
        sims = sem @ centroid_norm.T  # (T, n_centroids)
        max_sim, _ = sims.max(dim=-1)  # (T,)
        is_syntactic = max_sim > (1.0 - threshold)  # close to centroid ⇒ syntactic

        # 4. Score non-syntactic tokens by saliency head
        contextualised = self.context_encoder(sem.unsqueeze(0)).squeeze(0)  # (T, d_sem)
        saliency_scores = self.saliency_head(contextualised).squeeze(-1)  # (T,)
        saliency_scores = saliency_scores + norms  # combine with norm

        # Mask out syntactic tokens
        saliency_scores[is_syntactic] = -float("inf")

        # Select top k
        actual_k = min(k, (~is_syntactic).sum().item())
        if actual_k < self.config.semantic_vacuum_min_anchors:
            return SaliencyResult(
                anchor_ids=[],
                anchor_mask=torch.zeros(T, dtype=torch.bool, device=token_ids.device),
                is_vacuum=True,
            ), contextualised

        _, topk_idx = saliency_scores.topk(actual_k)
        anchor_mask = torch.zeros(T, dtype=torch.bool, device=token_ids.device)
        anchor_mask[topk_idx] = True

        anchor_token_ids = token_ids[topk_idx].cpu().tolist()

        return SaliencyResult(
            anchor_ids=anchor_token_ids,
            anchor_mask=anchor_mask,
            is_vacuum=False,
        ), contextualised

    # ════════════════════════════════════════════════════════════
    # Phase 2 — Tiered Consistency Gate (τ)
    # ════════════════════════════════════════════════════════════

    def consistency_gate(
        self,
        v_new: torch.Tensor,
        hyper_map: HyperAdjacencyMap,
        tier_label: str = "domain",
    ) -> Tuple[bool, float, int]:
        """Validate a novel hyperedge candidate against existing Axioms.

        Gate(H_new) = σ( Sim(v_new, v_axioms) − τ_tiered )

        Parameters
        ----------
        v_new : Tensor[d_semantic]
        hyper_map : HyperAdjacencyMap
        tier_label : str

        Returns
        -------
        accepted : bool
        gate_score : float
        init_sbase : int
            Starting S_base (1 for normal, 25 for paraphrase match).
        """
        # Pick threshold
        if tier_label == "axiom":
            tau = self.config.tau_axiom
        elif tier_label == "domain":
            tau = self.config.tau_domain
        else:
            tau = self.config.tau_user

        axioms = hyper_map.get_all_axioms(min_sbase=50)
        if not axioms:
            # No axioms yet — accept freely
            return True, 1.0, 1

        # Similarity to all existing axioms
        axiom_vectors = torch.stack([a.v_target for a in axioms]).to(v_new.device)
        axiom_vectors = F.normalize(axiom_vectors, dim=-1)
        v_new_norm = F.normalize(v_new.unsqueeze(0), dim=-1)

        sims = (v_new_norm @ axiom_vectors.T).squeeze(0)  # (n_axioms,)
        max_sim = sims.max().item()

        # Gate score: σ(max_sim − τ)
        gate_score = 1.0 / (1.0 + math.exp(-(max_sim - tau)))

        accepted = gate_score > 0.5

        # Paraphrase pass: if rejected but within radius of a Tier 2 Axiom
        # and not contradicting it, start at S_base=25
        init_sbase = 1
        if not accepted:
            # Check if it's a near-paraphrase of an existing axiom
            close_mask = sims > (1.0 - self.config.paraphrase_cosine_radius)
            non_contradicting = sims > 0
            paraphrase = close_mask & non_contradicting
            if paraphrase.any():
                accepted = True
                init_sbase = self.config.paraphrase_init_sbase

        return accepted, gate_score, init_sbase

    # ════════════════════════════════════════════════════════════
    # Phase 3 — Active Conflict & Inversion Detection
    # ════════════════════════════════════════════════════════════

    def conflict_scan(
        self,
        v_target: torch.Tensor,
        hyper_map: HyperAdjacencyMap,
        decay_engine: DecayEngine,
    ) -> List[str]:
        """Scan v_target against all existing Axioms for contradictions.

        Conflict(v_new, v_old) = cos_sim(v_new, v_old)

        Dead Zone: −δ ≤ conflict ≤ 0  → no action (geometric noise)
        Inversion: conflict < −δ     → S_base_old × (1 + conflict)

        Parameters
        ----------
        v_target : Tensor[d_semantic]
        hyper_map : HyperAdjacencyMap
        decay_engine : DecayEngine

        Returns
        -------
        actions : list[str]
            One entry per axiom that was penalised ('eroded', 'refuted', etc.)
        """
        axioms = hyper_map.get_all_axioms(min_sbase=50)
        if not axioms:
            return []

        actions = []
        v = F.normalize(v_target.unsqueeze(0), dim=-1)

        for axiom in axioms:
            v_old = F.normalize(axiom.v_target.unsqueeze(0).to(v.device), dim=-1)
            conflict = (v @ v_old.T).item()

            if conflict < -self.config.delta:
                status = decay_engine.apply_inversion_penalty(axiom, conflict, hyper_map)
                actions.append(f"key={axiom.key} conflict={conflict:.3f} → {status}")

        return actions

    # ════════════════════════════════════════════════════════════
    # Phase 4 — Injection Coordination
    # ════════════════════════════════════════════════════════════

    def compute_alpha(self, edge: Hyperedge) -> float:
        """Sigmoid-Ramped Influence Scaler.

            α(S_base) = σ((S_base − 50) / 10)

        With Logic Duel suppression applied.
        """
        return edge.alpha_effective

    def is_function_word_position(
        self,
        contextualised: torch.Tensor,
    ) -> bool:
        """Classify whether the last position expects a function word.

        Uses the contextualised representation of the last token.

        Parameters
        ----------
        contextualised : Tensor[T, d_semantic]
        """
        last = contextualised[-1:]  # (1, d_sem)
        prob = self.function_word_head(last).item()
        return prob > 0.5

    def classify_tier(self, contextualised: torch.Tensor) -> str:
        """Classify the epistemic tier of a new candidate.

        Returns one of 'axiom', 'domain', 'user'.
        """
        last = contextualised[-1:]
        logits = self.tier_classifier(last)  # (1, 3)
        idx = logits.argmax(dim=-1).item()
        return self._tier_labels[idx]

    # ════════════════════════════════════════════════════════════
    # Full Pipeline — called by Arnold at each generation step
    # ════════════════════════════════════════════════════════════

    def forward(
        self,
        context_ids: torch.Tensor,
        hyper_map: HyperAdjacencyMap,
        decay_engine: DecayEngine,
    ) -> InjectionPlan:
        """Execute the full 4-phase Reasoning Head pipeline.

        Parameters
        ----------
        context_ids : LongTensor[T]
            Current context tokens (1-D, single example).
        hyper_map : HyperAdjacencyMap
        decay_engine : DecayEngine

        Returns
        -------
        InjectionPlan
        """
        # Phase 1 — Saliency Filter
        saliency, contextualised = self.saliency_filter(context_ids)

        if saliency.is_vacuum:
            return InjectionPlan(
                v_target=None,
                alpha=0.0,
                edge=None,
                is_function_word=False,
                is_vacuum=True,
                conflict_actions=[],
            )

        # Phase 2+3 — Retrieval + Conflict Scan
        edge = hyper_map.lookup(saliency.anchor_ids)

        if edge is None:
            return InjectionPlan(
                v_target=None,
                alpha=0.0,
                edge=None,
                is_function_word=self.is_function_word_position(contextualised),
                is_vacuum=False,
                conflict_actions=[],
            )

        # Phase 3 — Conflict scan
        conflict_actions = self.conflict_scan(
            edge.v_target.to(context_ids.device),
            hyper_map,
            decay_engine,
        )

        # Phase 4 — α computation
        alpha = self.compute_alpha(edge)

        # Function word suppression
        is_fw = self.is_function_word_position(contextualised)
        if is_fw:
            alpha *= self.config.function_word_suppression

        return InjectionPlan(
            v_target=edge.v_target,
            alpha=alpha,
            edge=edge,
            is_function_word=is_fw,
            is_vacuum=False,
            conflict_actions=conflict_actions,
        )

    # ════════════════════════════════════════════════════════════
    # Knowledge Ingestion — adding new facts to System 2
    # ════════════════════════════════════════════════════════════

    def ingest_fact(
        self,
        context_ids: torch.Tensor,
        target_token_id: int,
        hyper_map: HyperAdjacencyMap,
    ) -> Optional[Hyperedge]:
        """Attempt to add a new factual pathway to System 2.

        Runs saliency filter, consistency gate, and tier classification
        before inserting into the map.

        Parameters
        ----------
        context_ids : LongTensor[T]
        target_token_id : int
        hyper_map : HyperAdjacencyMap

        Returns
        -------
        Hyperedge if accepted, None if rejected.
        """
        saliency, contextualised = self.saliency_filter(context_ids)

        if saliency.is_vacuum:
            return None

        # Target embedding
        v_target = self._embed_semantic(
            torch.tensor([target_token_id], device=context_ids.device)
        ).squeeze(0)

        # Tier classification
        tier_label = self.classify_tier(contextualised)

        # Consistency gate
        accepted, gate_score, init_sbase = self.consistency_gate(
            v_target, hyper_map, tier_label
        )

        if not accepted:
            return None

        # Insert
        edge = hyper_map.insert(
            anchor_ids=saliency.anchor_ids,
            v_target=v_target,
            target_token_id=target_token_id,
            s_base=init_sbase,
            tier_label=tier_label,
        )
        return edge

    def load_frozen_embeddings(self, weight: torch.Tensor):
        """Load pretrained embeddings into the frozen semantic space.

        Parameters
        ----------
        weight : Tensor[vocab_size, d_semantic]
        """
        assert weight.shape == self.semantic_embed.weight.shape, (
            f"Expected {self.semantic_embed.weight.shape}, got {weight.shape}"
        )
        self.semantic_embed.weight.copy_(weight)

    def load_syntactic_centroids(self, centroids: torch.Tensor):
        """Load pre-computed syntactic centroid vectors.

        Parameters
        ----------
        centroids : Tensor[n_centroids, d_semantic]
        """
        assert centroids.shape[-1] == self.config.d_semantic
        self.syntactic_centroids.copy_(centroids)
