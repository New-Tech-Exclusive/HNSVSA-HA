"""
System 2 — SHADE: Sparse Hyperedge Attractor with Distributed Encoding
========================================================================

ARNL V1.1 — replaces the V1.0 Hyper-Adjacency Map.

SHADE is a non-neural, self-indexing hyperedge attractor graph.
Nodes store concept embeddings in frozen SALT space; retrieval uses
nearest-neighbor search over those embeddings.  Injection is direct
logit boosting — no projection matrix, no hidden-state modification.

Node Structure:
    N = { concept_emb, target_dist, context_window, confidence_vec,
          S_base, S_overflow }

Edge Structure (organic co-occurrence):
    E = { node_a_id, node_b_id, co_weight, last_co_step }

Retrieval Pipeline:
    1. Context fingerprinting (average-pool last K SAC tokens)
    2. Nearest-neighbor candidate search (top-M by cosine sim)
    3. Scoring with edge boost
    4. Direct logit injection
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch

from arnl.config import ARNLConfig


# ────────────────────────────────────────────────────────────────
# SHADE Node — a self-contained belief unit
# ────────────────────────────────────────────────────────────────

@dataclass
class ShadeNode:
    """A single concept-attractor node in SHADE.

    Each node stores a concept embedding, a small target distribution
    over vocabulary tokens, and strength/decay counters.
    """
    node_id: int
    concept_emb: np.ndarray              # int8 quantised, shape (d_semantic,)
    target_dist: Dict[int, float]        # token_id → probability weight (top-K)
    context_window: List[int]            # recent SAC token IDs at creation/reinforcement
    confidence_vec: np.ndarray           # int8, shape (len(context_window),)
    s_base: int = 1
    s_overflow: int = 0
    s_base_original: int = 1
    miss_count: int = 0
    hard_locked: bool = False
    tier_label: str = "domain"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0

    # ── Derived properties ──

    @property
    def tier(self) -> str:
        if self.s_base < 50:
            return "incubation"
        elif self.s_base < 1000:
            return "axiomatic"
        else:
            return "overflow"

    @property
    def alpha(self) -> float:
        """α(S_base) = σ((S_base − 50) / 10)"""
        return 1.0 / (1.0 + math.exp(-(self.s_base - 50) / 10))

    @property
    def alpha_effective(self) -> float:
        """Logic Duel suppression: α_eff = α × (S_base / S_base_original)"""
        if self.s_base_original == 0:
            return 0.0
        return self.alpha * (self.s_base / self.s_base_original)

    @property
    def primary_target_token_id(self) -> int:
        """The highest-weight token in target_dist."""
        if not self.target_dist:
            return -1
        return max(self.target_dist, key=self.target_dist.get)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "concept_emb": self.concept_emb.tolist(),
            "target_dist": {str(k): v for k, v in self.target_dist.items()},
            "context_window": self.context_window,
            "confidence_vec": self.confidence_vec.tolist(),
            "s_base": self.s_base,
            "s_overflow": self.s_overflow,
            "s_base_original": self.s_base_original,
            "miss_count": self.miss_count,
            "hard_locked": self.hard_locked,
            "tier_label": self.tier_label,
            "tier": self.tier,
            "alpha": round(self.alpha, 6),
            "alpha_effective": round(self.alpha_effective, 6),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "hit_count": self.hit_count,
            "primary_target": self.primary_target_token_id,
        }


# ────────────────────────────────────────────────────────────────
# SHADE Edge — organic co-occurrence link
# ────────────────────────────────────────────────────────────────

@dataclass
class ShadeEdge:
    """Co-occurrence link between two SHADE nodes."""
    node_a_id: int
    node_b_id: int
    co_weight: int = 1           # [0, 1000], created at 1
    last_co_step: int = 0        # generation step of last co-activation
    single_miss_counter: int = 0 # counts single-activations without co-activation

    @property
    def edge_key(self) -> Tuple[int, int]:
        return (min(self.node_a_id, self.node_b_id),
                max(self.node_a_id, self.node_b_id))


# ────────────────────────────────────────────────────────────────
# SHADE — The full System 2 graph
# ────────────────────────────────────────────────────────────────

class SHADE:
    """System 2 — Sparse Hyperedge Attractor with Distributed Encoding.

    Non-neural, no gradients.  Self-indexing via frozen SALT embeddings.
    Supports nearest-neighbor retrieval, direct logit injection, organic
    edge formation, and full belief-state export for auditability.
    """

    def __init__(self, config: ARNLConfig):
        self.config = config
        self._nodes: Dict[int, ShadeNode] = {}
        self._edges: Dict[Tuple[int, int], ShadeEdge] = {}
        self._next_id: int = 0
        self._global_step: int = 0

        # Retrieval parameters
        self._top_m = config.shade_top_m
        self._top_k = config.shade_top_k
        self._context_k = config.shade_context_k
        self._sim_min = config.shade_sim_min

        # Frozen SALT embedding table — set via load_embeddings()
        self._emb_table: Optional[np.ndarray] = None  # (vocab_size, d_semantic), float32
        self._emb_norms: Optional[np.ndarray] = None   # (vocab_size,), for fast cosine

        # Node embedding matrix for batch cosine search
        self._node_embs: Optional[np.ndarray] = None   # (N, d_semantic), int8
        self._node_ids_index: List[int] = []            # parallel to _node_embs rows

        # Diagnostics
        self.total_retrievals: int = 0
        self.total_hits: int = 0
        self.total_abstentions: int = 0
        self.semantic_vacuum_events: int = 0
        self._recently_active: List[int] = []  # node_ids active in current step

    # ── Embedding Setup ─────────────────────────────────────────

    def load_embeddings(self, embedding_weight: torch.Tensor):
        """Load the frozen SALT token embedding table for fingerprinting.

        Parameters
        ----------
        embedding_weight : Tensor[vocab_size, d_model]
            The token embedding weight from System 1.  Will be stored
            as float32 numpy for fast cosine operations.
        """
        w = embedding_weight.detach().cpu().float().numpy()
        norms = np.linalg.norm(w, axis=1, keepdims=True).clip(min=1e-8)
        self._emb_table = w / norms  # L2-normalised
        self._emb_norms = norms.squeeze()

    def _rebuild_index(self):
        """Rebuild the flat embedding index from current nodes."""
        if not self._nodes:
            self._node_embs = None
            self._node_ids_index = []
            return
        ids = list(self._nodes.keys())
        embs = np.stack([self._nodes[i].concept_emb.astype(np.float32) for i in ids])
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        self._node_embs = embs / norms
        self._node_ids_index = ids

    # ── Context Fingerprinting (Step 1) ─────────────────────────

    def fingerprint(self, sac_token_ids: List[int]) -> Optional[np.ndarray]:
        """Create a context query vector from the last K SAC tokens.

        Returns None if the embedding table is not loaded or no SAC tokens.
        """
        if self._emb_table is None or not sac_token_ids:
            return None

        k = min(self._context_k, len(sac_token_ids))
        recent = sac_token_ids[-k:]

        # Average-pool the L2-normalised embeddings
        vecs = self._emb_table[recent]  # (k, d)
        q = vecs.mean(axis=0)
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return None
        return q / norm

    # ── Nearest-Neighbor Retrieval (Steps 2–3) ──────────────────

    def retrieve(
        self,
        sac_token_ids: List[int],
        step: Optional[int] = None,
    ) -> Optional[ShadeNode]:
        """Full SHADE retrieval pipeline: fingerprint → search → score → return.

        Parameters
        ----------
        sac_token_ids : list[int]
            Recent SAC (Semantic Anchor Candidate) token IDs from context.
        step : int, optional
            Current generation step (for edge recency weighting).

        Returns
        -------
        ShadeNode or None
            The best-matching node, or None if no candidate exceeds sim_min.
        """
        self.total_retrievals += 1
        if step is not None:
            self._global_step = step

        q = self.fingerprint(sac_token_ids)
        if q is None or self._node_embs is None or len(self._node_ids_index) == 0:
            self.total_abstentions += 1
            return None

        # Cosine similarity against all nodes
        sims = self._node_embs @ q  # (N,)
        top_indices = np.argsort(-sims)[:self._top_m]

        best_node = None
        best_score = -1.0

        for idx in top_indices:
            node_id = self._node_ids_index[idx]
            sim = float(sims[idx])

            if sim < self._sim_min:
                break  # sorted, so remaining are worse

            node = self._nodes[node_id]

            # Confidence vector dot product with context
            conf_score = self._confidence_score(node, sac_token_ids)

            # Strength factor
            strength = math.log1p(node.s_base)

            # Edge boost
            eb = self._edge_boost(node_id)

            score = sim * conf_score * strength * eb

            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None:
            self.total_abstentions += 1
            self.semantic_vacuum_events += 1
            return None

        best_node.last_accessed = time.time()
        self.total_hits += 1
        self._recently_active.append(best_node.node_id)
        return best_node

    def _confidence_score(self, node: ShadeNode, sac_tokens: List[int]) -> float:
        """Compute dot(q_ctx, confidence_vec) contribution."""
        if len(node.context_window) == 0:
            return 1.0
        overlap = 0.0
        for i, ctx_tok in enumerate(node.context_window):
            if ctx_tok in sac_tokens:
                if i < len(node.confidence_vec):
                    overlap += max(0, node.confidence_vec[i] / 127.0)
                else:
                    overlap += 0.5
        return max(1.0, 1.0 + overlap)

    def _edge_boost(self, node_id: int) -> float:
        """1.0 + (sum of co_weights to recently active nodes) / 1000."""
        boost = 0.0
        for other_id in self._recently_active:
            if other_id == node_id:
                continue
            key = (min(node_id, other_id), max(node_id, other_id))
            edge = self._edges.get(key)
            if edge is not None:
                boost += edge.co_weight
        return 1.0 + boost / 1000.0

    # ── Direct Logit Injection (Step 4) ─────────────────────────

    def inject_logits(
        self,
        logits: torch.Tensor,
        node: ShadeNode,
        is_function_word: bool = False,
    ) -> torch.Tensor:
        """Apply direct logit boosting from a SHADE node.

        logit'(t) = logit(t) + α · target_dist[t]

        Parameters
        ----------
        logits : Tensor[V] or Tensor[1, V]
            Pre-softmax logit vector from System 1.
        node : ShadeNode
        is_function_word : bool
            If True, suppress injection by function_word_suppression factor.

        Returns
        -------
        Modified logits (same shape as input).
        """
        alpha = node.alpha_effective
        if is_function_word:
            alpha *= self.config.function_word_suppression

        squeeze = logits.dim() == 1
        if squeeze:
            logits = logits.unsqueeze(0)

        for tok_id, weight in node.target_dist.items():
            if 0 <= tok_id < logits.size(-1):
                logits[0, tok_id] = logits[0, tok_id] + alpha * weight

        if squeeze:
            logits = logits.squeeze(0)
        return logits

    # ── Node Management ─────────────────────────────────────────

    def insert_node(
        self,
        concept_emb: np.ndarray,
        target_dist: Dict[int, float],
        context_window: List[int],
        s_base: int = 1,
        tier_label: str = "domain",
    ) -> ShadeNode:
        """Create and insert a new SHADE node."""
        node_id = self._next_id
        self._next_id += 1

        confidence_vec = np.ones(len(context_window), dtype=np.int8) * 64

        node = ShadeNode(
            node_id=node_id,
            concept_emb=concept_emb.astype(np.int8) if concept_emb.dtype != np.int8 else concept_emb,
            target_dist=dict(target_dist),
            context_window=list(context_window),
            confidence_vec=confidence_vec,
            s_base=s_base,
            s_base_original=max(s_base, 1),
            tier_label=tier_label,
        )
        self._nodes[node_id] = node
        self._rebuild_index()
        return node

    def delete_node(self, node_id: int) -> Optional[ShadeNode]:
        """Hard Delete — remove a node and all its edges."""
        node = self._nodes.pop(node_id, None)
        if node is not None:
            # Remove all edges involving this node
            to_remove = [k for k in self._edges if node_id in k]
            for k in to_remove:
                del self._edges[k]
            self._rebuild_index()
        return node

    def hard_lock(self, node_id: int) -> bool:
        """Hard Lock — S_base=1000, decay disabled."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.s_base = 1000
            node.s_base_original = 1000
            node.hard_locked = True
            return True
        return False

    def hard_override(self, node_id: int) -> Optional[ShadeNode]:
        """Hard Override — S_base=0, schedule for deletion."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.s_base = 0
            node.hard_locked = False
            return node
        return None

    # ── Edge Management ─────────────────────────────────────────

    def add_or_strengthen_edge(self, node_a_id: int, node_b_id: int, step: int = 0):
        """Create or strengthen a co-occurrence edge between two nodes."""
        key = (min(node_a_id, node_b_id), max(node_a_id, node_b_id))
        edge = self._edges.get(key)
        if edge is None:
            edge = ShadeEdge(node_a_id=key[0], node_b_id=key[1],
                             co_weight=1, last_co_step=step)
            self._edges[key] = edge
        else:
            edge.co_weight = min(edge.co_weight + 1, 1000)
            edge.last_co_step = step
            edge.single_miss_counter = 0

    def decay_edge(self, key: Tuple[int, int]):
        """Decay an edge by the miss rule: -1 per 10 single-activations."""
        edge = self._edges.get(key)
        if edge is None:
            return
        edge.single_miss_counter += 1
        if edge.single_miss_counter >= 10:
            edge.co_weight -= 1
            edge.single_miss_counter = 0
        if edge.co_weight <= 0:
            del self._edges[key]

    def step_edges(self, active_node_ids: List[int], step: int):
        """Post-generation edge update: strengthen co-active, decay others."""
        # Strengthen edges between all co-active pairs
        active_set = set(active_node_ids)
        for a in active_node_ids:
            for b in active_node_ids:
                if a < b:
                    self.add_or_strengthen_edge(a, b, step)

        # Decay edges where only one end was active
        for key, edge in list(self._edges.items()):
            a_active = edge.node_a_id in active_set
            b_active = edge.node_b_id in active_set
            if (a_active and not b_active) or (b_active and not a_active):
                self.decay_edge(key)

    def begin_step(self):
        """Reset per-step tracking. Call at start of each generation step."""
        self._recently_active = []

    # ── Bulk Operations ─────────────────────────────────────────

    def get_all_axioms(self, min_sbase: int = 50) -> List[ShadeNode]:
        return [n for n in self._nodes.values() if n.s_base >= min_sbase]

    def get_all_nodes(self) -> List[ShadeNode]:
        return list(self._nodes.values())

    def purge_dead(self) -> int:
        dead = [nid for nid, n in self._nodes.items() if n.s_base <= 0]
        for nid in dead:
            self.delete_node(nid)
        return len(dead)

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def diagnostics(self) -> dict:
        return {
            "total_nodes": self.size,
            "total_edges": self.edge_count,
            "total_retrievals": self.total_retrievals,
            "total_hits": self.total_hits,
            "total_abstentions": self.total_abstentions,
            "hit_rate": self.total_hits / max(1, self.total_retrievals),
            "semantic_vacuum_events": self.semantic_vacuum_events,
        }

    # ── Conflict Detection ──────────────────────────────────────

    def conflict_scan(
        self,
        new_concept_emb: np.ndarray,
        delta: float = 0.25,
    ) -> List[Tuple[ShadeNode, float]]:
        """Scan axiomatic nodes for conflicts with a new concept embedding.

        Returns list of (node, conflict_score) where conflict_score < -delta.
        """
        conflicts = []
        new_norm = np.linalg.norm(new_concept_emb).clip(min=1e-8)
        new_unit = new_concept_emb.astype(np.float32) / new_norm

        for node in self.get_all_axioms():
            old_unit = node.concept_emb.astype(np.float32)
            old_norm = np.linalg.norm(old_unit).clip(min=1e-8)
            old_unit = old_unit / old_norm
            sim = float(np.dot(new_unit, old_unit))
            if sim < -delta:
                conflicts.append((node, sim))
        return conflicts

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str):
        state = {
            "version": "1.1",
            "next_id": self._next_id,
            "global_step": self._global_step,
            "nodes": {str(nid): n.to_dict() for nid, n in self._nodes.items()},
            "edges": [
                {"a": e.node_a_id, "b": e.node_b_id,
                 "co_weight": e.co_weight, "last_co_step": e.last_co_step}
                for e in self._edges.values()
            ],
            "diagnostics": self.diagnostics,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            state = json.load(f)

        self._next_id = state.get("next_id", 0)
        self._global_step = state.get("global_step", 0)

        for _nid_str, rec in state.get("nodes", {}).items():
            node = ShadeNode(
                node_id=rec["node_id"],
                concept_emb=np.array(rec["concept_emb"], dtype=np.int8),
                target_dist={int(k): v for k, v in rec["target_dist"].items()},
                context_window=rec["context_window"],
                confidence_vec=np.array(rec["confidence_vec"], dtype=np.int8),
                s_base=rec["s_base"],
                s_overflow=rec["s_overflow"],
                s_base_original=rec["s_base_original"],
                miss_count=rec["miss_count"],
                hard_locked=rec["hard_locked"],
                tier_label=rec["tier_label"],
                created_at=rec.get("created_at", 0),
                last_accessed=rec.get("last_accessed", 0),
                hit_count=rec.get("hit_count", 0),
            )
            self._nodes[node.node_id] = node

        for erec in state.get("edges", []):
            key = (min(erec["a"], erec["b"]), max(erec["a"], erec["b"]))
            self._edges[key] = ShadeEdge(
                node_a_id=key[0], node_b_id=key[1],
                co_weight=erec["co_weight"],
                last_co_step=erec.get("last_co_step", 0),
            )

        self._rebuild_index()
        print(f"  SHADE loaded: {self.size} nodes, {self.edge_count} edges")

    # ── Belief State Export ─────────────────────────────────────

    def export_belief_state(self) -> List[dict]:
        return [n.to_dict() for n in self._nodes.values()]

    def export_belief_state_json(self) -> str:
        return json.dumps(self.export_belief_state(), indent=2)

    def __repr__(self) -> str:
        return (
            f"SHADE(nodes={self.size}, edges={self.edge_count}, "
            f"axioms={len(self.get_all_axioms())}, "
            f"hit_rate={self.diagnostics['hit_rate']:.2%})"
        )
