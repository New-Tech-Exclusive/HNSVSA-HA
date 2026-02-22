"""
System 2 — The Sparse Hyper-Adjacency Map
==========================================

Role: Reliability Core — stores discrete factual pathways (Hyperedges)
in a content-addressable hash map.  System 2 is **not** a neural network.
It has no trainable parameters and does not participate in gradient
computation.

Hyperedge Structure:
    H = { Hash(C_salient) → v_target,  S_base,  S_overflow }

Tiers:
    Incubation  (S_base   1–49)   — unverified hypothesis, zero decay
    Axiomatic   (S_base  50–1000) — verified factual pathway, 1-in-5 decay
    Overflow    (S_overflow 0→S_max) — contextual salience, 5-20 rule
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch

from arnl.config import ARNLConfig
from arnl.utils import semantic_hash, semantic_hash_combinations


# ────────────────────────────────────────────────────────────────
# Hyperedge — a single factual pathway record
# ────────────────────────────────────────────────────────────────

@dataclass
class Hyperedge:
    """A discrete factual pathway stored in the Hyper-Adjacency Map.

    Attributes
    ----------
    key : int
        Content-addressable hash derived from semantic anchors.
    anchor_ids : list[int]
        The token IDs of the semantic anchors that produced this key.
    v_target : torch.Tensor
        Target token embedding in the frozen hyperspherical space R^d.
    target_token_id : int
        The vocabulary index of the target token.
    s_base : int
        Axiomatic strength counter ∈ [0, 1000].
        Below 50 = Incubation tier; 50–1000 = Axiomatic tier.
    s_overflow : int
        Contextual salience counter ∈ [0, S_max].
        Active only when s_base = 1000.
    s_base_original : int
        The peak S_base value used for Logic Duel suppression.
    miss_count : int
        Running miss counter for the 1-in-5 decay rule.
    hard_locked : bool
        If True, all decay is suppressed (Hard Lock via admin).
    tier_label : str
        Epistemic category: 'axiom', 'domain', or 'user'.
    created_at : float
        Timestamp of creation.
    last_accessed : float
        Timestamp of most recent hit.
    hit_count : int
        Total number of hits (for diagnostics).
    """

    key: int
    anchor_ids: List[int]
    v_target: torch.Tensor
    target_token_id: int
    s_base: int = 1
    s_overflow: int = 0
    s_base_original: int = 1
    miss_count: int = 0
    hard_locked: bool = False
    tier_label: str = "domain"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0

    # ── Derived properties ──────────────────────────────────────

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
        """Sigmoid-Ramped Influence Scaler: α(S_base) = σ((S_base − 50) / 10)"""
        return 1.0 / (1.0 + math.exp(-(self.s_base - 50) / 10))

    @property
    def alpha_effective(self) -> float:
        """Logic Duel suppression: α_eff = α × (S_base_current / S_base_original)"""
        if self.s_base_original == 0:
            return 0.0
        return self.alpha * (self.s_base / self.s_base_original)

    @property
    def total_weight(self) -> float:
        """W_total = W_base + β · ln(1 + S_overflow)  (β=1 by default)."""
        return self.alpha_effective + math.log1p(self.s_overflow)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary (for auditability)."""
        return {
            "key": self.key,
            "anchor_ids": self.anchor_ids,
            "target_token_id": self.target_token_id,
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
        }


# ────────────────────────────────────────────────────────────────
# Hyper-Adjacency Map — the sparse factual hash map
# ────────────────────────────────────────────────────────────────

class HyperAdjacencyMap:
    """System 2 — content-addressable hash map of Hyperedges.

    O(1) lookup by saliency-weighted hash.  Non-neural, no gradients.
    Supports CRUD, Hard Lock/Override, and full belief-state export
    for Granular Semantic Auditability.

    Parameters
    ----------
    config : ARNLConfig
        Architecture configuration.
    """

    def __init__(self, config: ARNLConfig):
        self.config = config
        self._map: Dict[int, Hyperedge] = {}
        # Diagnostic counters
        self.total_lookups: int = 0
        self.total_hits: int = 0
        self.total_misses: int = 0
        self.semantic_vacuum_events: int = 0
        self.refuted_pathways: List[dict] = []

    # ── Core Operations ─────────────────────────────────────────

    def lookup(self, anchor_ids: List[int]) -> Optional[Hyperedge]:
        """O(1) retrieval by semantic anchor hash.

        Tries the full anchor set first, then progressively smaller
        subsets down to pairs for partial-match robustness.
        """
        self.total_lookups += 1

        if len(anchor_ids) < self.config.semantic_vacuum_min_anchors:
            self.semantic_vacuum_events += 1
            return None

        # Try full hash first, then subsets
        keys = semantic_hash_combinations(anchor_ids, min_k=2)
        for k in keys:
            if k in self._map:
                edge = self._map[k]
                edge.last_accessed = time.time()
                self.total_hits += 1
                return edge

        self.total_misses += 1
        return None

    def insert(
        self,
        anchor_ids: List[int],
        v_target: torch.Tensor,
        target_token_id: int,
        s_base: int = 1,
        tier_label: str = "domain",
    ) -> Hyperedge:
        """Insert a new Hyperedge into the map.

        If a hyperedge with the same key already exists, the new one is
        ignored (use ``update`` to modify existing entries).
        """
        key = semantic_hash(anchor_ids)
        if key in self._map:
            return self._map[key]

        edge = Hyperedge(
            key=key,
            anchor_ids=list(anchor_ids),
            v_target=v_target.detach().clone(),
            target_token_id=target_token_id,
            s_base=s_base,
            s_base_original=max(s_base, 1),
            tier_label=tier_label,
        )
        self._map[key] = edge
        return edge

    def delete(self, key: int) -> Optional[Hyperedge]:
        """Hard Delete — permanently remove a hyperedge."""
        edge = self._map.pop(key, None)
        if edge is not None:
            self.refuted_pathways.append(edge.to_dict())
        return edge

    def hard_lock(self, key: int) -> bool:
        """Hard Lock — set S_base=1000, disable all decay."""
        if key in self._map:
            edge = self._map[key]
            edge.s_base = 1000
            edge.s_base_original = 1000
            edge.hard_locked = True
            return True
        return False

    def hard_override(self, key: int) -> Optional[Hyperedge]:
        """Hard Override — set S_base=0, trigger deletion on next step."""
        if key in self._map:
            edge = self._map[key]
            edge.s_base = 0
            edge.hard_locked = False
            return edge
        return None

    # ── Bulk Operations ─────────────────────────────────────────

    def get_all_axioms(self, min_sbase: int = 50) -> List[Hyperedge]:
        """Return all Hyperedges at or above Axiomatic tier."""
        return [e for e in self._map.values() if e.s_base >= min_sbase]

    def get_all_edges(self) -> List[Hyperedge]:
        """Return every Hyperedge in the map."""
        return list(self._map.values())

    def purge_dead(self) -> int:
        """Remove all Hyperedges with S_base ≤ 0.  Returns count purged."""
        dead = [k for k, e in self._map.items() if e.s_base <= 0]
        for k in dead:
            self.delete(k)
        return len(dead)

    # ── Belief State Export (Granular Semantic Auditability) ────

    def export_belief_state(self) -> List[dict]:
        """Export every belief as a human-readable record."""
        return [e.to_dict() for e in self._map.values()]

    def export_belief_state_json(self) -> str:
        return json.dumps(self.export_belief_state(), indent=2)

    @property
    def size(self) -> int:
        return len(self._map)

    @property
    def diagnostics(self) -> dict:
        return {
            "total_edges": self.size,
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.total_hits / max(1, self.total_lookups),
            "semantic_vacuum_events": self.semantic_vacuum_events,
            "refuted_pathways_count": len(self.refuted_pathways),
        }

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str):
        """Serialise the full map to disk."""
        state = {
            "config_version": self.config.version,
            "edges": {},
            "diagnostics": self.diagnostics,
        }
        for key, edge in self._map.items():
            state["edges"][str(key)] = {
                **edge.to_dict(),
                "v_target": edge.v_target.cpu().tolist(),
            }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str, device: torch.device = torch.device("cpu")):
        """Deserialise a saved map from disk."""
        with open(path, "r") as f:
            state = json.load(f)
        for _key_str, rec in state["edges"].items():
            v_target = torch.tensor(rec["v_target"], device=device)
            key = rec["key"]
            edge = Hyperedge(
                key=key,
                anchor_ids=rec["anchor_ids"],
                v_target=v_target,
                target_token_id=rec["target_token_id"],
                s_base=rec["s_base"],
                s_overflow=rec["s_overflow"],
                s_base_original=rec["s_base_original"],
                miss_count=rec["miss_count"],
                hard_locked=rec["hard_locked"],
                tier_label=rec["tier_label"],
                created_at=rec["created_at"],
                last_accessed=rec["last_accessed"],
                hit_count=rec["hit_count"],
            )
            self._map[key] = edge

    def __repr__(self) -> str:
        return (
            f"HyperAdjacencyMap(edges={self.size}, "
            f"axioms={len(self.get_all_axioms())}, "
            f"hit_rate={self.diagnostics['hit_rate']:.2%})"
        )
