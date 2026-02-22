"""
Decay Engine — Memory Lifecycle Manager
========================================

Rule-based arithmetic engine governing the survival of Hyperedges.
No trainable parameters — purely deterministic counter logic.

Tiers:
    Incubation (S_base  1–49):  Zero decay.  +1 on hit.  Promotes at 50.
    Axiomatic  (S_base 50–1000): 1-in-5 decay (−1 per 5 misses).
    Overflow   (S_overflow 0–S_max): 5-20 rule (+20 hit, −20 miss).

Overflow Ceiling:
    S_overflow = min(S_overflow + 20, S_max)    (S_max default = 2000)
    Prevents frequency-based dominance over axiomatic hierarchy.
"""

from __future__ import annotations

from typing import Optional, List

from arnl.config import ARNLConfig
from arnl.system2 import Hyperedge, HyperAdjacencyMap


class DecayEngine:
    """Deterministic memory lifecycle manager for Hyperedges.

    Called after every token generation step to update the strength
    counters of all relevant Hyperedges.

    Parameters
    ----------
    config : ARNLConfig
    """

    def __init__(self, config: ARNLConfig):
        self.config = config
        self.s_max = config.s_max
        self.miss_interval = config.miss_decay_interval
        self.overflow_hit = config.overflow_hit_delta
        self.overflow_miss = config.overflow_miss_delta

    # ── Post-Generation Update ──────────────────────────────────

    def update(
        self,
        edge: Hyperedge,
        is_hit: bool,
        hyper_map: HyperAdjacencyMap,
    ) -> str:
        """Apply the post-generation reinforcement rules.

        Parameters
        ----------
        edge : Hyperedge
            The Hyperedge that was retrieved this step.
        is_hit : bool
            True if the generated token matches v_target.
        hyper_map : HyperAdjacencyMap
            The map (needed for deletion logging).

        Returns
        -------
        status : str
            One of 'hit', 'miss', 'promoted', 'deleted'.
        """
        if edge.hard_locked:
            # Hard-locked edges are immune to all decay
            if is_hit:
                edge.hit_count += 1
                edge.s_overflow = min(edge.s_overflow + self.overflow_hit, self.s_max)
            return "hit" if is_hit else "miss"

        if is_hit:
            return self._process_hit(edge)
        else:
            return self._process_miss(edge, hyper_map)

    # ── HIT Logic ───────────────────────────────────────────────

    def _process_hit(self, edge: Hyperedge) -> str:
        """Token matched v_target → reinforce."""
        edge.hit_count += 1
        edge.miss_count = 0  # Reset miss counter

        if edge.s_base < 50:
            # Tier 1 — Incubation: +1 per hit
            edge.s_base += 1
            if edge.s_base >= 50:
                edge.s_base_original = max(edge.s_base_original, edge.s_base)
                return "promoted"
        elif edge.s_base < 1000:
            # Tier 2 — Axiomatic: +1 per hit (strengthens toward ceiling)
            edge.s_base = min(edge.s_base + 1, 1000)
            edge.s_base_original = max(edge.s_base_original, edge.s_base)
        else:
            # Tier 3 — Overflow: +20 on hit, capped at S_max
            edge.s_overflow = min(edge.s_overflow + self.overflow_hit, self.s_max)

        return "hit"

    # ── MISS Logic ──────────────────────────────────────────────

    def _process_miss(self, edge: Hyperedge, hyper_map: HyperAdjacencyMap) -> str:
        """Context was present but generated token ≠ v_target → decay."""
        edge.miss_count += 1

        if edge.s_base < 50:
            # Tier 1 — Incubation: ZERO decay (protected from noise)
            pass
        elif edge.s_base <= 1000:
            # Tier 2 — Axiomatic: 1-in-5 rule
            if edge.miss_count % self.miss_interval == 0:
                edge.s_base -= 1
        # Regardless of tier, overflow decays on miss
        if edge.s_overflow > 0:
            edge.s_overflow = max(0, edge.s_overflow - self.overflow_miss)

        # Deletion check: S_base reached 0
        if edge.s_base <= 0:
            hyper_map.delete(edge.key)
            return "deleted"

        return "miss"

    # ── Inversion Decay (called by Reasoning Head conflict scan) ─

    def apply_inversion_penalty(
        self,
        edge: Hyperedge,
        conflict_score: float,
        hyper_map: HyperAdjacencyMap,
    ) -> str:
        """Apply inversion decay from a genuine semantic contradiction.

        Inversion Decay Rule (with Dead Zone):
            If conflict < −δ:   S_base_old = S_base_old × (1 + conflict)
            If −δ ≤ conflict ≤ 0:  No penalty (dead zone)
            δ = 0.25

        Parameters
        ----------
        edge : Hyperedge
        conflict_score : float
            cos_sim(v_new, v_old) — negative values indicate contradiction.
        hyper_map : HyperAdjacencyMap

        Returns
        -------
        status : str
        """
        delta = self.config.delta

        if edge.hard_locked:
            return "locked"

        if conflict_score >= -delta:
            # Dead zone — geometric noise, not genuine contradiction
            return "dead_zone"

        # Genuine inversion: Conflict < -δ
        # S_base_old = S_base_old × (1 + conflict)
        # Since conflict < -δ < 0, the multiplier (1 + conflict) < 1 ⇒ erosion
        multiplier = 1.0 + conflict_score
        edge.s_base = max(0, int(edge.s_base * multiplier))

        if edge.s_base <= 0:
            hyper_map.delete(edge.key)
            return "refuted"

        return "eroded"

    # ── Batch Processing ────────────────────────────────────────

    def batch_update(
        self,
        edges_and_hits: List[tuple[Hyperedge, bool]],
        hyper_map: HyperAdjacencyMap,
    ) -> dict:
        """Process multiple edge updates in a batch.

        Returns a summary dict with counts per status.
        """
        summary = {"hit": 0, "miss": 0, "promoted": 0, "deleted": 0}
        for edge, is_hit in edges_and_hits:
            status = self.update(edge, is_hit, hyper_map)
            summary[status] = summary.get(status, 0) + 1
        return summary
