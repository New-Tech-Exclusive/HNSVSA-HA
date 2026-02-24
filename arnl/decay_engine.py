"""
Decay Engine — Memory Lifecycle Manager
========================================

ARNL V1.1 — SHADE Edition

Rule-based arithmetic engine governing the survival of SHADE nodes
and edges.  No trainable parameters — purely deterministic counter logic.

Node Tiers:
    Incubation (S_base  1–49):  Zero decay.  +1 on hit.  Promotes at 50.
    Axiomatic  (S_base 50–1000): 1-in-5 decay (−1 per 5 misses).
    Overflow   (S_overflow 0–S_max): Injection value weighting:
        High-value hit (+20), Low-value hit (+5), Miss (−20).

Edge Decay (V1.1):
    co_weight +1 per co-activation (capped at 1000).
    co_weight -1 per 10 single-activations without co-activation.
    Edge deleted at co_weight = 0.
"""

from __future__ import annotations

from typing import List, Optional

from arnl.config import ARNLConfig


class DecayEngine:
    """Deterministic memory lifecycle manager for SHADE nodes and edges.

    Called after every token generation step to update strength counters.

    Parameters
    ----------
    config : ARNLConfig
    shade : SHADE instance (passed at construction)
    """

    def __init__(self, config: ARNLConfig, shade=None):
        self.config = config
        self.shade = shade
        self.s_max = config.s_max
        self.miss_interval = config.miss_decay_interval
        self.overflow_hit = config.overflow_hit_delta
        self.overflow_miss = config.overflow_miss_delta

        # Diagnostics
        self.total_hits = 0
        self.total_misses = 0
        self.total_promoted = 0
        self.total_deleted = 0

    # ── Post-Generation Update ──────────────────────────────────

    def update_node(self, node, is_hit: bool, injection_value: str = "low") -> str:
        """Apply the post-generation reinforcement rules to a SHADE node.

        Parameters
        ----------
        node : ShadeNode
        is_hit : bool
            True if the generated token matches target_dist.
        injection_value : str
            'high' — SHADE injected a token System 1 ranked ≥20
            'low'  — SHADE injected a token System 1 ranked top-3
            'miss' — generated token not in target_dist

        Returns
        -------
        status : str
            One of 'hit', 'miss', 'promoted', 'deleted'.
        """
        if node.hard_locked:
            if is_hit:
                node.hit_count += 1
                if injection_value == "high":
                    node.s_overflow = min(node.s_overflow + self.overflow_hit, self.s_max)
                else:
                    node.s_overflow = min(node.s_overflow + 5, self.s_max)
            return "hit" if is_hit else "miss"

        if is_hit:
            return self._process_hit(node, injection_value)
        else:
            return self._process_miss(node)

    # ── HIT Logic ───────────────────────────────────────────────

    def _process_hit(self, node, injection_value: str = "low") -> str:
        """Token matched target_dist → reinforce."""
        node.hit_count += 1
        self.total_hits += 1

        if node.s_base < 50:
            # Tier 1 — Incubation: +1 per hit
            node.s_base += 1
            if node.s_base >= 50:
                node.s_base_original = max(node.s_base_original, node.s_base)
                self.total_promoted += 1
                return "promoted"
        elif node.s_base < 1000:
            # Tier 2 — Axiomatic: +1 per hit
            node.s_base = min(node.s_base + 1, 1000)
            node.s_base_original = max(node.s_base_original, node.s_base)
        else:
            # Tier 3 — Overflow: injection value weighting (V1.1)
            if injection_value == "high":
                node.s_overflow = min(node.s_overflow + self.overflow_hit, self.s_max)
            else:
                node.s_overflow = min(node.s_overflow + 5, self.s_max)

        return "hit"

    # ── MISS Logic ──────────────────────────────────────────────

    def _process_miss(self, node) -> str:
        """Context was present but generated token not in target_dist → decay."""
        node.miss_count += 1
        self.total_misses += 1

        if node.s_base < 50:
            # Tier 1 — Incubation: ZERO decay
            pass
        elif node.s_base <= 1000:
            # Tier 2 — Axiomatic: 1-in-5 rule
            if node.miss_count % self.miss_interval == 0:
                node.s_base -= 1

        # Overflow decays on miss
        if node.s_overflow > 0:
            node.s_overflow = max(0, node.s_overflow - self.overflow_miss)

        # Deletion check: S_base reached 0
        if node.s_base <= 0 and self.shade is not None:
            self.shade.delete_node(node.node_id)
            self.total_deleted += 1
            return "deleted"

        return "miss"

    # ── Inversion Decay ─────────────────────────────────────────

    def apply_inversion_penalty(self, node, conflict_score: float) -> str:
        """Apply inversion decay from a semantic contradiction.

        Inversion Rule (with Dead Zone):
            If conflict < −δ:   S_base = S_base × (1 + conflict)
            If −δ ≤ conflict ≤ 0:  No penalty (dead zone)

        Parameters
        ----------
        node : ShadeNode
        conflict_score : float
            cos_sim(v_new, v_old) — negative = contradiction.
        """
        delta = self.config.delta

        if node.hard_locked:
            return "locked"

        if conflict_score >= -delta:
            return "dead_zone"

        # Genuine inversion
        multiplier = 1.0 + conflict_score
        node.s_base = max(0, int(node.s_base * multiplier))

        if node.s_base <= 0 and self.shade is not None:
            self.shade.delete_node(node.node_id)
            return "refuted"

        return "eroded"

    # ── Edge Decay (V1.1) ───────────────────────────────────────

    def decay_edges(self, active_node_ids: List[int], step: int):
        """Post-generation edge lifecycle.

        Strengthens edges between co-active nodes; decays edges where
        only one endpoint was active.

        This is called by Arnold.generate() but can also be called
        externally for batch processing.
        """
        if self.shade is None:
            return
        self.shade.step_edges(active_node_ids, step)

    # ── Batch Processing ────────────────────────────────────────

    def purge_dead_nodes(self) -> int:
        """Remove all nodes with S_base ≤ 0."""
        if self.shade is None:
            return 0
        count = self.shade.purge_dead()
        self.total_deleted += count
        return count

    @property
    def diagnostics(self) -> dict:
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_promoted": self.total_promoted,
            "total_deleted": self.total_deleted,
        }
