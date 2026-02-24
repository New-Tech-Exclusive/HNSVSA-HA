"""
Arnold — ARNL V1.1 Full Model
==============================

SHADE Edition — the main model class that coordinates System 1 (LLSU)
and System 2 (SHADE).

V1.1 Changes:
    • Reasoning Head and W_proj Injection Bridge REMOVED.
    • Gap detection via System 1's H_SAC entropy.
    • SHADE provides direct logit injection (no hidden-state modification).
    • 2-phase training (EMLM + LoRA fine-tuning).
    • 10-step generative loop per the V1.1 spec (Section IV).
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from arnl.config import ARNLConfig
from arnl.system1 import LLSU
from arnl.system2 import SHADE
from arnl.decay_engine import DecayEngine
from arnl.utils import apply_lora


# ────────────────────────────────────────────────────────────────
# Arnold — the unified ARNL V1.1 model
# ────────────────────────────────────────────────────────────────

class Arnold(nn.Module):
    """ARNL V1.1 dual-process model.

    Components:
        system1 : LLSU — the fact-blind recurrent backbone (~95% params)
        shade   : SHADE — non-neural self-indexing belief store (off-budget)
        decay   : DecayEngine — rule-based strength lifecycle manager

    Generative loop (Section IV of spec):
        1. SALT encode → 2. System 1 forward → 3. Gap detection (H_SAC) →
        4–5. SHADE retrieval → 6. Direct logit injection → 7. Sample →
        8–9. Post-gen reinforcement + Decay → 10. Append and loop
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.config = config

        # ── System 1 (neural, trainable) ──
        self.system1 = LLSU(config)

        # ── System 2 (non-neural, no gradients) ──
        self.shade = SHADE(config)

        # ── Decay Engine (rule-based) ──
        self.decay = DecayEngine(config, self.shade)

        # ── Gap detection ──
        self.h_gap = config.h_gap

        # ── SAC token mask (set by load_salt_classes) ──
        self._sac_mask: Optional[torch.Tensor] = None   # bool[vocab_size]
        self._syn_mask: Optional[torch.Tensor] = None   # bool[vocab_size]

        # ── Continuation state for multi-token injection ──
        self._continuation_node = None
        self._continuation_tokens_remaining: List[int] = []

    @property
    def device(self) -> torch.device:
        """Return device of the first parameter (mirrors HuggingFace convention)."""
        return next(self.parameters()).device

    # ── Setup Hooks ─────────────────────────────────────────────

    def load_salt_classes(self, token_classes: torch.Tensor):
        """Load SALT token classifications for gap detection.

        Parameters
        ----------
        token_classes : LongTensor[vocab_size]
            0 = SYN, 1 = SAC, 2 = AMB
        """
        self._sac_mask = (token_classes == 1)
        self._syn_mask = (token_classes == 0)

    def init_shade(self):
        """Give SHADE access to System 1's frozen embedding table."""
        self.shade.load_embeddings(self.system1.token_embed.weight)

    # ── Training Phase Setup ────────────────────────────────────

    def setup_phase1(self):
        """Phase 1 — EMLM Pretraining: train all of System 1."""
        for p in self.system1.parameters():
            p.requires_grad = True
        print("  Phase 1: System 1 fully trainable")

    def setup_phase2(self):
        """Phase 2 — LoRA Fine-Tuning on unmasked domain data.

        Freeze System 1 core, add LoRA adapters (r=16).
        SHADE should be pre-populated before this phase.
        """
        # Freeze System 1 core
        for p in self.system1.parameters():
            p.requires_grad = False

        # Apply LoRA adapters to output projections
        apply_lora(self.system1,
                   rank=self.config.lora_rank,
                   alpha=self.config.lora_alpha,
                   dropout=self.config.lora_dropout)

        n_lora = sum(p.numel() for p in self.system1.parameters() if p.requires_grad)
        print(f"  Phase 2: LoRA adapters active — {n_lora:,} trainable params")

    # ── Forward (teacher-forced training) ───────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Standard teacher-forced forward pass (no SHADE at train time).

        Parameters
        ----------
        input_ids : LongTensor[B, T]
        states : per-layer recurrent states or None

        Returns
        -------
        logits : Tensor[B, T, vocab_size]
        new_states : list of per-layer recurrent states
        """
        return self.system1(input_ids, states)

    # ── Gap Detection ───────────────────────────────────────────

    def compute_h_sac(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy over SAC-class tokens.

        Parameters
        ----------
        logits : Tensor[V] or Tensor[1, V]
            Pre-softmax logit vector from System 1.

        Returns
        -------
        H_SAC in nats.
        """
        if logits.dim() == 2:
            logits = logits[0]

        if self._sac_mask is not None:
            sac_mask = self._sac_mask.to(logits.device)
            sac_logits = logits[sac_mask]
        else:
            # Fallback: treat all tokens as SAC
            sac_logits = logits

        if sac_logits.numel() == 0:
            return 0.0

        probs = F.softmax(sac_logits, dim=-1)
        # Clamp to avoid log(0)
        probs = probs.clamp(min=1e-10)
        h = -(probs * probs.log()).sum().item()
        return h

    def is_function_word(self, token_id: int) -> bool:
        """Check if a token is classified as SYN (function word)."""
        if self._syn_mask is None:
            return False
        if 0 <= token_id < self._syn_mask.size(0):
            return bool(self._syn_mask[token_id].item())
        return False

    # ── Autoregressive Generation (V1.1 10-step loop) ──────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        sac_token_ids_init: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Generate tokens using the full V1.1 generative loop.

        Steps (per spec Section IV):
            1. Input ingestion (SALT encode happens before this call)
            2. System 1 forward pass
            3. Gap detection (H_SAC)
            4. SHADE context fingerprinting (if gap)
            5. NN retrieval + conflict scan (if gap)
            6. Direct logit injection (if SHADE hit)
            7. Token generation (sample)
            8. Post-generation reinforcement
            9. Decay engine pass
           10. Autoregressive append

        Parameters
        ----------
        input_ids : LongTensor[1, T]
            Prompt token IDs.
        max_new_tokens : int
        temperature : float
        top_k : int
        top_p : float
        eos_token_id : int, optional
        sac_token_ids_init : list[int], optional
            Initial SAC tokens from prompt for SHADE fingerprinting.

        Returns
        -------
        LongTensor[1, T + generated] — full sequence including prompt.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        device = input_ids.device
        B = input_ids.size(0)
        assert B == 1, "generate() only supports batch_size=1"

        # ── Step 1: Process prompt ──
        states = self.system1.init_states(B)
        logits, states = self.system1(input_ids, states)
        # logits: (1, T, V) — we only need the last position
        last_logits = logits[0, -1]  # (V,)

        # Track SAC tokens in context for SHADE fingerprinting
        sac_tokens: List[int] = []
        if sac_token_ids_init:
            sac_tokens = list(sac_token_ids_init)
        elif self._sac_mask is not None:
            # Extract SAC tokens from prompt
            for tid in input_ids[0].tolist():
                if 0 <= tid < self._sac_mask.size(0) and self._sac_mask[tid]:
                    sac_tokens.append(tid)

        generated = input_ids.clone()
        self._continuation_node = None
        self._continuation_tokens_remaining = []

        for step_i in range(max_new_tokens):
            self.shade.begin_step()

            # ── Step 3: Gap Detection ──
            shade_node = None
            use_shade = False

            if self._continuation_node is not None:
                # Multi-token continuation: bypass gap detection
                shade_node = self._continuation_node
                use_shade = True
            else:
                h_sac = self.compute_h_sac(last_logits)
                if h_sac > self.h_gap and self.shade.size > 0:
                    # ── Steps 4–5: SHADE Retrieval ──
                    shade_node = self.shade.retrieve(sac_tokens, step=step_i)
                    if shade_node is not None:
                        use_shade = True

            # ── Step 6: Direct Logit Injection ──
            if use_shade and shade_node is not None:
                top1_id = int(last_logits.argmax().item())
                is_fw = self.is_function_word(top1_id)
                last_logits = self.shade.inject_logits(last_logits, shade_node, is_function_word=is_fw)

            # ── Step 7: Token Generation ──
            next_token = self._sample(last_logits, temperature, top_k, top_p)
            next_id = next_token.item()

            # ── Step 8: Post-generation reinforcement ──
            if use_shade and shade_node is not None:
                self._post_gen_reinforce(shade_node, next_id, last_logits, step_i)

                # Multi-token continuation check
                if next_id in shade_node.target_dist:
                    remaining = [t for t in shade_node.target_dist
                                 if t != next_id and shade_node.target_dist[t] > 0]
                    if remaining:
                        self._continuation_node = shade_node
                        self._continuation_tokens_remaining = remaining
                    else:
                        self._continuation_node = None
                        self._continuation_tokens_remaining = []
                else:
                    self._continuation_node = None
                    self._continuation_tokens_remaining = []

            # ── Step 9: Decay Engine pass ──
            active_ids = list(self.shade._recently_active)
            if active_ids:
                self.shade.step_edges(active_ids, step_i)

            # ── Step 10: Autoregressive append ──
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # Track SAC tokens
            if self._sac_mask is not None and 0 <= next_id < self._sac_mask.size(0):
                if self._sac_mask[next_id]:
                    sac_tokens.append(next_id)

            # EOS check
            if next_id == eos_token_id:
                break

            # Get logits for next step
            last_logits, states = self.system1.step(next_token.unsqueeze(0), states)
            last_logits = last_logits[0]  # (V,)

        return generated

    def _post_gen_reinforce(
        self,
        node,
        generated_token_id: int,
        logits: torch.Tensor,
        step: int,
    ):
        """Post-generation reinforcement (spec Step 8).

        Updates S_overflow with injection value weighting:
        - High-value hit (+20): SHADE injected token System 1 ranked ≥20
        - Low-value hit (+5):   SHADE injected token System 1 ranked top-3
        - Miss (-20):           generated token not in target_dist
        """
        if generated_token_id in node.target_dist:
            # Compute System 1's pre-injection rank for this token
            sorted_ids = logits.argsort(descending=True).tolist()
            try:
                rank = sorted_ids.index(generated_token_id)
            except ValueError:
                rank = len(sorted_ids)

            if rank >= 20:
                # High-value hit — SHADE did real work
                delta = self.config.overflow_hit_delta  # +20
            else:
                # Low-value hit — confirmation only
                delta = 5
            node.s_overflow = min(node.s_overflow + delta, self.config.s_max)
            node.hit_count += 1
        else:
            # Miss
            node.s_overflow = max(0, node.s_overflow - self.config.overflow_miss_delta)
            node.miss_count += 1
            # Axiomatic decay: 1-in-5
            if node.s_base >= 50 and node.miss_count % self.config.miss_decay_interval == 0:
                node.s_base = max(0, node.s_base - 1)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample a token with temperature, top-k, and top-p filtering."""
        if logits.dim() == 2:
            logits = logits[0]

        logits = logits / max(temperature, 1e-6)

        # Top-k filtering
        if top_k > 0 and top_k < logits.size(-1):
            indices_to_remove = logits < torch.topk(logits, top_k).values[-1]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ── Parameter Report ────────────────────────────────────────

    def param_report(self) -> Dict[str, int]:
        """Report parameter counts by component."""
        s1_trainable = sum(p.numel() for p in self.system1.parameters() if p.requires_grad)
        s1_frozen = sum(p.numel() for p in self.system1.parameters() if not p.requires_grad)
        total = s1_trainable + s1_frozen

        report = {
            "System 1 (trainable)": s1_trainable,
            "System 1 (frozen)": s1_frozen,
            "SHADE nodes": self.shade.size,
            "SHADE edges": self.shade.edge_count,
            "Total neural params": total,
        }

        print("\n  ─── ARNL V1.1 Parameter Report ───")
        for k, v in report.items():
            if isinstance(v, int) and v > 1000:
                print(f"    {k:30s} {v:>14,}")
            else:
                print(f"    {k:30s} {v:>14}")
        print(f"    {'System 1 % of neural':30s} {100*s1_trainable/max(1,total):>13.1f}%")
        print()
        return report

    # ── Chunked Cross Entropy (memory-efficient training) ──────

    @staticmethod
    def _chunked_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Compute cross-entropy in chunks to avoid OOM on large vocabs."""
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        n_chunks = (logits_flat.size(0) + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, logits_flat.size(0))
            chunk_loss = F.cross_entropy(
                logits_flat[start:end],
                targets_flat[start:end],
                ignore_index=-100,
                reduction='sum',
            )
            total_loss = total_loss + chunk_loss

        # Mean over valid positions
        valid = (targets_flat != -100).sum()
        return total_loss / valid.clamp(min=1)
