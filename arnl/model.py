"""
Arnold — Full ARNL Model
=========================

System 3 Symbiotic Bundle:  a heterogeneous, inference-locked
architecture in which two independently trained models and one
controller operate as a unified generative system.

    System 1 (LLSU)         — determines HOW something is said
    System 2 (Hyper-Map)    — determines WHAT is said
    Reasoning Head          — determines WHETHER and WHEN

Neither System 1 nor System 2 generates tokens alone; generation is
the emergent product of their interaction through the Injection Bridge.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from arnl.config import ARNLConfig
from arnl.system1 import LLSU
from arnl.system2 import HyperAdjacencyMap, Hyperedge
from arnl.reasoning_head import ReasoningHead, InjectionPlan
from arnl.injection_bridge import InjectionBridge
from arnl.decay_engine import DecayEngine


class Arnold(nn.Module):
    """Arnold — the complete ARNL V1.0 model.

    Implements the full 10-step generative loop from the spec:
        1. Input ingestion
        2. Semantic Saliency Filter
        3. Hyperedge retrieval
        4. Conflict scan
        5. System 1 forward pass
        6. Sigmoid ramp computation
        7. Latent injection
        8. Token generation
        9. Post-generation reinforcement
       10. Autoregressive append

    Parameters
    ----------
    config : ARNLConfig
        Architecture configuration.  Use preset constructors
        (``ARNLConfig.arnold_7b()``, etc.) for standard scales.
    """

    def __init__(self, config: ARNLConfig):
        super().__init__()
        self.config = config

        # ── System 1: Fluidity Core ──
        self.system1 = LLSU(config)

        # ── Reasoning Head: Control Logic ──
        self.reasoning_head = ReasoningHead(config)

        # ── Injection Bridge: W_proj ──
        self.injection_bridge = InjectionBridge(config)

        # ── System 2: Reliability Core (non-neural) ──
        self.hyper_map = HyperAdjacencyMap(config)

        # ── Decay Engine (non-neural) ──
        self.decay_engine = DecayEngine(config)

        # ── Generation diagnostics ──
        self._last_plans: List[InjectionPlan] = []
        self._generation_log: List[dict] = []

    @property
    def device(self) -> torch.device:
        return self.system1.device

    def param_report(self) -> dict:
        """Count parameters per component."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        s1 = _count(self.system1)
        rh = _count(self.reasoning_head)
        wb = _count(self.injection_bridge)
        total = s1 + rh + wb
        return {
            "system1_params": s1,
            "reasoning_head_params": rh,
            "injection_bridge_params": wb,
            "total_trainable": total,
            "system2_edges": self.hyper_map.size,
            "budget_pct": {
                "system1": f"{s1 / max(1, total):.1%}",
                "reasoning_head": f"{rh / max(1, total):.1%}",
                "injection_bridge": f"{wb / max(1, total):.1%}",
            },
        }

    # ────────────────────────────────────────────────────────────
    # Training Forward Pass (teacher-forced)
    # ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
        injections: Optional[Dict[int, torch.Tensor]] = None,
    ) -> dict:
        """Teacher-forced forward pass for training.

        During Phase 1 (EMLM), ``injections`` is None — System 1 trains
        alone on entity-masked data.

        During Phase 2/3, ``injections`` can be pre-computed by calling
        ``compute_batch_injections()`` or generated on the fly.

        Parameters
        ----------
        input_ids : LongTensor[B, T]
        labels : LongTensor[B, T] | None
            Shifted targets for cross-entropy loss.
        states : list | None
            Per-layer recurrent states.
        injections : dict | None
            Pre-computed injection tensors per layer.

        Returns
        -------
        dict with keys:
            logits : Tensor[B, T, vocab_size]
            loss : Tensor (scalar) if labels provided
            states : list of recurrent states
        """
        logits, new_states = self.system1(input_ids, states=states, injections=injections)

        result = {"logits": logits, "states": new_states}

        if labels is not None:
            # Standard shift for causal LM
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            result["loss"] = loss

        return result

    # ────────────────────────────────────────────────────────────
    # Batch Injection Computation (for Phase 2/3 training)
    # ────────────────────────────────────────────────────────────

    def compute_batch_injections(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Compute injection tensors for all positions in a batch.

        Used during Phase 2/3 training when ground-truth targets are
        known.  For each position, looks up the target token's semantic
        embedding, projects through W_proj, and computes a dummy α=1.0.

        Parameters
        ----------
        input_ids : LongTensor[B, T]
        target_ids : LongTensor[B, T]
            Ground-truth next tokens for each position.

        Returns
        -------
        dict mapping layer index → Tensor[B, T, d_model]
        """
        B, T = input_ids.shape

        # Get semantic embeddings for all target tokens
        with torch.no_grad():
            v_targets = self.reasoning_head._embed_semantic(target_ids)  # (B, T, d_sem)

        # Project through W_proj with α = 1.0
        projected = self.injection_bridge(v_targets, alpha=1.0)  # (B, T, d_model)

        # Distribute to injection layers
        inj_layers = self.config.injection_layers
        if inj_layers is None:
            inj_layers = [self.config.n_layers - 1]

        return {layer_idx: projected for layer_idx in inj_layers}

    # ────────────────────────────────────────────────────────────
    # Autoregressive Generation (the full 10-step loop)
    # ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Full autoregressive generation with the ARNL loop.

        Implements all 10 steps from the spec diagram:
            1. Input ingestion
            2. Semantic Saliency Filter [Reasoning Head]
            3. Hyperedge retrieval [System 2]
            4. Conflict scan [Reasoning Head]
            5. System 1 forward pass [LLSU]
            6. Sigmoid ramp computation [Reasoning Head]
            7. Latent injection [W_proj Bridge]
            8. Token generation [System 1 FFN + Softmax]
            9. Post-generation reinforcement [Decay Engine]
           10. Autoregressive append

        Parameters
        ----------
        input_ids : LongTensor[B, T]
            Prompt tokens.
        max_new_tokens : int
        temperature : float
        top_k : int
        top_p : float
        do_sample : bool
        eos_token_id : int | None

        Returns
        -------
        generated_ids : LongTensor[B, T + max_new_tokens]
        generation_log : list[dict]
            Per-step diagnostic log for auditability.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        B, T = input_ids.shape
        device = input_ids.device
        assert B == 1, "Generation currently supports batch_size=1"

        # ── Step 1: Initialise — process the prompt ──
        # Run System 1 through the full prompt to build recurrent state
        states = self.system1.init_states(B)
        logits, states = self.system1(input_ids, states=states)

        generated = input_ids.clone()
        generation_log: List[dict] = []

        for step_idx in range(max_new_tokens):
            context = generated[0]  # (T_current,)

            # ── Steps 2–4: Reasoning Head pipeline ──
            plan = self.reasoning_head(context, self.hyper_map, self.decay_engine)

            # ── Step 5–7: Compute injection and run System 1 step ──
            injection_t = None
            if plan.v_target is not None and plan.alpha > 1e-6:
                v = plan.v_target.to(device).unsqueeze(0)        # (1, d_sem)
                alpha_t = torch.tensor([plan.alpha], device=device, dtype=v.dtype)
                injection_t = self.injection_bridge(v, alpha_t)  # (1, d_model)
                injection_t = injection_t.squeeze(0)             # (d_model,) for step()

            # Single-token step through System 1
            last_token = generated[:, -1]  # (B,)
            step_logits, states = self.system1.step(last_token, states, injection_t)

            # ── Step 8: Token generation (sampling) ──
            next_logits = step_logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                v_topk, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v_topk[:, -1:]] = -float("inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = next_logits.sort(descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[mask] = -float("inf")
                # Scatter back
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, 1)  # (B, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # ── Step 9: Post-generation reinforcement ──
            step_log = {
                "step": step_idx,
                "generated_token_id": next_token.item(),
                "alpha": plan.alpha,
                "is_function_word": plan.is_function_word,
                "is_vacuum": plan.is_vacuum,
                "conflict_actions": plan.conflict_actions,
                "edge_key": plan.edge.key if plan.edge else None,
                "decay_status": None,
            }

            if plan.edge is not None:
                is_hit = next_token.item() == plan.edge.target_token_id
                status = self.decay_engine.update(plan.edge, is_hit, self.hyper_map)
                step_log["decay_status"] = status
                step_log["is_hit"] = is_hit
                step_log["s_base"] = plan.edge.s_base
                step_log["s_overflow"] = plan.edge.s_overflow

            generation_log.append(step_log)

            # ── Step 10: Autoregressive append ──
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        # Clean up dead edges accumulated during generation
        self.hyper_map.purge_dead()
        self._generation_log = generation_log

        return generated, generation_log

    # ────────────────────────────────────────────────────────────
    # Knowledge Management (Granular Semantic Auditability)
    # ────────────────────────────────────────────────────────────

    def hard_lock_fact(self, anchor_ids: List[int]) -> bool:
        """Hard Lock a factual pathway — cannot be argued out of this fact.

        Inversion decay is suppressed.  S_base set to 1000.
        """
        from arnl.utils import semantic_hash
        key = semantic_hash(anchor_ids)
        return self.hyper_map.hard_lock(key)

    def hard_delete_fact(self, anchor_ids: List[int]) -> bool:
        """Hard Delete — permanently removes the factual pathway."""
        from arnl.utils import semantic_hash
        key = semantic_hash(anchor_ids)
        return self.hyper_map.delete(key) is not None

    def hard_override_fact(self, anchor_ids: List[int]) -> bool:
        """Hard Override — sets S_base=0, triggers deletion next step."""
        from arnl.utils import semantic_hash
        key = semantic_hash(anchor_ids)
        return self.hyper_map.hard_override(key) is not None

    def export_beliefs(self) -> str:
        """Export the full belief state as human-readable JSON.

        Every factual assertion the model holds can be inspected:
            - The exact context hash that triggers retrieval
            - The target token and its axiomatic strength
            - Full decay history (hits, misses)
            - Tier status and timestamps
        """
        return self.hyper_map.export_belief_state_json()

    def ingest_knowledge(
        self,
        context_ids: torch.Tensor,
        target_token_id: int,
    ) -> Optional[Hyperedge]:
        """Add a new factual pathway through the Reasoning Head gate."""
        return self.reasoning_head.ingest_fact(context_ids, target_token_id, self.hyper_map)

    def prepopulate_axiom(
        self,
        anchor_ids: List[int],
        target_token_id: int,
        tier_label: str = "axiom",
    ) -> Hyperedge:
        """Pre-populate a Hard-Locked axiom at S_base=1000.

        Used for cold-start knowledge loading from curated knowledge bases.
        """
        v_target = self.reasoning_head._embed_semantic(
            torch.tensor([target_token_id], device=self.device)
        ).squeeze(0)

        edge = self.hyper_map.insert(
            anchor_ids=anchor_ids,
            v_target=v_target,
            target_token_id=target_token_id,
            s_base=1000,
            tier_label=tier_label,
        )
        edge.s_base_original = 1000
        edge.hard_locked = True
        return edge

    # ────────────────────────────────────────────────────────────
    # Phase-aware parameter freezing
    # ────────────────────────────────────────────────────────────

    def setup_phase1(self):
        """Phase 1: EMLM Pretraining — only System 1 trains."""
        for p in self.system1.parameters():
            p.requires_grad = True
        for p in self.reasoning_head.parameters():
            p.requires_grad = False
        for p in self.injection_bridge.parameters():
            p.requires_grad = False

    def setup_phase2(self):
        """Phase 2: Projection Alignment — W_proj + LoRA adapters train.

        System 1 core weights frozen.  Apply LoRA before calling this.
        Reasoning Head frozen.  Use setup_phase2_full to also train the
        Reasoning Head's learnable classifiers.
        """
        for p in self.system1.parameters():
            p.requires_grad = False
        # LoRA params (if applied) will have requires_grad=True by default
        for p in self.injection_bridge.parameters():
            p.requires_grad = True
        for p in self.reasoning_head.parameters():
            p.requires_grad = False

    def setup_phase2_full(self):
        """Phase 2 (full): W_proj + LoRA + Reasoning Head learnable sub-modules.

        Like setup_phase2 but also trains the Reasoning Head's context encoder,
        saliency head, function-word classifier, and tier classifier.
        The frozen semantic embedding and syntactic centroids remain non-trainable.
        """
        # Freeze System 1 core
        for p in self.system1.parameters():
            p.requires_grad = False
        # Train W_proj
        for p in self.injection_bridge.parameters():
            p.requires_grad = True
        # Freeze the entire Reasoning Head first, then selectively unfreeze
        for p in self.reasoning_head.parameters():
            p.requires_grad = False
        # Unfreeze learnable sub-modules (NOT semantic_embed, NOT syntactic_centroids)
        for sub in [
            self.reasoning_head.context_encoder,
            self.reasoning_head.saliency_head,
            self.reasoning_head.function_word_head,
            self.reasoning_head.tier_classifier,
        ]:
            for p in sub.parameters():
                p.requires_grad = True

    def setup_phase3(self):
        """Phase 3: End-to-End Fine-Tuning — W_proj + LoRA + Reasoning Head."""
        self.setup_phase2_full()  # Same freeze pattern as full Phase 2

    def apply_lora_adapters(self):
        """Apply LoRA adapters to System 1 for Phase 2/3 training."""
        from arnl.utils import apply_lora
        apply_lora(
            self.system1,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )

    # ────────────────────────────────────────────────────────────
    # Serialisation
    # ────────────────────────────────────────────────────────────

    def save_pretrained(self, path: str):
        """Save all components to disk."""
        import os
        os.makedirs(path, exist_ok=True)

        # Neural weights
        torch.save(self.system1.state_dict(), os.path.join(path, "system1.pt"))
        torch.save(self.reasoning_head.state_dict(), os.path.join(path, "reasoning_head.pt"))
        torch.save(self.injection_bridge.state_dict(), os.path.join(path, "injection_bridge.pt"))

        # System 2 map
        self.hyper_map.save(os.path.join(path, "hyper_map.json"))

        # Config
        import json
        from dataclasses import asdict
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "Arnold":
        """Load a saved Arnold model."""
        import os, json
        from dataclasses import fields

        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)

        # Filter out any keys not in ARNLConfig
        valid_keys = {f.name for f in fields(ARNLConfig)}
        config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ARNLConfig(**config_dict)

        model = cls(config)
        model.system1.load_state_dict(
            torch.load(os.path.join(path, "system1.pt"), map_location=device)
        )
        model.reasoning_head.load_state_dict(
            torch.load(os.path.join(path, "reasoning_head.pt"), map_location=device)
        )
        model.injection_bridge.load_state_dict(
            torch.load(os.path.join(path, "injection_bridge.pt"), map_location=device)
        )
        model.hyper_map.load(os.path.join(path, "hyper_map.json"), device=torch.device(device))
        model.to(device)
        return model

    def __repr__(self) -> str:
        report = self.param_report()
        return (
            f"Arnold(arch={self.config.arch_name} v{self.config.version}, "
            f"params={report['total_trainable']:,}, "
            f"system2_edges={report['system2_edges']}, "
            f"budget=[S1:{report['budget_pct']['system1']}, "
            f"RH:{report['budget_pct']['reasoning_head']}, "
            f"W:{report['budget_pct']['injection_bridge']}])"
        )
