"""
ARNL Training Utilities
========================

Three-phase training procedure for Arnold:

    Phase 1 — EMLM Pretraining
        Train System 1 (LLSU) on entity-masked language modelling.
        All factual entities replaced with [ENTITY_*] tokens.
        System 1 converges on pure syntactic structure.

    Phase 2 — Projection Alignment
        Train W_proj + lightweight LoRA adapters on System 1.
        System 1 core weights frozen.
        Objective: W_proj · v_sys2 maps to region of System 1's
        residual stream that produces the target token.

    Phase 3 — End-to-End Fine-Tuning
        Train W_proj + LoRA only.  Task-specific alignment.
        Ensures injection feels natural in System 1's generative flow.

Entity-Masked Language Modelling (EMLM):
    Proper Nouns      →  [ENTITY_NOUN]
    Numerical Data    →  [ENTITY_NUM]
    Named Concepts    →  [ENTITY_CONCEPT]
    Domain-Specific   →  [ENTITY_DOMAIN]
"""

from __future__ import annotations

import re
import math
import time
from typing import Optional, List, Dict, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arnl.config import ARNLConfig
from arnl.model import Arnold


# ════════════════════════════════════════════════════════════════
# Entity-Masked Language Modelling (EMLM) — Data Preparation
# ════════════════════════════════════════════════════════════════

class EMLMMasker:
    """Masks factual entities in text for EMLM training.

    Replaces identified entities with generic placeholder tokens so that
    System 1's weights optimise entirely for syntactic transitions.

    In production, entity detection would use a NER model or curated
    entity lists.  This implementation provides rule-based heuristics
    and a pluggable entity detector interface.

    Parameters
    ----------
    config : ARNLConfig
    entity_detector : callable | None
        Custom entity detector: text → list of (start, end, entity_type).
        If None, uses the built-in regex heuristic.
    """

    # Default entity type mapping
    TYPE_MAP = {
        "PERSON": "[ENTITY_NOUN]",
        "GPE": "[ENTITY_NOUN]",        # Geopolitical entity
        "LOC": "[ENTITY_NOUN]",
        "ORG": "[ENTITY_NOUN]",
        "PRODUCT": "[ENTITY_CONCEPT]",
        "EVENT": "[ENTITY_CONCEPT]",
        "DATE": "[ENTITY_NUM]",
        "TIME": "[ENTITY_NUM]",
        "MONEY": "[ENTITY_NUM]",
        "QUANTITY": "[ENTITY_NUM]",
        "CARDINAL": "[ENTITY_NUM]",
        "ORDINAL": "[ENTITY_NUM]",
        "PERCENT": "[ENTITY_NUM]",
        "NOUN": "[ENTITY_NOUN]",
        "NUM": "[ENTITY_NUM]",
        "CONCEPT": "[ENTITY_CONCEPT]",
        "DOMAIN": "[ENTITY_DOMAIN]",
    }

    def __init__(
        self,
        config: ARNLConfig,
        entity_detector: Optional[Callable] = None,
    ):
        self.config = config
        self.entity_detector = entity_detector or self._default_detector

    def mask_text(self, text: str) -> str:
        """Apply entity masking to a text string.

        Returns the text with all detected entities replaced by
        their corresponding [ENTITY_*] tokens.
        """
        entities = self.entity_detector(text)

        # Sort by start position, descending (to replace from end)
        entities.sort(key=lambda e: e[0], reverse=True)

        masked = text
        for start, end, etype in entities:
            replacement = self.TYPE_MAP.get(etype, "[ENTITY_NOUN]")
            masked = masked[:start] + replacement + masked[end:]

        return masked

    def mask_batch(self, texts: List[str]) -> List[str]:
        """Mask a batch of texts."""
        return [self.mask_text(t) for t in texts]

    @staticmethod
    def _default_detector(text: str) -> List[Tuple[int, int, str]]:
        """Simple regex-based entity detector (heuristic fallback).

        Detects:
            - Capitalised words (likely proper nouns)
            - Numbers and dates
            - Common domain-specific patterns
        """
        entities = []

        # Numbers (integers, decimals, percentages, dates)
        for m in re.finditer(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'      # dates
            r'|\b\d+(?:\.\d+)?%?\b'                      # numbers / percentages
            r'|\$\d+(?:,\d{3})*(?:\.\d+)?',              # money
            text,
        ):
            entities.append((m.start(), m.end(), "NUM"))

        # Capitalised words not at sentence start (likely proper nouns)
        for m in re.finditer(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]+', text):
            # Skip very short words and common non-entities
            word = m.group()
            if len(word) > 2 and word.lower() not in {
                "the", "and", "but", "for", "nor", "yet", "can",
                "may", "his", "her", "its", "our", "has", "was",
                "are", "did", "had", "not", "this", "that", "with",
            }:
                entities.append((m.start(), m.end(), "NOUN"))

        return entities


# ════════════════════════════════════════════════════════════════
# EMLM Verification Test
# ════════════════════════════════════════════════════════════════

def verify_emlm(model: Arnold, tokenizer, prompt: str = "The capital of France is") -> dict:
    """EMLM Verification: probe System 1 WITHOUT injection.

    A correctly trained LLSU produces a high-entropy distribution over
    all nouns — NOT 'Paris' with high probability.  If 'Paris' appears
    dominantly, EMLM masking was incomplete.

    Parameters
    ----------
    model : Arnold
    tokenizer : tokenizer with encode/decode methods
    prompt : str

    Returns
    -------
    dict with top-5 predictions and entropy.
    """
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=model.device)

    # Run System 1 ONLY (no injection)
    with torch.no_grad():
        states = model.system1.init_states(1)
        logits, _ = model.system1(input_ids, states=states)
        last_logits = logits[0, -1]  # last position
        probs = F.softmax(last_logits, dim=-1)

        # Entropy
        log_probs = F.log_softmax(last_logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()

        # Top-5
        top5_probs, top5_ids = probs.topk(5)
        top5 = [
            {"token": tokenizer.decode([tid.item()]), "prob": p.item()}
            for tid, p in zip(top5_ids, top5_probs)
        ]

    return {
        "prompt": prompt,
        "entropy": entropy,
        "max_entropy": math.log(model.config.vocab_size),
        "entropy_ratio": entropy / math.log(model.config.vocab_size),
        "top5": top5,
        "verdict": "PASS" if entropy / math.log(model.config.vocab_size) > 0.5 else "WARN: low entropy",
    }


# ════════════════════════════════════════════════════════════════
# Simple Text Dataset for EMLM
# ════════════════════════════════════════════════════════════════

class EMLMDataset(Dataset):
    """Simple text dataset for EMLM training.

    Parameters
    ----------
    texts : list[str]
        Raw texts (will be entity-masked during init).
    tokenizer : tokenizer with encode() method
    masker : EMLMMasker
    max_len : int
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        masker: EMLMMasker,
        max_len: int = 512,
    ):
        self.max_len = max_len
        self.samples = []

        for text in texts:
            masked = masker.mask_text(text)
            ids = tokenizer.encode(masked)
            if len(ids) > max_len:
                # Sliding window
                for i in range(0, len(ids) - max_len, max_len // 2):
                    self.samples.append(ids[i:i + max_len])
            elif len(ids) > 1:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        ids = ids + [0] * (self.max_len - len(ids))  # pad
        ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


# ════════════════════════════════════════════════════════════════
# Training Loop — Phase 1 (EMLM Pretraining)
# ════════════════════════════════════════════════════════════════

def train_phase1(
    model: Arnold,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 3e-4,
    warmup_steps: int = 1000,
    log_interval: int = 100,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
) -> List[dict]:
    """Phase 1: EMLM Pretraining — System 1 only.

    Next-token prediction on entity-masked corpus.  System 1 converges
    on syntactic structure with zero factual encoding.
    """
    model.setup_phase1()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Linear warmup + cosine decay
    total_steps = len(train_loader) * epochs

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    logs = []
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            result = model(input_ids, labels=labels)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                log = {
                    "phase": 1,
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
                logs.append(log)
                print(
                    f"[Phase 1] Epoch {epoch} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | Avg {avg_loss:.4f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )

    return logs


# ════════════════════════════════════════════════════════════════
# Training Loop — Phase 2 (Projection Alignment)
# ════════════════════════════════════════════════════════════════

def train_phase2(
    model: Arnold,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-4,
    log_interval: int = 100,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
) -> List[dict]:
    """Phase 2: Projection Alignment — W_proj + LoRA adapters.

    System 1 core frozen.  LoRA adapters allow minor geometric
    adjustments without corrupting the EMLM-trained structure.

    Objective: minimise reconstruction loss so that W_proj · v_sys2
    maps to the region of System 1's residual stream that produces
    the target token.
    """
    # Apply LoRA if not already applied
    model.apply_lora_adapters()
    model.setup_phase2()
    model.to(device)
    model.train()

    # Collect trainable params: LoRA + W_proj
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    logs = []
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Compute injections from ground-truth targets
            target_ids = labels[:, 1:]  # Shifted targets
            # Pad target_ids to match input length
            pad = torch.full(
                (target_ids.size(0), 1), model.config.pad_token_id,
                device=device, dtype=torch.long,
            )
            target_ids_padded = torch.cat([target_ids, pad], dim=1)

            injections = model.compute_batch_injections(input_ids, target_ids_padded)

            result = model(input_ids, labels=labels, injections=injections)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                log = {
                    "phase": 2,
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                }
                logs.append(log)
                print(
                    f"[Phase 2] Epoch {epoch} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | Avg {avg_loss:.4f}"
                )

    return logs


# ════════════════════════════════════════════════════════════════
# Training Loop — Phase 3 (End-to-End Fine-Tuning)
# ════════════════════════════════════════════════════════════════

def train_phase3(
    model: Arnold,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 5e-5,
    log_interval: int = 100,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
) -> List[dict]:
    """Phase 3: End-to-End Fine-Tuning — W_proj + LoRA only.

    Task-specific alignment.  Ensures injection feels natural in
    System 1's generative flow.
    """
    model.setup_phase3()
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    logs = []
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # With injection
            target_ids = labels[:, 1:]
            pad = torch.full(
                (target_ids.size(0), 1), model.config.pad_token_id,
                device=device, dtype=torch.long,
            )
            target_ids_padded = torch.cat([target_ids, pad], dim=1)
            injections = model.compute_batch_injections(input_ids, target_ids_padded)

            result = model(input_ids, labels=labels, injections=injections)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                log = {
                    "phase": 3,
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                }
                logs.append(log)
                print(
                    f"[Phase 3] Epoch {epoch} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | Avg {avg_loss:.4f}"
                )

    return logs


# ════════════════════════════════════════════════════════════════
# Full Training Pipeline
# ════════════════════════════════════════════════════════════════

def train_full_pipeline(
    model: Arnold,
    phase1_loader: DataLoader,
    phase2_loader: DataLoader,
    phase3_loader: Optional[DataLoader] = None,
    phase1_epochs: int = 3,
    phase2_epochs: int = 1,
    phase3_epochs: int = 1,
    phase1_lr: float = 3e-4,
    phase2_lr: float = 1e-4,
    phase3_lr: float = 5e-5,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> dict:
    """Run the complete 3-phase ARNL training pipeline.

    Parameters
    ----------
    model : Arnold
    phase1_loader : DataLoader
        EMLM-masked data for Phase 1.
    phase2_loader : DataLoader
        Factual data with ground-truth targets for Phase 2.
    phase3_loader : DataLoader | None
        Task-specific data for Phase 3.  If None, uses phase2_loader.
    save_path : str | None
        If provided, saves the model after each phase.

    Returns
    -------
    dict with all training logs.
    """
    all_logs = {"phase1": [], "phase2": [], "phase3": []}
    t0 = time.time()

    # ── Phase 1: EMLM Pretraining ──
    print("=" * 60)
    print("Phase 1: EMLM Pretraining (System 1 only)")
    print("=" * 60)
    all_logs["phase1"] = train_phase1(
        model, phase1_loader, epochs=phase1_epochs, lr=phase1_lr, device=device,
    )
    if save_path:
        model.save_pretrained(save_path + "_phase1")

    # ── Phase 2: Projection Alignment ──
    print("=" * 60)
    print("Phase 2: Projection Alignment (W_proj + LoRA)")
    print("=" * 60)
    all_logs["phase2"] = train_phase2(
        model, phase2_loader, epochs=phase2_epochs, lr=phase2_lr, device=device,
    )
    if save_path:
        model.save_pretrained(save_path + "_phase2")

    # ── Phase 3: End-to-End Fine-Tuning ──
    if phase3_loader is None:
        phase3_loader = phase2_loader

    print("=" * 60)
    print("Phase 3: End-to-End Fine-Tuning (W_proj + LoRA)")
    print("=" * 60)
    all_logs["phase3"] = train_phase3(
        model, phase3_loader, epochs=phase3_epochs, lr=phase3_lr, device=device,
    )
    if save_path:
        model.save_pretrained(save_path + "_final")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    return all_logs
