"""
ARNL Training Utilities  (V1.1 — SHADE Edition)
=================================================

Two-phase training procedure for Arnold:

    Phase 1 — EMLM Pretraining
        Train System 1 (LLSU) on entity-masked language modelling.
        All factual entities replaced with [ENTITY_*] tokens.
        System 1 converges on pure syntactic structure.

    Phase 2 — LoRA Fine-Tuning
        Attach lightweight LoRA adapters to System 1.
        System 1 core weights frozen; only LoRA deltas are trained.
        Improves System 1's downstream fluency without corrupting
        the EMLM-trained syntactic backbone.

SHADE (System 2) is a non-neural database — it has no trainable
parameters and is populated offline via populate_map.py.

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

    Uses configurable regex patterns to identify factual spans and
    replaces them with [ENTITY_*] placeholders.

    Parameters
    ----------
    patterns : dict[str, str] | None
        Override regex patterns.  Keys: 'nouns', 'numbers', 'concepts',
        'domain'.  Values: regex pattern strings.
    """

    DEFAULT_PATTERNS = {
        "numbers": r"\b\d[\d,]*\.?\d*\b",
        "nouns": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
        "concepts": r"\b(?:DNA|RNA|HTTP|API|GPU|CPU|AI|ML|NLP|LLM)\b",
        "domain": r"",
    }

    REPLACEMENTS = {
        "numbers": "[ENTITY_NUM]",
        "nouns": "[ENTITY_NOUN]",
        "concepts": "[ENTITY_CONCEPT]",
        "domain": "[ENTITY_DOMAIN]",
    }

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        self.patterns = {**self.DEFAULT_PATTERNS, **(patterns or {})}
        self._compiled = {
            k: re.compile(v, re.MULTILINE) if v else None
            for k, v in self.patterns.items()
        }

    def mask_text(self, text: str) -> str:
        """Replace factual entities with EMLM placeholders."""
        for key in ["concepts", "domain", "numbers", "nouns"]:
            pat = self._compiled.get(key)
            if pat is not None:
                text = pat.sub(self.REPLACEMENTS[key], text)
        return text

    def mask_batch(self, texts: List[str]) -> List[str]:
        return [self.mask_text(t) for t in texts]


# ════════════════════════════════════════════════════════════════
# EMLM Verification Test
# ════════════════════════════════════════════════════════════════

def verify_emlm(model: Arnold, tokenizer, prompt: str = "The capital of France is") -> dict:
    """EMLM Verification: probe System 1 WITHOUT SHADE injection.

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

    # Run System 1 ONLY (no SHADE injection)
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

    Supports two tokenizer modes:
      1. **SALTTokenizer** — uses ``emlm_mask()`` for deterministic
         class-based masking (entity tokens = single IDs, no regex).
      2. **HuggingFace tokenizer** — uses EMLMMasker regex heuristic,
         then matches entity token subsequences in labels.

    Parameters
    ----------
    texts : list[str]
        Raw texts (will be entity-masked during init).
    tokenizer : tokenizer with encode() method (SALTTokenizer or HF)
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
        self._use_salt = hasattr(tokenizer, 'emlm_mask')

        if self._use_salt:
            # SALT path: use class_ids-based masking
            for text in texts:
                enc = tokenizer.emlm_mask(text)
                ids = enc.input_ids
                entity_spans = enc.entity_spans
                if len(ids) > max_len:
                    for i in range(0, len(ids) - max_len, max_len // 2):
                        chunk_ids = ids[i:i + max_len]
                        # Adjust entity spans to the chunk window
                        chunk_spans = [
                            (s - i, e - i, t) for s, e, t in entity_spans
                            if s >= i and e <= i + max_len
                        ]
                        self.samples.append((chunk_ids, chunk_spans))
                elif len(ids) > 1:
                    self.samples.append((ids, entity_spans))
        else:
            # Legacy HF path: regex masking + entity sequence detection
            _ENTITY_STRINGS = [
                "[ENTITY_NOUN]", "[ENTITY_NUM]",
                "[ENTITY_CONCEPT]", "[ENTITY_DOMAIN]",
            ]
            self._entity_seqs: List[List[int]] = [
                tokenizer.encode(e) for e in _ENTITY_STRINGS
            ]
            self._entity_seq_lens = [len(s) for s in self._entity_seqs]

            for text in texts:
                masked = masker.mask_text(text)
                ids = tokenizer.encode(masked)
                if len(ids) > max_len:
                    for i in range(0, len(ids) - max_len, max_len // 2):
                        self.samples.append(ids[i:i + max_len])
                elif len(ids) > 1:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._use_salt:
            ids, entity_spans = self.samples[idx]
            ids = ids + [0] * (self.max_len - len(ids))  # pad
            ids = torch.tensor(ids, dtype=torch.long)
            labels = ids.clone()

            # Mask entity spans → -100
            for start, end, _ in entity_spans:
                end_c = min(end, self.max_len)
                start_c = max(start, 0)
                if start_c < end_c:
                    labels[start_c:end_c] = -100

            # Mask padding
            labels[labels == 0] = -100
            return {"input_ids": ids, "labels": labels}
        else:
            # Legacy HF path
            ids = self.samples[idx]
            ids = ids + [0] * (self.max_len - len(ids))  # pad
            ids = torch.tensor(ids, dtype=torch.long)
            labels = ids.clone()

            ids_list = labels.tolist()
            for entity_seq, n in zip(self._entity_seqs, self._entity_seq_lens):
                for i in range(len(ids_list) - n + 1):
                    if ids_list[i:i + n] == entity_seq:
                        labels[i:i + n] = -100

            labels[labels == 0] = -100
            return {"input_ids": ids, "labels": labels}


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

            # V1.1: forward returns (logits, states)
            logits, _ = model(input_ids)

            # Compute loss externally (shifted NTP)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = model._chunked_cross_entropy(shift_logits, shift_labels)

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
# Training Loop — Phase 2 (LoRA Fine-Tuning)
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
    """Phase 2: LoRA Fine-Tuning — adapters only.

    System 1 core frozen.  LoRA adapters allow minor geometric
    adjustments to improve downstream fluency without corrupting
    the EMLM-trained structure.
    """
    # setup_phase2 handles LoRA application + freezing
    model.setup_phase2()
    model.to(device)
    model.train()

    # Collect trainable params (LoRA only)
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    logs = []
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # V1.1: forward returns (logits, states) — no injections
            logits, _ = model(input_ids)

            # Compute loss externally (shifted NTP)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = model._chunked_cross_entropy(shift_logits, shift_labels)

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
# Full Training Pipeline
# ════════════════════════════════════════════════════════════════

def train_full_pipeline(
    model: Arnold,
    phase1_loader: DataLoader,
    phase2_loader: Optional[DataLoader] = None,
    phase1_epochs: int = 3,
    phase2_epochs: int = 1,
    phase1_lr: float = 3e-4,
    phase2_lr: float = 1e-4,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> dict:
    """Run the complete 2-phase ARNL V1.1 training pipeline.

    Parameters
    ----------
    model : Arnold
    phase1_loader : DataLoader
        EMLM-masked data for Phase 1.
    phase2_loader : DataLoader | None
        Data for Phase 2 LoRA fine-tuning.  If None, uses phase1_loader.
    save_path : str | None
        If provided, saves the model after each phase.

    Returns
    -------
    dict with all training logs.
    """
    all_logs = {"phase1": [], "phase2": []}
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

    # ── Phase 2: LoRA Fine-Tuning ──
    if phase2_loader is None:
        phase2_loader = phase1_loader

    print("=" * 60)
    print("Phase 2: LoRA Fine-Tuning (adapters only)")
    print("=" * 60)
    all_logs["phase2"] = train_phase2(
        model, phase2_loader, epochs=phase2_epochs, lr=phase2_lr, device=device,
    )
    if save_path:
        model.save_pretrained(save_path + "_final")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    return all_logs
