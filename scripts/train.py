#!/usr/bin/env python3
"""
scripts/train.py — ARNL Three-Phase Training Script
=====================================================

Trains Arnold through all three phases with full checkpoint management.
On every run it inspects the checkpoint directory and resumes from
wherever it left off — no manual phase tracking required.

────────────────────────────────────────────────────────────────
Checkpoint layout (inside --checkpoint-dir):
    config.json                 — ARNLConfig (written on first run)
    phase1/
        system1.pt              — System 1 weights after Phase 1
        optimizer.pt            — Optimizer state (for resuming mid-epoch)
        state.json              — {epoch, global_step, best_loss, done}
    phase2/
        system1_lora.pt         — LoRA adapter weights grafted onto System 1
        reasoning_head.pt       — Reasoning Head learnable sub-modules
        injection_bridge.pt     — W_proj weights
        optimizer.pt
        state.json
    phase3/
        system1_lora.pt
        reasoning_head.pt
        injection_bridge.pt
        optimizer.pt
        state.json
────────────────────────────────────────────────────────────────

Data formats
────────────────────────────────────────────────────────────────
Phase 1  (--phase1-data):  plain text file, one document per line.
         The script applies EMLM masking automatically.
         Lines beginning with '#' are skipped.

Phase 2/3 (--phase2-data / --phase3-data):  JSONL file, one fact per line.
         Each line must be a JSON object with:
           {
             "input_ids":      [int, ...],   // context token IDs
             "target_id":      int,          // next token to predict
             "anchor_ids":     [int, ...],   // optional: override saliency filter
             "label_ids":      [int, ...]    // full target sequence for LM loss
           }
         If "anchor_ids" is omitted the Reasoning Head derives anchors live.
         If "label_ids" is omitted "input_ids" is used as self-supervision.

If you have no real data, run with --synthetic to generate random token
sequences for a smoke test.
────────────────────────────────────────────────────────────────

Usage examples
────────────────────────────────────────────────────────────────
# Full pipeline from scratch with synthetic data:
    python scripts/train.py --synthetic --checkpoint-dir checkpoints/

# Resume (will detect which phases are done and skip them):
    python scripts/train.py --synthetic --checkpoint-dir checkpoints/

# Phase 1 only on real text:
    python scripts/train.py --phase 1 --phase1-data data/corpus.txt \\
        --checkpoint-dir checkpoints/ --epochs 3

# Phase 2 only (Phase 1 must already be done):
    python scripts/train.py --phase 2 --phase2-data data/facts.jsonl \\
        --checkpoint-dir checkpoints/

# Custom model size:
    python scripts/train.py --synthetic --preset arnold_1b \\
        --checkpoint-dir checkpoints/1b/
────────────────────────────────────────────────────────────────
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure repo root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arnl.config import ARNLConfig
from arnl.model import Arnold
from arnl.training import EMLMMasker, EMLMDataset
from scripts.data_config import DataConfig, build_phase1_dataset, build_phase23_dataset

_DATA_CONFIG_FILENAME = "data_config.json"


# ════════════════════════════════════════════════════════════════
# Datasets
# ════════════════════════════════════════════════════════════════

class _TokenIdDataset(Dataset):
    """Dataset that returns fixed-length windows of integer token IDs.

    Used for Phase 1 EMLM training when no real tokenizer is available.
    Also used to wrap pre-tokenised JSONL data.
    """

    def __init__(self, sequences: List[List[int]], pad_id: int = 0):
        self.samples = sequences
        self.pad_id = pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "labels": t.clone()}


class Phase1TextDataset(Dataset):
    """Loads a plain-text file, applies EMLM masking, and produces
    fixed-length token windows using a simple char-to-int tokeniser
    (or a HuggingFace tokenizer if available)."""

    def __init__(
        self,
        path: str,
        config: ARNLConfig,
        masker: EMLMMasker,
        max_len: int = 512,
        tokenizer=None,
    ):
        self.samples: List[List[int]] = []
        # Build a simple vocabulary from the masked text if no tokenizer
        self._build(path, config, masker, max_len, tokenizer)

    def _build(self, path, config, masker, max_len, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]

        masked_lines = masker.mask_batch(lines)

        # Simple char tokeniser (prints a warning if used)
        if tokenizer is None:
            print("[train.py] No tokenizer provided — using character-level fallback.")
            all_text = " ".join(masked_lines)
            chars = sorted(set(all_text))
            # reserve 0=pad, 1=bos, 2=eos
            char2id = {c: i + 3 for i, c in enumerate(chars)}
            char2id["[PAD]"] = 0
            char2id["[BOS]"] = 1
            char2id["[EOS]"] = 2
            encode = lambda t: [1] + [char2id.get(c, 3) for c in t] + [2]
        else:
            encode = tokenizer.encode

        for text in masked_lines:
            ids = encode(text)
            if len(ids) < 2:
                continue
            # Clip to vocab size
            ids = [min(i, config.vocab_size - 1) for i in ids]
            for start in range(0, max(1, len(ids) - max_len + 1), max_len // 2):
                chunk = ids[start: start + max_len]
                if len(chunk) >= 2:
                    padded = chunk + [0] * (max_len - len(chunk))
                    self.samples.append(padded[:max_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class Phase23Dataset(Dataset):
    """Loads a JSONL file for Phase 2/3 training.

    Each record may contain:
        input_ids  : list[int]   — context
        label_ids  : list[int]   — target sequence (optional; defaults to input_ids)
        target_id  : int         — single target token for injection labels
        anchor_ids : list[int]   — anchor tokens (optional)
    """

    def __init__(self, path: str, config: ARNLConfig, max_len: int = 512):
        self.samples = []
        self.config = config
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                self.samples.append(self._process(rec, max_len))

    def _process(self, rec, max_len):
        inp = rec["input_ids"][:max_len]
        inp = [min(i, self.config.vocab_size - 1) for i in inp]
        pad_len = max_len - len(inp)
        padded = inp + [0] * pad_len

        lbl = rec.get("label_ids", inp)[:max_len]
        lbl = [min(i, self.config.vocab_size - 1) for i in lbl]
        lbl_padded = lbl + [0] * (max_len - len(lbl))

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "labels": torch.tensor(lbl_padded, dtype=torch.long),
            "target_id": rec.get("target_id", 0),
            "anchor_ids": rec.get("anchor_ids", []),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _synthetic_dataset(config: ARNLConfig, n_samples: int = 256, seq_len: int = 64):
    """Fast synthetic dataset for smoke tests (random token IDs)."""
    samples = [
        torch.randint(3, config.vocab_size, (seq_len,)).tolist()
        for _ in range(n_samples)
    ]
    return _TokenIdDataset(samples)


# ════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ════════════════════════════════════════════════════════════════

def _save_config(config: ARNLConfig, directory: str):
    os.makedirs(directory, exist_ok=True)
    from dataclasses import asdict
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)


def _load_config(directory: str) -> Optional[ARNLConfig]:
    path = os.path.join(directory, "config.json")
    if not os.path.exists(path):
        return None
    from dataclasses import fields
    with open(path) as f:
        d = json.load(f)
    valid = {fld.name for fld in fields(ARNLConfig)}
    return ARNLConfig(**{k: v for k, v in d.items() if k in valid})


def _state_path(checkpoint_dir: str, phase: int) -> str:
    return os.path.join(checkpoint_dir, f"phase{phase}", "state.json")


def _load_state(checkpoint_dir: str, phase: int) -> dict:
    p = _state_path(checkpoint_dir, phase)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"epoch": 0, "global_step": 0, "best_loss": float("inf"), "done": False}


def _save_state(checkpoint_dir: str, phase: int, state: dict):
    d = os.path.join(checkpoint_dir, f"phase{phase}")
    os.makedirs(d, exist_ok=True)
    with open(_state_path(checkpoint_dir, phase), "w") as f:
        json.dump(state, f, indent=2)


def _phase_dir(checkpoint_dir: str, phase: int) -> str:
    return os.path.join(checkpoint_dir, f"phase{phase}")


def _save_phase1(model: Arnold, checkpoint_dir: str, optimizer=None):
    d = _phase_dir(checkpoint_dir, 1)
    os.makedirs(d, exist_ok=True)
    torch.save(model.system1.state_dict(), os.path.join(d, "system1.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
    print(f"  Saved Phase 1 checkpoint → {d}/system1.pt")


def _load_phase1(model: Arnold, checkpoint_dir: str, optimizer=None):
    d = _phase_dir(checkpoint_dir, 1)
    pt = os.path.join(d, "system1.pt")
    if os.path.exists(pt):
        model.system1.load_state_dict(torch.load(pt, map_location=model.device))
        print(f"  Loaded Phase 1 System 1 weights from {pt}")
    opt_pt = os.path.join(d, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_pt):
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu"))
        print(f"  Loaded Phase 1 optimizer state from {opt_pt}")


def _save_phase23(model: Arnold, phase: int, checkpoint_dir: str, optimizer=None):
    d = _phase_dir(checkpoint_dir, phase)
    os.makedirs(d, exist_ok=True)

    # LoRA weights live inside system1 — save the full state dict
    # (frozen weights will be identical; only LoRA params have changed)
    torch.save(model.system1.state_dict(), os.path.join(d, "system1_lora.pt"))
    torch.save(model.reasoning_head.state_dict(), os.path.join(d, "reasoning_head.pt"))
    torch.save(model.injection_bridge.state_dict(), os.path.join(d, "injection_bridge.pt"))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(d, "optimizer.pt"))

    print(f"  Saved Phase {phase} checkpoint → {d}/")


def _load_phase23(model: Arnold, phase: int, checkpoint_dir: str, optimizer=None):
    d = _phase_dir(checkpoint_dir, phase)

    pt = os.path.join(d, "system1_lora.pt")
    if os.path.exists(pt):
        model.system1.load_state_dict(torch.load(pt, map_location=model.device))
        print(f"  Loaded Phase {phase} System 1 (LoRA) from {pt}")

    pt = os.path.join(d, "reasoning_head.pt")
    if os.path.exists(pt):
        model.reasoning_head.load_state_dict(torch.load(pt, map_location=model.device))
        print(f"  Loaded Phase {phase} Reasoning Head from {pt}")

    pt = os.path.join(d, "injection_bridge.pt")
    if os.path.exists(pt):
        model.injection_bridge.load_state_dict(torch.load(pt, map_location=model.device))
        print(f"  Loaded Phase {phase} W_proj from {pt}")

    opt_pt = os.path.join(d, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_pt):
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu"))
        print(f"  Loaded Phase {phase} optimizer from {opt_pt}")


# ════════════════════════════════════════════════════════════════
# Learning-rate schedule
# ════════════════════════════════════════════════════════════════

def _make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def _lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ════════════════════════════════════════════════════════════════
# Phase 1 — EMLM Pretraining
# ════════════════════════════════════════════════════════════════

def run_phase1(
    model: Arnold,
    loader: DataLoader,
    args,
    checkpoint_dir: str,
    device: str,
):
    """Train System 1 on entity-masked language modelling."""
    print("\n" + "═" * 60)
    print("  Phase 1 — EMLM Pretraining (System 1 only)")
    print("═" * 60)

    state = _load_state(checkpoint_dir, 1)
    if state["done"]:
        print("  Phase 1 already complete — loading weights and skipping.")
        _load_phase1(model, checkpoint_dir)
        return

    model.setup_phase1()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr1, betas=(0.9, 0.95), weight_decay=0.1)

    # Attempt to resume mid-phase
    resume_epoch = state["epoch"]
    resume_step = state["global_step"]
    if resume_epoch > 0 or resume_step > 0:
        print(f"  Resuming Phase 1 from epoch {resume_epoch}, step {resume_step}")
        _load_phase1(model, checkpoint_dir, optimizer=optimizer)

    total_steps = len(loader) * args.epochs1
    scheduler = _make_scheduler(optimizer, args.warmup, total_steps)
    # Fast-forward scheduler to resume_step
    for _ in range(resume_step):
        scheduler.step()

    global_step = resume_step
    best_loss = state["best_loss"]

    for epoch in range(resume_epoch, args.epochs1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            result = model(input_ids, labels=labels)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  [P1] Ep {epoch+1}/{args.epochs1}  "
                    f"step {global_step}  "
                    f"loss {loss.item():.4f}  avg {avg:.4f}  lr {lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / len(loader)
        elapsed = time.time() - epoch_start
        print(f"  [P1] Epoch {epoch+1} done — avg_loss {avg_epoch_loss:.4f}  ({elapsed:.1f}s)")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

        # Save after every epoch
        _save_phase1(model, checkpoint_dir, optimizer=optimizer)
        _save_state(checkpoint_dir, 1, {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_loss": best_loss,
            "done": (epoch + 1) >= args.epochs1,
        })

    _save_state(checkpoint_dir, 1, {
        "epoch": args.epochs1,
        "global_step": global_step,
        "best_loss": best_loss,
        "done": True,
    })
    print(f"  Phase 1 complete. Best loss: {best_loss:.4f}")


# ════════════════════════════════════════════════════════════════
# Phase 2 — Projection Alignment
# ════════════════════════════════════════════════════════════════

def run_phase2(
    model: Arnold,
    loader: DataLoader,
    args,
    checkpoint_dir: str,
    device: str,
):
    """Train W_proj + LoRA adapters + Reasoning Head classifiers."""
    print("\n" + "═" * 60)
    print("  Phase 2 — Projection Alignment (W_proj + LoRA + RH)")
    print("═" * 60)

    state = _load_state(checkpoint_dir, 2)
    if state["done"]:
        print("  Phase 2 already complete — loading weights and skipping.")
        model.apply_lora_adapters()
        _load_phase23(model, 2, checkpoint_dir)
        return

    # Ensure Phase 1 weights are loaded first
    p1_state = _load_state(checkpoint_dir, 1)
    if p1_state["done"]:
        _load_phase1(model, checkpoint_dir)
    else:
        print("  WARNING: Phase 1 is not marked done. Continuing anyway.")

    # Apply LoRA adapters to System 1
    model.apply_lora_adapters()

    # Use full Phase 2 setup (enables RH learnable sub-modules)
    model.setup_phase2_full()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr2, weight_decay=0.01)

    resume_epoch = state["epoch"]
    resume_step = state["global_step"]
    if resume_epoch > 0 or resume_step > 0:
        print(f"  Resuming Phase 2 from epoch {resume_epoch}, step {resume_step}")
        _load_phase23(model, 2, checkpoint_dir, optimizer=optimizer)

    total_steps = len(loader) * args.epochs2
    scheduler = _make_scheduler(optimizer, args.warmup // 2, total_steps)
    for _ in range(resume_step):
        scheduler.step()

    global_step = resume_step
    best_loss = state["best_loss"]

    for epoch in range(resume_epoch, args.epochs2):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Compute injection targets from label sequence
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
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  [P2] Ep {epoch+1}/{args.epochs2}  "
                    f"step {global_step}  "
                    f"loss {loss.item():.4f}  avg {avg:.4f}  lr {lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / len(loader)
        elapsed = time.time() - epoch_start
        print(f"  [P2] Epoch {epoch+1} done — avg_loss {avg_epoch_loss:.4f}  ({elapsed:.1f}s)")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

        _save_phase23(model, 2, checkpoint_dir, optimizer=optimizer)
        _save_state(checkpoint_dir, 2, {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_loss": best_loss,
            "done": (epoch + 1) >= args.epochs2,
        })

    _save_state(checkpoint_dir, 2, {
        "epoch": args.epochs2,
        "global_step": global_step,
        "best_loss": best_loss,
        "done": True,
    })
    print(f"  Phase 2 complete. Best loss: {best_loss:.4f}")


# ════════════════════════════════════════════════════════════════
# Phase 3 — End-to-End Fine-Tuning
# ════════════════════════════════════════════════════════════════

def run_phase3(
    model: Arnold,
    loader: DataLoader,
    args,
    checkpoint_dir: str,
    device: str,
):
    """End-to-end fine-tuning: W_proj + LoRA + Reasoning Head classifiers."""
    print("\n" + "═" * 60)
    print("  Phase 3 — End-to-End Fine-Tuning (W_proj + LoRA + RH)")
    print("═" * 60)

    state = _load_state(checkpoint_dir, 3)
    if state["done"]:
        print("  Phase 3 already complete — loading weights and skipping.")
        model.apply_lora_adapters()
        _load_phase23(model, 3, checkpoint_dir)
        return

    # Load Phase 2 weights first (LoRA must be applied before loading)
    model.apply_lora_adapters()
    p2_state = _load_state(checkpoint_dir, 2)
    if p2_state["done"]:
        _load_phase23(model, 2, checkpoint_dir)
    else:
        print("  WARNING: Phase 2 is not marked done. Continuing from best available.")

    model.setup_phase3()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr3, weight_decay=0.01)

    resume_epoch = state["epoch"]
    resume_step = state["global_step"]
    if resume_epoch > 0 or resume_step > 0:
        print(f"  Resuming Phase 3 from epoch {resume_epoch}, step {resume_step}")
        _load_phase23(model, 3, checkpoint_dir, optimizer=optimizer)

    total_steps = len(loader) * args.epochs3
    scheduler = _make_scheduler(optimizer, args.warmup // 4, total_steps)
    for _ in range(resume_step):
        scheduler.step()

    global_step = resume_step
    best_loss = state["best_loss"]

    for epoch in range(resume_epoch, args.epochs3):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

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
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  [P3] Ep {epoch+1}/{args.epochs3}  "
                    f"step {global_step}  "
                    f"loss {loss.item():.4f}  avg {avg:.4f}  lr {lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / len(loader)
        elapsed = time.time() - epoch_start
        print(f"  [P3] Epoch {epoch+1} done — avg_loss {avg_epoch_loss:.4f}  ({elapsed:.1f}s)")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

        _save_phase23(model, 3, checkpoint_dir, optimizer=optimizer)
        _save_state(checkpoint_dir, 3, {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_loss": best_loss,
            "done": (epoch + 1) >= args.epochs3,
        })

    _save_state(checkpoint_dir, 3, {
        "epoch": args.epochs3,
        "global_step": global_step,
        "best_loss": best_loss,
        "done": True,
    })
    print(f"  Phase 3 complete. Best loss: {best_loss:.4f}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def build_args():
    parser = argparse.ArgumentParser(
        description="Train Arnold (ARNL V1.0) with automatic checkpoint resume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Checkpoint & model ──────────────────────────────────────
    parser.add_argument("--checkpoint-dir", "-c", default="checkpoints",
                        help="Directory to save/load checkpoints (default: checkpoints/)")
    parser.add_argument("--preset", default="arnold_tiny",
                        choices=["arnold_tiny", "arnold_1b", "arnold_3b",
                                 "arnold_7b", "arnold_13b", "arnold_24b"],
                        help="Model size preset (default: arnold_tiny)")

    # ── Phase selection ─────────────────────────────────────────
    parser.add_argument("--phase", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="0=all phases, 1=phase1 only, 2=phase2 only, 3=phase3 only")

    # ── Data ────────────────────────────────────────────────────
    parser.add_argument("--data-config", default=None, metavar="PATH",
                        help="Path to a DataConfig JSON file (HuggingFace datasets + settings). "
                             "Generated with: python scripts/data_config.py --init-config PATH")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic random-token data for all phases (no files needed)")
    parser.add_argument("--phase1-data", default=None,
                        help="Path to plain text file for Phase 1 (one doc per line)")
    parser.add_argument("--phase2-data", default=None,
                        help="Path to JSONL file for Phase 2")
    parser.add_argument("--phase3-data", default=None,
                        help="Path to JSONL file for Phase 3 (defaults to phase2-data)")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length (default: 64)")
    parser.add_argument("--n-synthetic", type=int, default=512,
                        help="Number of synthetic samples per phase (default: 512)")

    # ── Training hyperparameters ────────────────────────────────
    parser.add_argument("--epochs1", type=int, default=3,
                        help="Phase 1 epochs (default: 3)")
    parser.add_argument("--epochs2", type=int, default=2,
                        help="Phase 2 epochs (default: 2)")
    parser.add_argument("--epochs3", type=int, default=1,
                        help="Phase 3 epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr1", type=float, default=3e-4,
                        help="Phase 1 learning rate (default: 3e-4)")
    parser.add_argument("--lr2", type=float, default=1e-4,
                        help="Phase 2 learning rate (default: 1e-4)")
    parser.add_argument("--lr3", type=float, default=5e-5,
                        help="Phase 3 learning rate (default: 5e-5)")
    parser.add_argument("--warmup", type=int, default=200,
                        help="Warmup steps for Phase 1 (default: 200)")
    parser.add_argument("--clip", type=float, default=1.0,
                        help="Gradient clip norm (default: 1.0)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Steps between log lines (default: 50)")

    # ── Device ──────────────────────────────────────────────────
    parser.add_argument("--device", default="auto",
                        help="Device: auto|cpu|cuda|cuda:0 (default: auto)")

    return parser.parse_args()


def main():
    args = build_args()

    # ── Device ──────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── Config ──────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    existing_config = _load_config(args.checkpoint_dir)
    if existing_config is not None:
        print(f"Loaded existing config from {args.checkpoint_dir}/config.json")
        config = existing_config
    else:
        config = getattr(ARNLConfig, args.preset)()
        _save_config(config, args.checkpoint_dir)
        print(f"Created new {args.preset} config → {args.checkpoint_dir}/config.json")

    # ── Model ────────────────────────────────────────────────────
    model = Arnold(config)
    print(model)

    # ── Status summary ───────────────────────────────────────────
    for ph in [1, 2, 3]:
        st = _load_state(args.checkpoint_dir, ph)
        status = "DONE" if st["done"] else f"epoch={st['epoch']} step={st['global_step']}"
        print(f"  Phase {ph}: {status}")

    # ── Determine which phases to run ───────────────────────────
    phases_to_run = [1, 2, 3] if args.phase == 0 else [args.phase]

    # ── Build datasets ───────────────────────────────────────────
    def _make_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    masker = EMLMMasker(config)

    # Load DataConfig if --data-config was supplied
    data_cfg: Optional[DataConfig] = None
    if args.data_config:
        data_cfg = DataConfig.load(args.data_config)
        print(f"  DataConfig loaded from {args.data_config}")
        # DataConfig max_seq_len overrides --seq-len when explicitly set
        if data_cfg.max_seq_len != DataConfig().max_seq_len or args.seq_len == 64:
            args.seq_len = data_cfg.max_seq_len
        # Save a copy into the checkpoint dir so populate_map.py can find it
        dest = os.path.join(args.checkpoint_dir, _DATA_CONFIG_FILENAME)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(args.data_config, dest)
            print(f"  DataConfig copied → {dest}") 

    def _get_loader(phase: int):
        if args.synthetic:
            return _make_loader(_synthetic_dataset(config, args.n_synthetic, args.seq_len))

        # ── DataConfig path (HuggingFace datasets) ───────────────
        if data_cfg is not None:
            if phase == 1:
                ds = build_phase1_dataset(data_cfg, config, masker)
            else:
                ds = build_phase23_dataset(data_cfg, phase, config)
            if ds is not None:
                print(f"  Phase {phase}: {len(ds):,} samples from DataConfig.")
                return _make_loader(ds)
            print(f"  Phase {phase}: DataConfig returned no data — falling back to CLI flags.")

        # ── Legacy CLI flags path ────────────────────────────────
        if phase == 1:
            if args.phase1_data is None:
                print("  --phase1-data not given; using synthetic data for Phase 1.")
                return _make_loader(_synthetic_dataset(config, args.n_synthetic, args.seq_len))
            return _make_loader(
                Phase1TextDataset(args.phase1_data, config, masker, max_len=args.seq_len)
            )
        else:
            data_path = args.phase2_data if phase == 2 else (args.phase3_data or args.phase2_data)
            if data_path is None:
                print(f"  --phase{phase}-data not given; using synthetic data for Phase {phase}.")
                return _make_loader(_synthetic_dataset(config, args.n_synthetic, args.seq_len))
            return _make_loader(
                Phase23Dataset(data_path, config, max_len=args.seq_len)
            )

    # ── Run selected phases ──────────────────────────────────────
    for phase in phases_to_run:
        loader = _get_loader(phase)

        if phase == 1:
            run_phase1(model, loader, args, args.checkpoint_dir, device)
        elif phase == 2:
            run_phase2(model, loader, args, args.checkpoint_dir, device)
        elif phase == 3:
            run_phase3(model, loader, args, args.checkpoint_dir, device)

    print("\n" + "═" * 60)
    print("  Training complete.")
    print(f"  Checkpoints saved to: {os.path.abspath(args.checkpoint_dir)}")
    print("═" * 60)


if __name__ == "__main__":
    main()
