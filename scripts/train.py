#!/usr/bin/env python3
"""
scripts/train.py — ARNL V1.1 Two-Phase Training Script
========================================================

Trains Arnold through two phases with full checkpoint management.
On every run it inspects the checkpoint directory and resumes from
wherever it left off.

────────────────────────────────────────────────────────────────
V1.1 Changes:
    • Only 2 phases (was 3 in V1.0)
    • Phase 1: EMLM Pretraining — all of System 1
    • Phase 2: LoRA Fine-Tuning — lightweight adapters on System 1,
               SHADE pre-populated before this phase
    • Reasoning Head and W_proj are eliminated
    • No compute_batch_injections — SHADE is off the training loop

Checkpoint layout:
    config.json
    phase1/
        system1.pt
        optimizer.pt
        state.json
    phase2/
        system1_lora.pt
        optimizer.pt
        state.json
────────────────────────────────────────────────────────────────

Usage examples:
    # Full pipeline with synthetic data:
        python scripts/train.py --synthetic --checkpoint-dir checkpoints/

    # Phase 1 only on real text:
        python scripts/train.py --phase 1 --checkpoint-dir checkpoints/ --epochs1 10

    # Phase 2 only (Phase 1 must be done):
        python scripts/train.py --phase 2 --checkpoint-dir checkpoints/

    # Custom preset:
        python scripts/train.py --synthetic --preset arnold_1b --checkpoint-dir checkpoints/1b/
────────────────────────────────────────────────────────────────
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure repo root is on path
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
    """Dataset that returns fixed-length windows of integer token IDs."""

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
    """Loads a plain-text file, applies EMLM masking, produces token windows."""

    def __init__(self, path, config, masker, max_len=512, tokenizer=None):
        self.samples: List[List[int]] = []
        self._build(path, config, masker, max_len, tokenizer)

    def _build(self, path, config, masker, max_len, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]

        masked_lines = masker.mask_batch(lines)

        if tokenizer is None:
            print("[train.py] No tokenizer — using character-level fallback.")
            all_text = " ".join(masked_lines)
            chars = sorted(set(all_text))
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


def _synthetic_dataset(config, n_samples=256, seq_len=64):
    """Synthetic random-token dataset for smoke tests."""
    samples = [
        torch.randint(3, config.vocab_size, (seq_len,)).tolist()
        for _ in range(n_samples)
    ]
    return _TokenIdDataset(samples)


# ════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ════════════════════════════════════════════════════════════════

def _save_config(config, directory):
    os.makedirs(directory, exist_ok=True)
    from dataclasses import asdict
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)


def _load_config(directory):
    path = os.path.join(directory, "config.json")
    if not os.path.exists(path):
        return None
    from dataclasses import fields
    with open(path) as f:
        d = json.load(f)
    valid = {fld.name for fld in fields(ARNLConfig)}
    return ARNLConfig(**{k: v for k, v in d.items() if k in valid})


def _state_path(checkpoint_dir, phase):
    return os.path.join(checkpoint_dir, f"phase{phase}", "state.json")


def _load_state(checkpoint_dir, phase):
    p = _state_path(checkpoint_dir, phase)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"epoch": 0, "global_step": 0, "best_loss": float("inf"), "done": False}


def _save_state(checkpoint_dir, phase, state):
    d = os.path.join(checkpoint_dir, f"phase{phase}")
    os.makedirs(d, exist_ok=True)
    with open(_state_path(checkpoint_dir, phase), "w") as f:
        json.dump(state, f, indent=2)


def _phase_dir(checkpoint_dir, phase):
    return os.path.join(checkpoint_dir, f"phase{phase}")


def _save_phase1(model, checkpoint_dir, optimizer=None):
    d = _phase_dir(checkpoint_dir, 1)
    os.makedirs(d, exist_ok=True)
    torch.save(model.system1.state_dict(), os.path.join(d, "system1.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
    print(f"  Saved Phase 1 → {d}/system1.pt")


def _load_phase1(model, checkpoint_dir, optimizer=None):
    d = _phase_dir(checkpoint_dir, 1)
    pt = os.path.join(d, "system1.pt")
    if os.path.exists(pt):
        model.system1.load_state_dict(torch.load(pt, map_location=model.device, weights_only=True))
        print(f"  Loaded Phase 1 System 1 from {pt}")
    opt_pt = os.path.join(d, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_pt):
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu", weights_only=True))
        print(f"  Loaded Phase 1 optimizer from {opt_pt}")


def _save_phase2(model, checkpoint_dir, optimizer=None):
    d = _phase_dir(checkpoint_dir, 2)
    os.makedirs(d, exist_ok=True)
    torch.save(model.system1.state_dict(), os.path.join(d, "system1_lora.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
    print(f"  Saved Phase 2 → {d}/system1_lora.pt")


def _load_phase2(model, checkpoint_dir, optimizer=None):
    d = _phase_dir(checkpoint_dir, 2)
    pt = os.path.join(d, "system1_lora.pt")
    if os.path.exists(pt):
        model.system1.load_state_dict(torch.load(pt, map_location=model.device, weights_only=True))
        print(f"  Loaded Phase 2 System 1 (LoRA) from {pt}")
    opt_pt = os.path.join(d, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_pt):
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu", weights_only=True))
        print(f"  Loaded Phase 2 optimizer from {opt_pt}")


# ════════════════════════════════════════════════════════════════
# Learning-rate schedule
# ════════════════════════════════════════════════════════════════

def _make_scheduler(optimizer, warmup_steps, total_steps):
    def _lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ════════════════════════════════════════════════════════════════
# Phase 1 — EMLM Pretraining
# ════════════════════════════════════════════════════════════════

def run_phase1(model, loader, args, checkpoint_dir, device):
    """Train System 1 on entity-masked language modelling."""
    print("\n" + "=" * 60)
    print("  Phase 1 — EMLM Pretraining (System 1 only)")
    print("=" * 60)

    state = _load_state(checkpoint_dir, 1)
    if state["done"]:
        print("  Phase 1 already complete — loading weights and skipping.")
        _load_phase1(model, checkpoint_dir)
        return

    model.setup_phase1()
    model.to(device)
    model.param_report()

    use_amp = getattr(args, "amp", True) and device != "cpu"
    grad_accum = max(1, getattr(args, "grad_accum", 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = lambda: torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()
    print(f"  AMP: {'enabled' if use_amp else 'disabled'}  "
          f"grad_accum: {grad_accum}  "
          f"(effective batch = {args.batch_size * grad_accum})")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr1, betas=(0.9, 0.95), weight_decay=0.1)

    resume_epoch = state["epoch"]
    resume_step = state["global_step"]
    if resume_epoch > 0 or resume_step > 0:
        print(f"  Resuming Phase 1 from epoch {resume_epoch}, step {resume_step}")
        _load_phase1(model, checkpoint_dir, optimizer=optimizer)

    total_steps = (len(loader) // grad_accum) * args.epochs1
    scheduler = _make_scheduler(optimizer, args.warmup, total_steps)
    for _ in range(resume_step):
        scheduler.step()

    global_step = resume_step
    best_loss = state["best_loss"]

    for epoch in tqdm(range(resume_epoch, args.epochs1), desc="Phase 1", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader),
                                     desc=f"Epoch {epoch+1}/{args.epochs1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with amp_ctx():
                logits, _ = model(input_ids)
                # Shift: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = Arnold._chunked_cross_entropy(shift_logits, shift_labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum

            is_update = ((batch_idx + 1) % grad_accum == 0 or batch_idx == len(loader) - 1)
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg = epoch_loss / (batch_idx + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [P1] Ep {epoch+1}/{args.epochs1}  "
                        f"step {global_step}  "
                        f"loss {loss.item() * grad_accum:.4f}  avg {avg:.4f}  lr {lr:.2e}"
                    )

        avg_epoch_loss = epoch_loss / len(loader)
        elapsed = time.time() - epoch_start
        print(f"  [P1] Epoch {epoch+1} done — avg_loss {avg_epoch_loss:.4f}  ({elapsed:.1f}s)")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

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
# Phase 2 — LoRA Fine-Tuning
# ════════════════════════════════════════════════════════════════

def run_phase2(model, loader, args, checkpoint_dir, device):
    """Fine-tune System 1 with LoRA adapters on unmasked domain data.

    SHADE should be pre-populated before running this phase.
    LoRA adapters train System 1's receptivity to SHADE's logit
    injection style without corrupting EMLM structure.
    """
    print("\n" + "=" * 60)
    print("  Phase 2 — LoRA Fine-Tuning (System 1 LoRA adapters)")
    print("=" * 60)

    state = _load_state(checkpoint_dir, 2)
    if state["done"]:
        print("  Phase 2 already complete — loading weights and skipping.")
        _load_phase2(model, checkpoint_dir)
        return

    # Load Phase 1 weights first
    p1_state = _load_state(checkpoint_dir, 1)
    if p1_state["done"]:
        _load_phase1(model, checkpoint_dir)
    else:
        print("  WARNING: Phase 1 is not marked done. Continuing anyway.")

    model.setup_phase2()
    model.to(device)
    model.param_report()

    use_amp = getattr(args, "amp", True) and device != "cpu"
    grad_accum = max(1, getattr(args, "grad_accum", 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = lambda: torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable (LoRA) parameters: {n_trainable:,}")
    print(f"  AMP: {'enabled' if use_amp else 'disabled'}  grad_accum: {grad_accum}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr2, weight_decay=0.01)

    resume_epoch = state["epoch"]
    resume_step = state["global_step"]
    if resume_epoch > 0 or resume_step > 0:
        print(f"  Resuming Phase 2 from epoch {resume_epoch}, step {resume_step}")
        _load_phase2(model, checkpoint_dir, optimizer=optimizer)

    total_steps = (len(loader) // grad_accum) * args.epochs2
    scheduler = _make_scheduler(optimizer, args.warmup // 2, total_steps)
    for _ in range(resume_step):
        scheduler.step()

    global_step = resume_step
    best_loss = state["best_loss"]

    for epoch in tqdm(range(resume_epoch, args.epochs2), desc="Phase 2", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader),
                                     desc=f"Epoch {epoch+1}/{args.epochs2}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with amp_ctx():
                logits, _ = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = Arnold._chunked_cross_entropy(shift_logits, shift_labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(loader) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg = epoch_loss / (batch_idx + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [P2] Ep {epoch+1}/{args.epochs2}  "
                        f"step {global_step}  "
                        f"loss {loss.item() * grad_accum:.4f}  avg {avg:.4f}  lr {lr:.2e}"
                    )

        avg_epoch_loss = epoch_loss / len(loader)
        elapsed = time.time() - epoch_start
        print(f"  [P2] Epoch {epoch+1} done — avg_loss {avg_epoch_loss:.4f}  ({elapsed:.1f}s)")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

        _save_phase2(model, checkpoint_dir, optimizer=optimizer)
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
# Main
# ════════════════════════════════════════════════════════════════

def build_args():
    parser = argparse.ArgumentParser(
        description="Train Arnold (ARNL V1.1) — 2-phase training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoint & model
    parser.add_argument("--checkpoint-dir", "-c", default="checkpoints")
    parser.add_argument("--preset", default="arnold_tiny",
                        choices=["arnold_tiny", "arnold_1b", "arnold_3b",
                                 "arnold_7b", "arnold_13b", "arnold_24b"])

    # Phase selection
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="0=all phases, 1=Phase 1 only, 2=Phase 2 only")

    # Data
    parser.add_argument("--data-config", default=None, metavar="PATH",
                        help="Path to DataConfig JSON")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for all phases")
    parser.add_argument("--phase1-data", default="data/corpus.txt")
    parser.add_argument("--phase2-data", default=None,
                        help="Path to JSONL for Phase 2 fine-tuning")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-synthetic", type=int, default=1024)

    # Hyperparameters
    parser.add_argument("--epochs1", type=int, default=10)
    parser.add_argument("--epochs2", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr1", type=float, default=3e-4)
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)

    # Mixed precision
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    # Device
    parser.add_argument("--device", default="auto")

    return parser.parse_args()


def main():
    args = build_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Config
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    existing_config = _load_config(args.checkpoint_dir)
    if existing_config is not None:
        print(f"Loaded existing config from {args.checkpoint_dir}/config.json")
        config = existing_config
    else:
        config = getattr(ARNLConfig, args.preset)()

        # Sync vocab_size from SALT tokenizer
        tok_dir = config.tokenizer_dir
        if args.data_config:
            dc = DataConfig.load(args.data_config)
            if dc.tokenizer_name and dc.tokenizer_name.startswith("salt"):
                parts = dc.tokenizer_name.split(":", 1)
                tok_dir = parts[1] if len(parts) > 1 else "tokenizer/"

        from arnl.salt_tokenizer import SALTTokenizer
        try:
            tok = SALTTokenizer(tok_dir)
            if tok.vocab_size != config.vocab_size:
                print(f"  Updating vocab_size {config.vocab_size} → {tok.vocab_size}")
                config.vocab_size = tok.vocab_size
        except Exception:
            pass

        if args.seq_len > config.max_seq_len:
            config.max_seq_len = args.seq_len

        _save_config(config, args.checkpoint_dir)
        print(f"Created {args.preset} config → {args.checkpoint_dir}/config.json")

    # Model
    model = Arnold(config)
    model.param_report()

    # Status
    for ph in [1, 2]:
        st = _load_state(args.checkpoint_dir, ph)
        status = "DONE" if st["done"] else f"epoch={st['epoch']} step={st['global_step']}"
        print(f"  Phase {ph}: {status}")

    # Phases to run
    phases_to_run = [1, 2] if args.phase == 0 else [args.phase]

    # DataLoader builder
    def _make_loader(dataset):
        return DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, drop_last=True, num_workers=0,
        )

    masker = EMLMMasker()

    # Load DataConfig
    data_cfg: Optional[DataConfig] = None
    if args.data_config:
        data_cfg = DataConfig.load(args.data_config)
        print(f"  DataConfig loaded from {args.data_config}")
        if data_cfg.max_seq_len != DataConfig().max_seq_len or args.seq_len == 64:
            args.seq_len = data_cfg.max_seq_len
        dest = os.path.join(args.checkpoint_dir, _DATA_CONFIG_FILENAME)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(args.data_config, dest)
    elif not args.synthetic:
        data_cfg = DataConfig.example()
        print("  Using built-in default: wikitext/wikitext-2-raw-v1")

    def _get_loader(phase: int):
        if args.synthetic:
            return _make_loader(_synthetic_dataset(config, args.n_synthetic, args.seq_len))

        if data_cfg is not None:
            if phase == 1:
                ds = build_phase1_dataset(data_cfg, config, masker)
            else:
                ds = build_phase23_dataset(data_cfg, phase, config)
            if ds is not None:
                print(f"  Phase {phase}: {len(ds):,} samples from DataConfig.")
                return _make_loader(ds)
            print(f"  Phase {phase}: DataConfig returned no data — falling back.")

        if phase == 1:
            path = args.phase1_data
            if path and os.path.exists(path):
                return _make_loader(
                    Phase1TextDataset(path, config, masker, max_len=args.seq_len)
                )
            if path == "data/corpus.txt":
                print(f"  Default corpus not found; using synthetic data.")
            elif path:
                print(f"  ERROR: Phase 1 data not found at {path}")
                sys.exit(1)
            return _make_loader(_synthetic_dataset(config, args.n_synthetic, args.seq_len))
        else:
            data_path = args.phase2_data
            if data_path and os.path.exists(data_path):
                from scripts.train import Phase1TextDataset as _  # reuse
                return _make_loader(
                    Phase1TextDataset(data_path, config, masker, max_len=args.seq_len)
                )
            if data_path:
                print(f"  ERROR: Phase 2 data not found at {data_path}")
                sys.exit(1)
            print(f"  --phase2-data not given; using synthetic data.")
            return _make_loader(_synthetic_dataset(config, args.n_synthetic, args.seq_len))

    # Run phases
    for phase in phases_to_run:
        loader = _get_loader(phase)
        if phase == 1:
            run_phase1(model, loader, args, args.checkpoint_dir, device)
        elif phase == 2:
            run_phase2(model, loader, args, args.checkpoint_dir, device)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Checkpoints: {os.path.abspath(args.checkpoint_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
