#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for HybridNSVSA-HA.

Loads a pretrained checkpoint and fine-tunes on instruction/chat datasets
in .json or .jsonl format (HuggingFace-style SFT datasets).

Supported dataset formats:
  ─────────────────────────────────────────────────────────────────────
  1. Instruction format (Alpaca-style):
     {"instruction": "...", "input": "...", "output": "..."}
     {"instruction": "...", "output": "..."}

  2. Chat/messages format (ShareGPT / OpenAI style):
     {"conversations": [{"from": "human", "value": "..."},
                         {"from": "gpt",   "value": "..."}]}
     {"messages": [{"role": "user",      "content": "..."},
                   {"role": "assistant", "content": "..."}]}

  3. Simple prompt/completion:
     {"prompt": "...", "completion": "..."}
     {"question": "...", "answer": "..."}

  4. Text-only (pre-formatted):
     {"text": "..."}
  ─────────────────────────────────────────────────────────────────────

Features:
  - Prompt masking: loss only on assistant/output tokens (labels=-100 for input)
  - All stability features from train.py: z-loss, gate entropy, logit softcap
  - Cosine LR with warmup
  - LoRA-style selective freezing (freeze base, train only specified layers)
  - EMA model tracking
  - Gradient accumulation & clipping
  - Automatic train/val split

Usage:
  # Fine-tune from best checkpoint on a JSONL instruction dataset
  python finetune.py --checkpoint checkpoints/best.pt \\
                     --dataset data/sft/my_instructions.jsonl

  # Fine-tune with chat-format dataset, custom LR
  python finetune.py --checkpoint checkpoints/best.pt \\
                     --dataset data/sft/conversations.json \\
                     --max_lr 2e-5 --max_steps 5000

  # Freeze everything except FFN + LM head (parameter-efficient)
  python finetune.py --checkpoint checkpoints/best.pt \\
                     --dataset data/sft/my_data.jsonl \\
                     --trainable_modules ffn,lm_head
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

# Optional HuggingFace datasets support
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig
from nsvsa_ha.tokenizer import BaseTokenizer, load_tokenizer, tokenizer_compatible


# ─── Default paths ───────────────────────────────────────────────────────────

DEFAULT_CUSTOM_TOKENIZER_JSON = "tokenizers/vsa65k_mix/tokenizer.json"
DEFAULT_CUSTOM_TOKENIZER_META = "tokenizers/vsa65k_mix/tokenizer_meta.json"


# ─── Dataset parsing ─────────────────────────────────────────────────────────

def _load_json_or_jsonl(path: Path) -> list[dict]:
    """Load a .json (single array) or .jsonl (one object per line) file."""
    text = path.read_text(encoding="utf-8")

    # Try .json first (single array)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
            if isinstance(data, dict):
                # Some datasets wrap in {"data": [...]} or {"train": [...]}
                for key in ("data", "train", "rows", "instances", "examples"):
                    if key in data and isinstance(data[key], list):
                        return [r for r in data[key] if isinstance(r, dict)]
                return [data]
        except json.JSONDecodeError:
            pass  # Fall through to line-by-line

    # .jsonl: one JSON object per line
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def _load_hf_dataset(name: str) -> list[dict]:
    """Load a HuggingFace dataset by name and return a list of dict rows.

    Returns a list of plain Python dicts combining available splits.
    """
    if load_dataset is None:
        raise RuntimeError(
            "The Python package 'datasets' is required to load HuggingFace datasets. "
            "Install with 'pip install datasets' or pass a local .json/.jsonl file."
        )

    print(f"Loading HuggingFace dataset: {name}")
    ds = load_dataset(name)
    rows: list[dict] = []

    # If we get a DatasetDict, iterate splits; otherwise treat as single Dataset
    if hasattr(ds, "items"):
        # DatasetDict-like
        for split_name, split in ds.items():
            try:
                for ex in split:
                    if isinstance(ex, dict):
                        rows.append(ex)
            except Exception:
                continue
    else:
        # Single Dataset
        for ex in ds:
            if isinstance(ex, dict):
                rows.append(ex)

    print(f"Loaded {len(rows)} rows from HF dataset {name}")
    return rows


def _extract_turns(row: dict) -> list[dict[str, str]]:
    """
    Normalize a dataset row into a list of turns:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Supports:
      - ShareGPT: conversations[].from/value
      - OpenAI:   messages[].role/content
      - Alpaca:   instruction + input + output
      - Simple:   prompt + completion / question + answer
      - Text:     text (single assistant turn)
    """
    # ShareGPT format
    convos = row.get("conversations") or row.get("conversation")
    if convos and isinstance(convos, list):
        turns = []
        for turn in convos:
            role_raw = str(turn.get("from", turn.get("role", ""))).strip().lower()
            content = str(turn.get("value", turn.get("content", ""))).strip()
            if not content:
                continue
            if role_raw in ("human", "user", "prompter"):
                turns.append({"role": "user", "content": content})
            elif role_raw in ("gpt", "assistant", "model", "chatbot", "bot"):
                turns.append({"role": "assistant", "content": content})
            elif role_raw == "system":
                turns.append({"role": "system", "content": content})
        if turns:
            return turns

    # OpenAI messages format
    msgs = row.get("messages")
    if msgs and isinstance(msgs, list):
        turns = []
        for msg in msgs:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role and content:
                turns.append({"role": role, "content": content})
        if turns:
            return turns

    # Alpaca instruction format
    instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
    extra_input = str(row.get("input", "")).strip()
    output = str(
        row.get("output", "")
        or row.get("response", "")
        or row.get("completion", "")
        or row.get("answer", "")
        or row.get("chosen", "")
    ).strip()

    if instruction and output:
        user_text = instruction
        if extra_input:
            user_text += f"\n\n{extra_input}"
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output},
        ]

    # Simple prompt/completion
    prompt = str(row.get("prompt", row.get("question", ""))).strip()
    completion = str(
        row.get("completion", "")
        or row.get("answer", "")
        or row.get("response", "")
        or row.get("output", "")
        or row.get("chosen", "")
    ).strip()

    if prompt and completion:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

    # Plain text — treat as a single assistant turn (unsupervised-style)
    text = str(row.get("text", "")).strip()
    if text:
        return [{"role": "assistant", "content": text}]

    return []


def format_turns(
    turns: list[dict[str, str]],
    tokenizer: BaseTokenizer,
    max_len: int,
    *,
    mask_prompt: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Convert turns into (input_ids, labels) with prompt masking.

    Format:
        [system prompt]
        ### User:
        {user message}
        ### Assistant:
        {assistant message}
        <|endoftext|>

    When mask_prompt=True, labels for user turns and formatting tokens are -100
    (ignored by cross-entropy), so loss is only on assistant-generated tokens.

    Returns:
        input_ids: list of token IDs (truncated to max_len)
        labels:    list of label IDs (-100 for masked positions)
    """
    input_ids: list[int] = []
    labels: list[int] = []
    IGNORE = -100
    eot = tokenizer.eot_token

    for turn in turns:
        role = turn["role"]
        content = turn["content"]

        if role == "system":
            header = "### System:\n"
        elif role == "user":
            header = "### User:\n"
        elif role == "assistant":
            header = "### Assistant:\n"
        else:
            header = f"### {role.capitalize()}:\n"

        header_ids = tokenizer.encode(header)
        content_ids = tokenizer.encode(content)
        sep_ids = tokenizer.encode("\n\n")

        # Should we mask this turn? Everything except assistant is masked.
        is_masked = mask_prompt and role != "assistant"

        # Header tokens: always masked (formatting, not content)
        for tid in header_ids:
            input_ids.append(tid)
            labels.append(IGNORE)

        # Content tokens: masked if user/system, real labels if assistant
        for tid in content_ids:
            input_ids.append(tid)
            labels.append(IGNORE if is_masked else tid)

        # Separator
        for tid in sep_ids:
            input_ids.append(tid)
            labels.append(IGNORE)

    # End of text
    input_ids.append(eot)
    labels.append(eot)

    # Truncate
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return input_ids, labels


def load_sft_dataset(
    path: str,
    tokenizer: BaseTokenizer,
    max_len: int,
    *,
    mask_prompt: bool = True,
    val_split: float = 0.05,
    seed: int = 42,
    max_samples: int = 0,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[list[int], list[int]]]]:
    """
    Load and tokenize an SFT dataset from a .json or .jsonl file.

    Returns (train_examples, val_examples) where each example is
    (input_ids, labels).
    """
    p = Path(path)
    if p.exists():
        raw_rows = _load_json_or_jsonl(p)
    else:
        # Treat non-existing path as a HuggingFace dataset identifier
        raw_rows = _load_hf_dataset(path)
    if not raw_rows:
        raise ValueError(f"No usable rows found in {path}")

    print(f"Loaded {len(raw_rows)} raw rows from {path}")

    # Parse and tokenize
    examples: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for row in raw_rows:
        turns = _extract_turns(row)
        if not turns:
            skipped += 1
            continue
        ids, labs = format_turns(turns, tokenizer, max_len, mask_prompt=mask_prompt)
        # Skip if too short or no trainable tokens
        if len(ids) < 4:
            skipped += 1
            continue
        n_trainable = sum(1 for l in labs if l != -100)
        if n_trainable == 0:
            skipped += 1
            continue
        examples.append((ids, labs))

    if not examples:
        raise ValueError(
            f"No usable examples after tokenization ({skipped} skipped). "
            "Check your dataset format."
        )

    if max_samples > 0 and len(examples) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(examples), size=max_samples, replace=False)
        examples = [examples[i] for i in indices]

    print(f"Tokenized {len(examples)} examples ({skipped} skipped)")

    # Train/val split
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)

    n_val = max(1, int(len(examples) * val_split))
    n_val = min(n_val, 1000)  # cap val set size
    val_indices = set(indices[:n_val])

    train = [examples[i] for i in range(len(examples)) if i not in val_indices]
    val = [examples[i] for i in val_indices]

    print(f"Split: {len(train)} train, {len(val)} val")
    return train, val


# ─── Batching ────────────────────────────────────────────────────────────────

def collate_sft(
    samples: list[tuple[list[int], list[int]]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of (input_ids, labels) to equal length and stack into tensors.

    Returns:
        input_ids: [B, max_len] padded with pad_id
        labels:    [B, max_len] padded with -100
    """
    max_len = max(len(ids) for ids, _ in samples)

    batch_ids = []
    batch_labels = []
    for ids, labs in samples:
        pad_len = max_len - len(ids)
        batch_ids.append(ids + [pad_id] * pad_len)
        batch_labels.append(labs + [-100] * pad_len)

    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_labels, dtype=torch.long),
    )


def make_sft_iterator(
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 42,
    pad_id: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Infinite iterator yielding (input_ids, labels) batches.
    Reshuffles each epoch.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))

    while True:
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_samples = [examples[int(i)] for i in batch_indices]
            yield collate_sft(batch_samples, pad_id=pad_id)


# ─── LR schedule ────────────────────────────────────────────────────────────

def get_finetune_lr(step: int, warmup: int, max_steps: int,
                    max_lr: float, min_lr: float) -> float:
    """Linear warmup → cosine decay."""
    if step < warmup:
        return min_lr + (max_lr - min_lr) * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float, vsa_scale: float = 0.1):
    """Apply lr to all groups, respecting VSA differential rate."""
    for group in optimizer.param_groups:
        if group.get("name") == "vsa_params":
            group["lr"] = lr * vsa_scale
        else:
            group["lr"] = lr


# ─── Parameter group construction ────────────────────────────────────────────

def make_finetune_param_groups(
    model: HybridNSVSA,
    base_lr: float,
    weight_decay: float,
    vsa_lr_scale: float,
    trainable_modules: Optional[list[str]] = None,
) -> list[dict]:
    """
    Build optimizer parameter groups for fine-tuning.

    If trainable_modules is specified, only parameters whose name contains
    one of the listed module names will have requires_grad=True.

    3-way split same as train.py: vsa_decay, other_decay, no_decay.
    """
    # Selective freezing
    if trainable_modules:
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        unfrozen = 0
        for name, p in model.named_parameters():
            if any(mod in name for mod in trainable_modules):
                p.requires_grad_(True)
                unfrozen += 1
        print(f"Selective training: {unfrozen} params unfrozen "
              f"(modules: {trainable_modules})")

    vsa_decay, other_decay, no_decay = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_vsa = "soft_vsa" in name
        is_nodecay = p.dim() < 2 or "norm" in name.lower()

        if is_vsa:
            vsa_decay.append(p)
        elif is_nodecay:
            no_decay.append(p)
        else:
            other_decay.append(p)

    groups = []
    if other_decay:
        groups.append({"params": other_decay, "lr": base_lr, "weight_decay": weight_decay})
    if vsa_decay:
        groups.append({"params": vsa_decay, "lr": base_lr * vsa_lr_scale,
                        "weight_decay": weight_decay, "name": "vsa_params"})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0})

    total_trainable = sum(p.numel() for g in groups for p in g["params"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {total_trainable:,} / {total_params:,} parameters "
          f"({100 * total_trainable / total_params:.1f}%)")

    return groups


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sft(model, val_iter, val_steps: int, device, ctx) -> float:
    """Average cross-entropy over val_steps batches (prompt-masked)."""
    model.eval()
    total_loss, n = 0.0, 0
    for i, (ids, labs) in enumerate(val_iter):
        if i >= val_steps:
            break
        ids, labs = ids.to(device), labs.to(device)
        with ctx:
            out = model(ids, labels=ids)
            # Recompute loss with proper masking
            logits = out["logits"]
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labs[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ─── Checkpoint handling ─────────────────────────────────────────────────────

def load_pretrained(path: Path, model: HybridNSVSA):
    """Load a pretrained checkpoint (weights only, no optimizer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    # Strip _orig_mod. prefix from torch.compile
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    # Handle vocab expansion
    ckpt_vocab = state["embedding.weight"].shape[0]
    model_vocab = model.config.vocab_size
    if ckpt_vocab < model_vocab:
        def _pad(w):
            padded = torch.zeros(model_vocab, w.shape[1], dtype=w.dtype)
            padded[:ckpt_vocab] = w
            return padded
        state = dict(state)
        state["embedding.weight"] = _pad(state["embedding.weight"])
        if "lm_head.weight" in state:
            state["lm_head.weight"] = _pad(state["lm_head.weight"])
        print(f"  Vocab expanded: {ckpt_vocab} → {model_vocab}")
    model.load_state_dict(state)
    if model.config.tie_weights:
        model.lm_head.weight = model.embedding.weight
    return ckpt.get("config", {}), ckpt.get("tokenizer")


def save_finetune_checkpoint(
    path: Path, model, optimizer, step: int, config: HybridNSVSAConfig,
    best_val: float, tokenizer_info: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else {},
        "step": step,
        "config": asdict(config),
        "best_val_loss": best_val,
        "tokenizer": tokenizer_info,
        "finetune": True,
    }, path)
    print(f"  💾 Saved {path}  (step {step})")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SFT Fine-tuning for HybridNSVSA-HA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset format examples:
  Alpaca:    {"instruction": "...", "output": "..."}
  ShareGPT:  {"conversations": [{"from": "human", "value": "..."}, ...]}
  OpenAI:    {"messages": [{"role": "user", "content": "..."}, ...]}
  Simple:    {"prompt": "...", "completion": "..."}
  Text:      {"text": "..."}
""",
    )

    # Required
    p.add_argument("--checkpoint", required=True,
                   help="Path to pretrained checkpoint (.pt)")
    p.add_argument("--dataset", required=True,
                   help="Path to .json or .jsonl SFT dataset")

    # Optional data
    p.add_argument("--val_dataset", default=None,
                   help="Separate validation .json/.jsonl (default: auto-split from train)")
    p.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction of train data to use for validation if no --val_dataset")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Max training samples to use (0=all)")
    p.add_argument("--mask_prompt", action="store_true", default=True,
                   help="Mask prompt tokens (loss only on assistant output)")
    p.add_argument("--no_mask_prompt", action="store_true",
                   help="Compute loss on ALL tokens (including user prompt)")

    # Tokenizer
    p.add_argument("--tokenizer_json", default=DEFAULT_CUSTOM_TOKENIZER_JSON)
    p.add_argument("--tokenizer_meta", default=DEFAULT_CUSTOM_TOKENIZER_META)
    p.add_argument("--tokenizer", default="cl100k_base")

    # Training hyperparameters
    p.add_argument("--max_seq_len", type=int, default=512,
                   help="Maximum sequence length for fine-tuning (lower = less VRAM)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--max_lr", type=float, default=2e-5,
                   help="Peak learning rate (lower than pretraining)")
    p.add_argument("--min_lr", type=float, default=2e-6)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="Weight decay (lower than pretraining to preserve pretrained features)")
    p.add_argument("--vsa_lr_scale", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--dropout", type=float, default=0.05,
                   help="Dropout override for fine-tuning (prevents overfitting on small data)")

    # Selective training (parameter-efficient fine-tuning)
    p.add_argument("--trainable_modules", type=str, default=None,
                   help="Comma-separated module name patterns to train (e.g. 'ffn,lm_head,gate_proj'). "
                        "All other parameters are frozen. None = train everything.")
    p.add_argument("--freeze_embeddings", action="store_true", default=False,
                   help="Freeze embedding layer (prevents catastrophic forgetting on small datasets)")

    # Stability / regularization
    p.add_argument("--z_loss", type=float, default=1e-4,
                   help="Z-loss coefficient (logit explosion prevention)")
    p.add_argument("--gate_entropy_loss", type=float, default=0.001,
                   help="Gate entropy loss (VSA gate utilization)")
    p.add_argument("--entropy_reg", type=float, default=0.0,
                   help="Output entropy regularization")

    # EMA (disabled by default to save VRAM; requires an extra full model copy)
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay rate; 0 = disabled (saves ~300 MB VRAM for a 100M-param model)")
    p.add_argument("--ema_update_interval", type=int, default=10)

    # Eval / logging
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--output_dir", default="checkpoints/sft",
                   help="Output directory for fine-tuned checkpoints")

    # Efficiency
    p.add_argument("--compile", action="store_true", default=False,
                   help="Enable torch.compile (uses dynamic=True to avoid recompile thrashing on variable lengths)")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true", default=False,
                   help="Enable gradient checkpointing to reduce activation memory at the cost of ~30%% slower backward")

    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default=None, metavar="DS_CONFIG",
                   help="Path to DeepSpeed config JSON (e.g. ds_config.json). "
                        "Enables DeepSpeed ZeRO optimization. Omit to use native PyTorch.")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for DeepSpeed distributed launcher (auto-set by deepspeed)")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}  |  AMP dtype: {dtype}")
    print(f"SFT Fine-tuning")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset:    {args.dataset}")

    # ── Tokenizer ────────────────────────────────────────────────────
    if args.tokenizer_json and not Path(args.tokenizer_json).exists():
        print(f"Custom tokenizer not found at {args.tokenizer_json}; falling back to tiktoken")
        args.tokenizer_json = None
        args.tokenizer_meta = None

    tokenizer = load_tokenizer(
        tokenizer_name=args.tokenizer,
        tokenizer_json=args.tokenizer_json,
        tokenizer_meta=args.tokenizer_meta,
    )
    tok_info = tokenizer.info()
    print(f"Tokenizer: {tok_info['backend']}:{tok_info['name']} "
          f"| vocab={tok_info['vocab_size']} | eot={tok_info['eot_token_id']}")

    # ── Load pretrained model ────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Peek at config
    peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = peek.get("config", {})
    del peek

    config = HybridNSVSAConfig(
        d_model=ckpt_cfg.get("d_model", 512),
        num_layers=ckpt_cfg.get("num_layers", 6),
        num_heads=ckpt_cfg.get("num_heads", 8),
        max_seq_len=ckpt_cfg.get("max_seq_len", 2048),
        window_size=ckpt_cfg.get("window_size", 128),
        group_size=ckpt_cfg.get("group_size", 64),
        ffn_ratio=ckpt_cfg.get("ffn_ratio", 4.0),
        dropout=args.dropout,
        gate_init_bias=ckpt_cfg.get("gate_init_bias", 0.0),
        num_kv_heads=ckpt_cfg.get("num_kv_heads", 0),
        qk_norm=ckpt_cfg.get("qk_norm", True),
        learned_vsa_positions=ckpt_cfg.get("learned_vsa_positions", True),
        logit_softcap=ckpt_cfg.get("logit_softcap", 30.0),
        embed_scale=ckpt_cfg.get("embed_scale", True),
        vocab_size=tok_info["vocab_size"],
    )

    model = HybridNSVSA(config).to(device)
    saved_cfg, saved_tok = load_pretrained(ckpt_path, model)
    n_params = model.num_parameters()
    print(f"Model: {n_params:,} parameters ({n_params / 1e6:.1f}M)")
    print(f"Config: d={config.d_model}, L={config.num_layers}, H={config.num_heads}, "
          f"Hkv={config.num_kv_heads}, W={config.window_size}, K={config.group_size}")

    # Gradient checkpointing: trades activations for recomputation, cuts activation memory ~60%
    if args.grad_checkpoint:
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled")
        except AttributeError:
            # Manual fallback: wrap each layer with checkpoint
            import torch.utils.checkpoint as _cp
            _orig_forward = model.layers.forward
            def _ckpt_forward(*a, **kw):
                return _cp.checkpoint(_orig_forward, *a, use_reentrant=False, **kw)
            model.layers.forward = _ckpt_forward
            print("Gradient checkpointing: enabled (manual layer wrap)")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Load and tokenize dataset ────────────────────────────────────
    mask_prompt = args.mask_prompt and not args.no_mask_prompt

    if args.val_dataset:
        train_examples, _ = load_sft_dataset(
            args.dataset, tokenizer, args.max_seq_len,
            mask_prompt=mask_prompt, val_split=0.0,
            max_samples=args.max_samples,
        )
        val_examples, _ = load_sft_dataset(
            args.val_dataset, tokenizer, args.max_seq_len,
            mask_prompt=mask_prompt, val_split=0.0,
        )
    else:
        train_examples, val_examples = load_sft_dataset(
            args.dataset, tokenizer, args.max_seq_len,
            mask_prompt=mask_prompt, val_split=args.val_split,
            max_samples=args.max_samples,
        )

    if not train_examples:
        raise ValueError("No training examples after loading dataset")

    # Dataset stats
    total_tokens = sum(len(ids) for ids, _ in train_examples)
    trainable_tokens = sum(
        sum(1 for l in labs if l != -100) for _, labs in train_examples
    )
    print(f"Training tokens: {total_tokens:,} total, {trainable_tokens:,} trainable "
          f"({100 * trainable_tokens / total_tokens:.1f}%)")

    # ── Build iterators ──────────────────────────────────────────────
    train_iter = make_sft_iterator(
        train_examples, args.batch_size,
        shuffle=True, seed=42, pad_id=tokenizer.eot_token,
    )
    val_iter_factory = lambda: make_sft_iterator(
        val_examples, args.batch_size,
        shuffle=False, seed=1337, pad_id=tokenizer.eot_token,
    )

    # ── Freeze embeddings if requested ───────────────────────────────
    if args.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)
        print("Embeddings frozen")

    # ── Raw model ref before compile ─────────────────────────────────
    raw_model = model

    # ── Optimizer ────────────────────────────────────────────────────
    trainable_modules = None
    if args.trainable_modules:
        trainable_modules = [m.strip() for m in args.trainable_modules.split(",") if m.strip()]

    param_groups = make_finetune_param_groups(
        model, args.max_lr, args.weight_decay, args.vsa_lr_scale,
        trainable_modules=trainable_modules,
    )

    if not param_groups:
        raise ValueError("No trainable parameters! Check --trainable_modules setting.")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.beta1, args.beta2),
        fused=not getattr(args, "deepspeed", None),  # fused conflicts with DeepSpeed
    )

    # ── DeepSpeed ────────────────────────────────────────────────────
    use_deepspeed = getattr(args, "deepspeed", None) is not None
    ds_engine = None

    if use_deepspeed:
        import deepspeed

        # Single-GPU: pre-init torch.distributed if not already done
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.distributed.init_process_group(backend="nccl")

        with open(args.deepspeed) as f:
            ds_config = json.load(f)

        ds_config["gradient_accumulation_steps"] = 1
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["train_batch_size"] = 1
        ds_config["gradient_clipping"] = 0.0
        ds_config["zero_force_ds_cpu_optimizer"] = False
        ds_config["torch_autocast"] = {"enabled": True}

        if dtype == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}
            ds_config["fp16"] = {"enabled": False}
        else:
            ds_config["bf16"] = {"enabled": False}
            ds_config["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}

        ds_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
        )
        model = ds_engine
        raw_model = ds_engine.module
        print(f"DeepSpeed: ZeRO stage {ds_config.get('zero_optimization', {}).get('stage', 0)}, "
              f"CPU offload={'on' if ds_config.get('zero_optimization', {}).get('offload_optimizer', {}).get('device') == 'cpu' else 'off'}")

    # ── EMA ──────────────────────────────────────────────────────────
    ema_model = None
    if args.ema_decay > 0:
        from copy import deepcopy
        ema_model = deepcopy(raw_model).to(device).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"EMA model: decay={args.ema_decay}")

    # ── Compile ──────────────────────────────────────────────────────
    compile_ok = args.compile and not args.no_compile and sys.version_info < (3, 14) and not use_deepspeed
    if use_deepspeed and args.compile and not args.no_compile:
        print("Skipping torch.compile: incompatible with DeepSpeed engine")
    elif compile_ok:
        try:
            # dynamic=True is critical: avoids recompilation on every new sequence length
            print("Compiling model with torch.compile (dynamic=True)...")
            model = torch.compile(model, mode="default", dynamic=True)
        except Exception as exc:
            print(f"Skipping torch.compile: {exc}")
    else:
        print("torch.compile: disabled (pass --compile to enable)")

    # ── Training loop ────────────────────────────────────────────────
    model.train()
    if not use_deepspeed:
        optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    t0 = time.perf_counter()
    tokens_seen = 0
    best_val = float("inf")
    ckpt_dir = Path(args.output_dir)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nFine-tuning for {args.max_steps} steps "
          f"(effective batch = {eff_batch} × ≤{args.max_seq_len} tokens)\n")

    step_iter = tqdm(
        range(args.max_steps),
        total=args.max_steps,
        desc="SFT",
        dynamic_ncols=True,
    )

    for step in step_iter:
        lr = get_finetune_lr(step, args.warmup_steps, args.max_steps,
                             args.max_lr, args.min_lr)
        set_lr(optimizer, lr, args.vsa_lr_scale)

        _micro_loss_sum = 0.0
        for micro in range(args.grad_accum):
            batch_ids, batch_labels = next(train_iter)
            batch_ids = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            with ctx:
                out = model(batch_ids, labels=batch_ids)

                # Recompute loss with prompt masking
                logits = out["logits"]
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = batch_labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ) / args.grad_accum

                # Z-loss
                if args.z_loss > 0.0:
                    _log_z = torch.logsumexp(logits.float(), dim=-1)
                    _z_penalty = (_log_z ** 2).mean()
                    loss = loss + args.z_loss * _z_penalty / args.grad_accum

                # Gate entropy loss
                if args.gate_entropy_loss > 0.0:
                    _ge = torch.zeros(1, device=device, dtype=loss.dtype)
                    _eps_ge = 1e-8
                    for _gl in raw_model.layers.layers:
                        _g = torch.sigmoid(_gl.gate_proj.bias.float())
                        _h = -_g * (_g + _eps_ge).log() - (1 - _g) * (1 - _g + _eps_ge).log()
                        _ge = _ge + _h.mean()
                    loss = loss - args.gate_entropy_loss * _ge / args.grad_accum

                # Entropy regularization
                if args.entropy_reg > 0.0:
                    _log_p = torch.log_softmax(logits.float(), dim=-1)
                    _H = -(_log_p.exp() * _log_p).sum(dim=-1).mean()
                    loss = loss - args.entropy_reg * _H / args.grad_accum

            if use_deepspeed:
                ds_engine.backward(loss)
            else:
                loss.backward()
            _lv = loss.item()
            accum_loss += _lv
            _micro_loss_sum += _lv
            tokens_seen += batch_ids.numel()

        # Gradient clipping
        if args.grad_clip > 0:
            _clip_params = raw_model.parameters() if use_deepspeed else model.parameters()
            grad_norm = nn.utils.clip_grad_norm_(_clip_params, args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        grad_norm_value = float(grad_norm)

        if use_deepspeed:
            ds_engine.step()
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # EMA update
        if ema_model is not None and (step + 1) % args.ema_update_interval == 0:
            with torch.no_grad():
                for p_ema, p_live in zip(ema_model.parameters(), raw_model.parameters()):
                    p_ema.data.mul_(args.ema_decay).add_(p_live.data, alpha=1.0 - args.ema_decay)

        # ── Logging ──────────────────────────────────────────────────
        if (step + 1) % args.log_interval == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = tokens_seen / dt
            avg_loss = accum_loss / args.log_interval
            step_iter.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{lr:.2e}",
                grad=f"{grad_norm_value:.2f}",
                tok_s=f"{tok_per_sec:,.0f}",
            )
            tqdm.write(
                f"step {step + 1:>6d}/{args.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm_value:.2f} | "
                f"{tok_per_sec:,.0f} tok/s"
            )
            accum_loss = 0.0

        # ── Eval ─────────────────────────────────────────────────────
        if (step + 1) % args.eval_interval == 0:
            val_iter = val_iter_factory()
            val_loss = evaluate_sft(model, val_iter, args.eval_steps, device, ctx)

            ema_val_str = ""
            ema_val_loss = None
            if ema_model is not None:
                val_iter_ema = val_iter_factory()
                ema_val_loss = evaluate_sft(ema_model, val_iter_ema, args.eval_steps, device, ctx)
                ema_val_str = f"  ema: {ema_val_loss:.4f}"

            tqdm.write(
                f"  ── val loss: {val_loss:.4f}  (best: {best_val:.4f}){ema_val_str}"
            )

            tracked = min(val_loss, ema_val_loss) if ema_val_loss is not None else val_loss
            if tracked < best_val:
                best_val = tracked
                save_finetune_checkpoint(
                    ckpt_dir / "best_sft.pt",
                    model, optimizer, step + 1, config, best_val, tok_info,
                )
                if ema_model is not None and ema_val_loss is not None and ema_val_loss < val_loss:
                    save_finetune_checkpoint(
                        ckpt_dir / "best_sft_ema.pt",
                        ema_model, None, step + 1, config, ema_val_loss, tok_info,
                    )

        # ── Periodic save ────────────────────────────────────────────
        if (step + 1) % args.save_interval == 0:
            save_finetune_checkpoint(
                ckpt_dir / f"sft_step_{step + 1}.pt",
                model, optimizer, step + 1, config, best_val, tok_info,
            )

    step_iter.close()

    # ── Final save ───────────────────────────────────────────────────
    save_finetune_checkpoint(
        ckpt_dir / "sft_final.pt",
        model, optimizer, args.max_steps, config, best_val, tok_info,
    )

    dt_total = time.perf_counter() - t0
    print(f"\nDone. {args.max_steps} steps in {dt_total / 60:.1f} min. "
          f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
