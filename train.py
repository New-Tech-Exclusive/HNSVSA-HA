#!/usr/bin/env python3
"""
Training script for HybridNSVSA-HA language model.

Supports flat (single-stage) and curriculum (multi-stage) training:

  Curriculum stages progressively increase sequence length so the model
  learns local syntax first, then activates VSA for long-range context:

    Stage 1  (60%):  seq_len=512   — fast iteration, attention-dominated
    Stage 2  (25%):  seq_len=1024  — VSA activation, half beyond window
    Stage 3  (15%):  seq_len=2048  — full hierarchical VSA pressure

Features:
  - bf16 mixed precision (AMP)
  - torch.compile (auto-skipped on Python 3.14+)
  - Per-stage cosine LR with linear warmup + intra-stage warmup
  - AdamW with β₂=0.95, weight-decay only on 2-D+ params
  - Gradient accumulation & clipping
  - Differential LR: VSA params at 0.1× base LR
  - Diagnostics: per-layer gate values, per-branch gradient norms
  - Top-K checkpoint retention (best + recent)
  - Pretokenized data at multiple seq_lens
  - Stage-aware resume

Usage:
  # Quick smoke run (~2 min, exercises full curriculum)
  python train.py --smoke

  # Real curriculum training
  python train.py --enable_curriculum

  # Flat training (no curriculum, original behaviour)
  python train.py --no_curriculum

  # Resume from checkpoint
  python train.py --resume checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

# Allow CUDA to release / re-use memory segments dynamically.
# Prevents OOM fragmentation on long runs without extra VRAM overhead.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig
from nsvsa_ha.tokenizer import BaseTokenizer, load_tokenizer, tokenizer_compatible
from nsvsa_ha.modes import MODE_FAST, MODE_REASON, MODE_DEEP, NUM_MODES, DEFAULT_THINK_FRACS


DEFAULT_CUSTOM_TOKENIZER_JSON = "tokenizers/vsa65k_mix/tokenizer.json"
DEFAULT_CUSTOM_TOKENIZER_META = "tokenizers/vsa65k_mix/tokenizer_meta.json"


# ─── Curriculum stage ────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    """One training stage in the curriculum."""
    index: int          # 0-based
    seq_len: int
    batch_size: int
    grad_accum: int
    max_lr: float
    min_lr: float
    warmup_steps: int   # intra-stage warmup
    start_step: int     # inclusive
    end_step: int       # exclusive
    cot_mix: float = 0.0  # fraction of batches from CoT dataset (0=none, 1=all CoT)
    instruction_mix: float = 0.0  # fraction from instruction SFT dataset
    preference_mix: float = 0.0  # fraction from preference-chosen dataset (RLHF-style SFT)


def build_curriculum(args) -> list[CurriculumStage]:
    """Parse curriculum config into resolved CurriculumStage objects."""
    raw = args.curriculum
    total = args.max_steps
    stages: list[CurriculumStage] = []
    cursor = 0
    for i, cfg in enumerate(raw):
        n_steps = max(1, round(total * cfg["pct"]))
        # Clamp last stage to exactly fill total
        if i == len(raw) - 1:
            n_steps = total - cursor
        stages.append(CurriculumStage(
            index=i,
            seq_len=cfg["seq_len"],
            batch_size=cfg["batch_size"],
            grad_accum=cfg["grad_accum"],
            max_lr=cfg["max_lr"],
            min_lr=cfg["min_lr"],
            warmup_steps=cfg.get("intra_warmup", 200),
            start_step=cursor,
            end_step=cursor + n_steps,
            cot_mix=float(cfg.get("cot_mix", 0.0)),
            instruction_mix=float(cfg.get("instruction_mix", 0.0)),
            preference_mix=float(cfg.get("preference_mix", 0.0)),
        ))
        cursor += n_steps
    return stages


def get_stage_for_step(step: int, stages: list[CurriculumStage]) -> CurriculumStage:
    """Return the curriculum stage that owns the given step."""
    for s in stages:
        if s.start_step <= step < s.end_step:
            return s
    return stages[-1]


def make_flat_stage(args) -> list[CurriculumStage]:
    """Single flat stage for non-curriculum training."""
    return [CurriculumStage(
        index=0,
        seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        start_step=0,
        end_step=args.max_steps,
    )]


# ─── Dataset ─────────────────────────────────────────────────────────────────

def build_token_iterator(
    dataset_name: str,
    split: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    batch_size: int,
    *,
    text_field: str = "text",
    streaming: bool = True,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Yields [batch_size, seq_len+1] int64 tensors of packed token IDs.

    Streams from HF datasets, packing documents with <|endoftext|> separator.
    No padding — every token is a real training signal (maximizes efficiency).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    eot = tokenizer.eot_token
    total = seq_len + 1   # +1 for the shifted label

    buf: list[int] = []

    for example in ds:
        text = example.get(text_field, "")
        if not text:
            continue
        tokens = tokenizer.encode(text)
        buf.extend(tokens)
        buf.append(eot)

        while len(buf) >= total * batch_size:
            batch = torch.tensor(
                [buf[i * total : (i + 1) * total] for i in range(batch_size)],
                dtype=torch.long,
            )
            buf = buf[total * batch_size :]
            yield batch


def build_synthetic_iterator(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    *,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Infinite iterator of random token IDs for offline testing.
    Produces batches with simple patterns (repeated n-grams) so the model
    can learn *something* even on synthetic data.
    """
    rng = torch.Generator().manual_seed(seed)
    total = seq_len + 1
    while True:
        # Mix uniform random tokens with some repeating patterns
        batch = torch.randint(0, vocab_size, (batch_size, total), generator=rng)
        # Inject a repeating motif in the first half so loss can actually decrease
        motif_len = torch.randint(3, 8, (1,), generator=rng).item()
        motif = torch.randint(0, vocab_size, (batch_size, motif_len), generator=rng)
        repeats = total // motif_len + 1
        pattern = motif.repeat(1, repeats)[:, :total]
        # Half the batch is patterned, half is random
        half = batch_size // 2
        batch[:half] = pattern[:half]
        yield batch


def _load_pretok_manifest(data_dir: str) -> dict:
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    shard_paths = sorted(data_path.glob("*.npy"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No pretokenized shards found in {data_dir}. "
            "Expected manifest.json or one or more .npy shard files."
        )

    return {
        "shards": [p.name for p in shard_paths],
        "num_shards": len(shard_paths),
    }


def build_pretokenized_iterator(
    data_dir: str,
    batch_size: int,
    *,
    seed: int = 42,
    shuffle: bool = True,
) -> Iterator[torch.Tensor]:
    """
    Iterate over pretokenized fixed-length shards saved as .npy arrays.

    Shard format:
      each .npy is [N, seq_len+1] int32/int64 token IDs.
    """
    data_path = Path(data_dir)
    manifest = _load_pretok_manifest(data_dir)
    shard_names = manifest.get("shards", [])
    if not shard_names:
        raise RuntimeError(f"No shards listed in {data_dir}/manifest.json")

    rng = np.random.default_rng(seed)

    while True:
        shard_order = np.arange(len(shard_names))
        if shuffle:
            rng.shuffle(shard_order)

        for shard_idx in shard_order:
            shard_entry = Path(shard_names[int(shard_idx)])
            shard_file = shard_entry if shard_entry.is_absolute() else (data_path / shard_entry)
            # Load fully into RAM — avoids page-fault overhead on random row access.
            arr = np.load(shard_file)
            if arr.ndim != 2:
                raise ValueError(f"Shard {shard_file} must be rank-2 [N, T], got {arr.shape}")

            num_rows = arr.shape[0]
            if num_rows < batch_size:
                continue

            row_idx = np.arange(num_rows)
            if shuffle:
                rng.shuffle(row_idx)

            max_start = num_rows - batch_size + 1
            for start in range(0, max_start, batch_size):
                idx = row_idx[start : start + batch_size]
                batch_np = np.asarray(arr[idx], dtype=np.int64)
                yield torch.from_numpy(batch_np)


def _blend_iterators(
    primary: Iterator[torch.Tensor],
    secondary: Iterator[torch.Tensor],
    primary_frac: float,
    rng: np.random.Generator,
) -> Iterator[torch.Tensor]:
    """Blend two iterators: yield from primary with probability primary_frac."""
    while True:
        if rng.random() < primary_frac:
            yield next(primary)
        else:
            yield next(secondary)


def build_cot_iterator(
    cot_path: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    batch_size: int,
    *,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Yields [batch_size, seq_len+1] batches from a CoT text file.

    Each non-blank, non-comment line is a complete example already formatted
    with a leading mode token and embedded <|think|> ... <|/think|> spans::

        <|reason|> Question <|think|> step-by-step reasoning... <|/think|> Answer.
        <|deep|> Hard question <|think|> detailed analysis... <|/think|> Answer.

    Lines are tokenized and packed end-to-end with <|endoftext|> separators,
    matching the main training data format.  The example order is shuffled on
    every epoch so each packed window contains a varied neighbourhood.

    Because mode tokens and think tokens are already embedded in the text,
    the training loop must NOT call sample_mode_tokens() on CoT batches.
    """
    path = Path(cot_path)
    if not path.exists():
        raise FileNotFoundError(f"CoT dataset not found: {cot_path}")
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not lines:
        raise ValueError(f"CoT dataset {cot_path} contains no usable lines")

    rng = np.random.default_rng(seed)
    eot = tokenizer.eot_token
    total = seq_len + 1

    buf: list[int] = []
    while True:
        order = np.arange(len(lines))
        rng.shuffle(order)
        for idx in order:
            tokens = tokenizer.encode(lines[int(idx)])
            buf.extend(tokens)
            buf.append(eot)
            while len(buf) >= total * batch_size:
                batch = torch.tensor(
                    [buf[i * total : (i + 1) * total] for i in range(batch_size)],
                    dtype=torch.long,
                )
                buf = buf[total * batch_size :]
                yield batch


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def build_instruction_iterator(
    instruction_path: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    batch_size: int,
    *,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Yields [batch_size, seq_len+1] batches from an instruction dataset.

    Supports:
      - .jsonl with fields like instruction/prompt/input and output/response/answer
      - plain text (one preformatted instruction-response example per line)
    """
    path = Path(instruction_path)
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {instruction_path}")

    examples: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for obj in _load_jsonl(path):
            instr = str(
                obj.get("instruction")
                or obj.get("prompt")
                or obj.get("question")
                or ""
            ).strip()
            extra = str(obj.get("input") or "").strip()
            resp = str(
                obj.get("output")
                or obj.get("response")
                or obj.get("answer")
                or obj.get("chosen")
                or ""
            ).strip()
            if not instr or not resp:
                continue
            if extra:
                text = f"### Instruction:\n{instr}\n\n### Input:\n{extra}\n\n### Response:\n{resp}"
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
            examples.append(text)
    else:
        examples = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]

    if not examples:
        raise ValueError(f"Instruction dataset {instruction_path} contains no usable examples")

    rng = np.random.default_rng(seed)
    eot = tokenizer.eot_token
    total = seq_len + 1
    buf: list[int] = []

    while True:
        order = np.arange(len(examples))
        rng.shuffle(order)
        for idx in order:
            tokens = tokenizer.encode(examples[int(idx)])
            buf.extend(tokens)
            buf.append(eot)
            while len(buf) >= total * batch_size:
                batch = torch.tensor(
                    [buf[i * total : (i + 1) * total] for i in range(batch_size)],
                    dtype=torch.long,
                )
                buf = buf[total * batch_size :]
                yield batch


def build_preference_iterator(
    preference_path: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    batch_size: int,
    *,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Yields [batch_size, seq_len+1] batches from preference data.

    This uses only the preferred (chosen) response as supervised signal,
    which is an RLHF-style approximation without policy optimization.
    """
    path = Path(preference_path)
    if not path.exists():
        raise FileNotFoundError(f"Preference dataset not found: {preference_path}")

    examples: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for obj in _load_jsonl(path):
            prompt = str(
                obj.get("prompt")
                or obj.get("instruction")
                or obj.get("question")
                or ""
            ).strip()
            chosen = str(
                obj.get("chosen")
                or obj.get("output")
                or obj.get("response")
                or obj.get("answer")
                or ""
            ).strip()
            if not chosen:
                continue
            if prompt:
                text = f"### Prompt:\n{prompt}\n\n### Preferred Response:\n{chosen}"
            else:
                text = chosen
            examples.append(text)
    else:
        examples = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]

    if not examples:
        raise ValueError(f"Preference dataset {preference_path} contains no usable examples")

    rng = np.random.default_rng(seed)
    eot = tokenizer.eot_token
    total = seq_len + 1
    buf: list[int] = []

    while True:
        order = np.arange(len(examples))
        rng.shuffle(order)
        for idx in order:
            tokens = tokenizer.encode(examples[int(idx)])
            buf.extend(tokens)
            buf.append(eot)
            while len(buf) >= total * batch_size:
                batch = torch.tensor(
                    [buf[i * total : (i + 1) * total] for i in range(batch_size)],
                    dtype=torch.long,
                )
                buf = buf[total * batch_size :]
                yield batch


def _mix_iterators(
    iterators: list[Iterator[torch.Tensor]],
    probs: list[float],
    rng: np.random.Generator,
) -> Iterator[torch.Tensor]:
    """Sample from multiple iterators using categorical probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    while True:
        i = int(rng.choice(len(iterators), p=p))
        yield next(iterators[i])


# ─── Learning-rate schedule ──────────────────────────────────────────────────

def get_lr(step: int, stage: CurriculumStage) -> float:
    """
    Per-stage linear warmup → cosine decay.

    Each stage has its own warmup and cosine decay schedule relative to its
    own start/end bounds.
    """
    local_step = step - stage.start_step
    stage_len = stage.end_step - stage.start_step

    # Warmup phase
    if local_step < stage.warmup_steps:
        return stage.max_lr * (local_step + 1) / stage.warmup_steps

    # Past end (shouldn't happen but be safe)
    if local_step >= stage_len:
        return stage.min_lr

    # Cosine decay
    progress = (local_step - stage.warmup_steps) / max(1, stage_len - stage.warmup_steps)
    return stage.min_lr + 0.5 * (stage.max_lr - stage.min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float, vsa_scale: float = 0.1):
    """Apply lr to all groups, respecting VSA differential rate."""
    for group in optimizer.param_groups:
        if group.get("name") == "vsa_params":
            group["lr"] = lr * vsa_scale
        else:
            group["lr"] = lr


# ─── Weight-decay parameter separation ──────────────────────────────────────

def make_param_groups(
    model: HybridNSVSA,
    base_lr: float,
    weight_decay: float,
    vsa_lr_scale: float,
) -> list[dict]:
    """
    3-way split:
      1. VSA params (decay + query_proj): low LR, with weight decay
      2. Other ≥2-D params (Linear weights, Embedding): base LR, with weight decay
      3. 1-D params (biases, LayerNorm, vsa_gate): base LR, NO weight decay

    This avoids decaying norms/biases (hurts training) and keeps VSA
    parameters on a gentler learning trajectory.
    """
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

    return [
        {"params": other_decay, "lr": base_lr,                   "weight_decay": weight_decay},
        {"params": vsa_decay,   "lr": base_lr * vsa_lr_scale,    "weight_decay": weight_decay, "name": "vsa_params"},
        {"params": no_decay,    "lr": base_lr,                   "weight_decay": 0.0},
    ]


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def log_gate_values(model: HybridNSVSA) -> str:
    """Return a compact string of per-layer VSA gate means."""
    parts = []
    layers = model.layers.layers  # nn.ModuleList of HybridNSVSALayer
    for i, layer in enumerate(layers):
        gate_mean = torch.sigmoid(layer.vsa_gate).mean().item()
        parts.append(f"L{i}={gate_mean:.3f}")
    return "  gates: " + "  ".join(parts)


def compute_branch_grad_norms(model: HybridNSVSA) -> dict[str, float]:
    """Compute L2 gradient norms for attention, VSA, FFN parameter groups."""
    attn_sq, vsa_sq, ffn_sq, other_sq = 0.0, 0.0, 0.0, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = p.grad.data.norm(2).item() ** 2
        if "local_attn" in name:
            attn_sq += g2
        elif "soft_vsa" in name or "vsa_gate" in name or "vsa_out_proj" in name or "norm_vsa" in name:
            vsa_sq += g2
        elif "ffn" in name:
            ffn_sq += g2
        else:
            other_sq += g2
    return {
        "attn": attn_sq ** 0.5,
        "vsa": vsa_sq ** 0.5,
        "ffn": ffn_sq ** 0.5,
        "other": other_sq ** 0.5,
    }


# ─── Checkpoint management ───────────────────────────────────────────────────

def _ledger_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "checkpoints.json"


def _load_ledger(ckpt_dir: Path) -> dict:
    lp = _ledger_path(ckpt_dir)
    if lp.exists():
        return json.loads(lp.read_text())
    return {}


def _save_ledger(ckpt_dir: Path, ledger: dict):
    lp = _ledger_path(ckpt_dir)
    lp.write_text(json.dumps(ledger, indent=2))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    config: HybridNSVSAConfig,
    best_val: float,
    tokenizer_info: dict | None = None,
    stage_idx: int = 0,
    curriculum_config: list | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "step": step,
        "config": asdict(config),
        "best_val_loss": best_val,
        "tokenizer": tokenizer_info,
        "stage_idx": stage_idx,
        "curriculum": curriculum_config,
    }, path)
    print(f"  💾 Saved {path}  (step {step}, stage {stage_idx})")


def _expand_vocab_in_state(state: dict, model) -> dict:
    """
    If the saved embedding weights are smaller than the model's current
    vocab_size, zero-pad the embedding.weight (and lm_head.weight if
    present) so load_state_dict can succeed.  New token rows stay zero.
    """
    model_vocab = model.config.vocab_size
    ckpt_vocab = state["embedding.weight"].shape[0]
    if ckpt_vocab == model_vocab:
        return state
    if ckpt_vocab > model_vocab:
        raise ValueError(
            f"Checkpoint vocab ({ckpt_vocab}) > model vocab ({model_vocab}). "
            "Either bump vocab_size or use the correct checkpoint."
        )
    def _pad(w: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(model_vocab, w.shape[1], dtype=w.dtype)
        padded[:ckpt_vocab] = w
        return padded
    state = dict(state)  # shallow copy — don't mutate original
    state["embedding.weight"] = _pad(state["embedding.weight"])
    if "lm_head.weight" in state:
        state["lm_head.weight"] = _pad(state["lm_head.weight"])
    print(f"  vocab expanded: {ckpt_vocab} → {model_vocab} (new tokens zero-initialised)")
    return state


def load_checkpoint(path: Path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    # Strip _orig_mod. prefix added by torch.compile when saving
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    # Expand vocabulary if tokenizer has grown since the checkpoint was saved
    vocab_expanded = state["embedding.weight"].shape[0] < model.config.vocab_size
    state = _expand_vocab_in_state(state, model)
    model.load_state_dict(state)
    # Re-tie lm_head → embedding after load_state_dict (which breaks the tie)
    if model.config.tie_weights:
        model.lm_head.weight = model.embedding.weight
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            if vocab_expanded:
                print(f"  WARNING: optimizer state not restored after vocab expansion "
                      f"({exc}).  Optimizer starts fresh for embedding/lm_head.")
            else:
                raise
        else:
            # After load_state_dict the moment tensors (exp_avg, exp_avg_sq) may
            # have the OLD vocab size while the model parameters have grown.
            # Fused AdamW requires them to match — pad any mismatched moments now.
            if vocab_expanded:
                fixed = 0
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p not in optimizer.state:
                            continue
                        pstate = optimizer.state[p]
                        for key in ("exp_avg", "exp_avg_sq"):
                            m = pstate.get(key)
                            if m is None or m.shape == p.shape:
                                continue
                            if m.ndim >= 1 and m.shape[0] < p.shape[0]:
                                # Pad first dimension (vocab rows) with zeros
                                new_m = torch.zeros(p.shape, dtype=m.dtype, device=m.device)
                                idx = (slice(m.shape[0]),) + (slice(None),) * (m.ndim - 1)
                                new_m[idx] = m
                                pstate[key] = new_m
                                fixed += 1
                if fixed:
                    print(f"  optimizer moments padded: {fixed} tensor(s) grown to match new vocab")
    return (
        ckpt.get("step", 0),
        ckpt.get("best_val_loss", float("inf")),
        ckpt.get("tokenizer"),
        ckpt.get("stage_idx", 0),
    )


def update_ledger(ckpt_dir: Path, filename: str, step: int,
                  val_loss: float, stage_idx: int):
    """Record a checkpoint entry in the ledger."""
    ledger = _load_ledger(ckpt_dir)
    ledger[filename] = {
        "step": step,
        "val_loss": val_loss,
        "stage": stage_idx,
        "timestamp": time.time(),
    }
    _save_ledger(ckpt_dir, ledger)


def manage_checkpoints(ckpt_dir: Path, keep_best_k: int = 3, keep_recent_k: int = 2):
    """
    Prune old checkpoints, keeping:
      - The keep_best_k with lowest val_loss
      - The keep_recent_k most recent by step number
      - 'best.pt' and 'final.pt' are always kept
    """
    ledger = _load_ledger(ckpt_dir)
    if not ledger:
        return

    # Never delete these
    protected = {"best.pt", "final.pt"}

    # Separate step_*.pt entries from others
    step_entries = {
        f: info for f, info in ledger.items()
        if f.startswith("step_") and f.endswith(".pt")
    }
    if not step_entries:
        return

    # Best by val loss
    by_loss = sorted(step_entries.items(), key=lambda x: x[1].get("val_loss", float("inf")))
    keep_best = {f for f, _ in by_loss[:keep_best_k]}

    # Most recent by step
    by_step = sorted(step_entries.items(), key=lambda x: x[1].get("step", 0), reverse=True)
    keep_recent = {f for f, _ in by_step[:keep_recent_k]}

    keep = protected | keep_best | keep_recent

    removed = 0
    for filename in list(step_entries.keys()):
        if filename not in keep:
            fp = ckpt_dir / filename
            if fp.exists():
                fp.unlink()
                removed += 1
            del ledger[filename]

    if removed:
        _save_ledger(ckpt_dir, ledger)
        tqdm.write(f"  🗑  Pruned {removed} old checkpoint(s), kept {len(keep & set(step_entries))} step_*.pt")


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_iter, val_steps: int, device, ctx) -> float:
    """Average cross-entropy over val_steps batches."""
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(val_iter):
        if i >= val_steps:
            break
        batch = batch.to(device)
        with ctx:
            out = model(batch[:, :-1], labels=batch[:, :-1])
        loss = out["loss"]
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ─── Mode-conditioned training helpers ───────────────────────────────────────

def sample_mode_tokens(
    batch: torch.Tensor,
    mode_token_ids: dict[int, int],
    mode_mix: tuple[float, float, float] = (0.50, 0.30, 0.20),
    rng: np.random.Generator | None = None,
    think_token_id: int | None = None,
    end_think_token_id: int | None = None,
    think_fracs: dict[int, float] | None = None,
) -> torch.Tensor:
    """
    Prepend a mode control token to each sequence and optionally inject
    a think block based on the mode's thinking level:

        fast   → [fast_tok] [seq …]                      (no thinking)
        reason → [reason_tok] <|think|> [seq[:t]] <|/think|> [seq[t:L-2]]
        deep   → [deep_tok]  <|think|> [seq[:t]] <|/think|> [seq[t:L-2]]

    The think block length is proportional to the mode's think_frac:
        fast=0%, reason=~25%, deep=~50% of the sequence.

    All output rows have the same length L+1.

    Args:
        batch:              [B, L] token IDs.
        mode_token_ids:     {mode_id: token_id} mapping.
        mode_mix:           (p_fast, p_reason, p_deep) probabilities.
        rng:                Optional numpy RNG for reproducibility.
        think_token_id:     ID of <|think|> token, or None to skip think blocks.
        end_think_token_id: ID of <|/think|> token, or None to skip think blocks.
        think_fracs:        {mode_id: fraction} of tokens devoted to thinking.
                            Defaults to DEFAULT_THINK_FRACS.

    Returns:
        [B, L+1] tensor with mode token (and optional think block) prepended.
    """
    B, L = batch.shape
    if rng is None:
        rng = np.random.default_rng()
    if think_fracs is None:
        think_fracs = DEFAULT_THINK_FRACS

    modes = [MODE_FAST, MODE_REASON, MODE_DEEP]
    probs = np.array(mode_mix, dtype=np.float64)
    probs /= probs.sum()
    sampled_modes = rng.choice(modes, size=B, p=probs)

    use_think = (
        think_token_id is not None
        and end_think_token_id is not None
        and L >= 8
    )

    result_rows: list[torch.Tensor] = []
    for i, mode in enumerate(sampled_modes):
        mode_tok = mode_token_ids[int(mode)]
        row = batch[i]  # [L]

        frac = think_fracs.get(int(mode), 0.0)

        if use_think and frac > 0.0:
            # Sample think-block length around the target fraction
            # with ±10% jitter, clamped to [2, L-4]
            lo = max(2, int((frac - 0.10) * L))
            hi = max(lo + 1, int((frac + 0.10) * L))
            hi = min(hi, L - 4)
            lo = min(lo, hi - 1)
            t = int(rng.integers(lo, hi))
            # Resulting length: 2 + t + 1 + (L - t - 2) = L + 1
            head = torch.tensor(
                [mode_tok, think_token_id],
                dtype=batch.dtype, device=batch.device,
            )
            sep = torch.tensor(
                [end_think_token_id],
                dtype=batch.dtype, device=batch.device,
            )
            row_out = torch.cat([head, row[:t], sep, row[t:L - 2]], dim=0)
        else:
            row_out = torch.cat([
                torch.tensor([mode_tok], dtype=batch.dtype, device=batch.device),
                row,
            ], dim=0)  # [L+1]

        result_rows.append(row_out)

    return torch.stack(result_rows, dim=0)  # [B, L+1]


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HybridNSVSA-HA")

    # Data
    p.add_argument("--dataset",      default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset name (streaming)")
    p.add_argument("--text_field",   default="text")
    p.add_argument("--split",        default="train")
    p.add_argument("--val_dataset",  default=None,
                   help="Separate val dataset (default: same as train, reshuffled)")
    p.add_argument("--data_mode", choices=["streaming", "pretokenized", "synthetic"],
                   default="streaming",
                   help="Data source mode for training/eval batches")
    p.add_argument("--pretokenized_train_dir", default="data/pretokenized/train",
                   help="Directory containing pretokenized train shards (flat mode)")
    p.add_argument("--pretokenized_val_dir", default="data/pretokenized/val",
                   help="Directory containing pretokenized val shards (flat mode)")
    p.add_argument("--pretokenized_dir_pattern", default="data/pretokenized/{split}_{seq_len}",
                   help="Template for curriculum pretokenized dirs ({split}, {seq_len} replaced)")
    p.add_argument("--pretokenized_shuffle", action="store_true", default=True,
                   help="Shuffle rows/shards in pretokenized mode")
    p.add_argument("--no_pretokenized_shuffle", action="store_true",
                   help="Disable shuffling for pretokenized mode")
    p.add_argument("--cot_dataset", default="data/cot/cot_examples.txt",
                   help="Path to CoT text file for think-token supervision (Stage 3)")
    p.add_argument("--instruction_dataset", default="data/instruction/high_quality_instruction.jsonl",
                   help="Path to instruction-tuning dataset (.jsonl or .txt)")
    p.add_argument("--preference_dataset", default="data/preference/high_quality_preference.jsonl",
                   help="Path to preference dataset (.jsonl or .txt), using chosen responses")

    # Model
    p.add_argument("--d_model",      type=int, default=512)
    p.add_argument("--num_layers",   type=int, default=12)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--window_size",  type=int, default=128)
    p.add_argument("--group_size",   type=int, default=64)
    p.add_argument("--ffn_ratio",    type=float, default=4.0)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--max_seq_len",  type=int, default=1024)
    p.add_argument("--vocab_size",   type=int, default=0,
                   help="0 = auto-detect from tokenizer")
    p.add_argument("--tokenizer",    default="cl100k_base",
                   help="tiktoken encoding name (used when --tokenizer_json is not set)")
    p.add_argument("--tokenizer_json", default=DEFAULT_CUSTOM_TOKENIZER_JSON,
                   help="Path to custom HF tokenizers tokenizer.json")
    p.add_argument("--tokenizer_meta", default=DEFAULT_CUSTOM_TOKENIZER_META,
                   help="Path to tokenizer metadata JSON (optional)")
    p.add_argument("--strict_tokenizer_match", action="store_true",
                   help="Fail resume if checkpoint tokenizer metadata mismatches runtime tokenizer")

    # Training
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--grad_accum",   type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max_steps",    type=int, default=50_000)
    p.add_argument("--max_lr",       type=float, default=3e-4)
    p.add_argument("--min_lr",       type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--vsa_lr_scale", type=float, default=0.1,
                   help="VSA params get lr * this scale")
    p.add_argument("--vsa_grad_scale", type=float, default=1.0,
                   help="Gradient multiplier for VSA params (counteracts EMA attenuation)")
    p.add_argument("--gate_init_bias", type=float, default=0.0,
                   help="Initial bias for vsa_gate; sigmoid(bias) = starting VSA contribution")
    p.add_argument("--grad_clip",    type=float, default=0.5)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.90,
                   help="Lower β₂ (0.90 vs 0.999) = faster adaptation, natural damper against gradient spikes")
    p.add_argument("--reason_warmup_steps", type=int, default=0,
                   help="(deprecated) No longer used — kept for config compat")
    p.add_argument("--mode_mix", type=str, default="0.50,0.30,0.20",
                   help="Mode sampling probabilities: p_fast,p_reason,p_deep (comma-separated)")
    p.add_argument("--enable_mode_training", action="store_true", default=True,
                   help="Prepend mode control tokens and inject think blocks per mode")
    p.add_argument("--no_mode_training", action="store_true",
                   help="Disable mode-conditioned training")

    # Eval / logging
    p.add_argument("--eval_interval",  type=int, default=500)
    p.add_argument("--eval_steps",     type=int, default=50)
    p.add_argument("--log_interval",   type=int, default=10)
    p.add_argument("--save_interval",  type=int, default=5000)
    p.add_argument("--checkpoint_dir", default="checkpoints")

    # Checkpoint management
    p.add_argument("--keep_best_k",    type=int, default=3,
                   help="Keep K checkpoints with lowest val loss")
    p.add_argument("--keep_recent_k",  type=int, default=2,
                   help="Keep K most recent periodic checkpoints")

    # Curriculum
    p.add_argument("--enable_curriculum", action="store_true", default=False,
                   help="Enable multi-stage curriculum training")
    p.add_argument("--no_curriculum", action="store_true",
                   help="Force flat single-stage training")

    # Diagnostics
    p.add_argument("--log_gate_values", action="store_true", default=False,
                   help="Log per-layer VSA gate means at eval time")
    p.add_argument("--log_branch_grad_norms", action="store_true", default=False,
                   help="Log attention/VSA/FFN gradient norms every log_interval")

    # Efficiency
    p.add_argument("--compile",      action="store_true", default=True,
                   help="torch.compile the model (default: on)")
    p.add_argument("--no_compile",   action="store_true")

    # Resume
    p.add_argument("--resume",       default=None, help="Checkpoint path to resume from")

    # Quick test
    p.add_argument("--smoke",        action="store_true",
                   help="Tiny config for fast curriculum validation (~2 min)")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic data (no network required)")

    # Regularization
    p.add_argument("--entropy_reg", type=float, default=0.0,
                   help="Entropy regularization coefficient: subtracts α·H(p) from loss "
                        "to penalize vocabulary collapse (typical: 0.005–0.02)")
    p.add_argument("--embed_reg", type=float, default=0.0,
                   help="Embedding weight RMS regularization coefficient "
                        "to prevent embedding norm collapse/explosion")

    # EMA model weights
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay for model weight averaging. 0 = disable. "
                        "Typical: 0.9999. Smoother eval + better generation quality.")
    p.add_argument("--ema_update_interval", type=int, default=10,
                   help="Update EMA weights every N optimizer steps")

    # VSA stability
    p.add_argument("--spectral_norm_vsa", action="store_true", default=False,
                   help="Apply spectral normalization to VSA Linear layers "
                        "to prevent vanishing/exploding gradient paths")

    # Optional central config defaults
    try:
        from training_config import DEFAULT_TRAINING_CONFIG
        p.set_defaults(**DEFAULT_TRAINING_CONFIG)
    except Exception:
        pass

    return p.parse_args()


def main():
    args = parse_args()

    # ── Smoke-test override ──────────────────────────────────────────
    if args.smoke:
        args.d_model     = 128
        args.num_layers  = 3
        args.num_heads   = 4
        args.max_seq_len = 256
        args.window_size = 32
        args.group_size  = 16
        args.batch_size  = 4
        args.grad_accum  = 1
        args.max_steps   = 32768 * 3   # 32,768 per stage
        args.warmup_steps = 5
        args.eval_interval = 15
        args.eval_steps  = 3
        args.log_interval = 5
        args.save_interval = 30
        args.synthetic   = True
        args.data_mode   = "synthetic"
        args.log_gate_values   = True
        args.log_branch_grad_norms = True
        # Enable curriculum unless explicitly overridden with --no_curriculum
        if not args.no_curriculum:
            args.enable_curriculum = True
            # Micro-curriculum for smoke test
            args.curriculum = [
                {"seq_len": 64,  "pct": 0.3333, "batch_size": 4, "grad_accum": 1,
                 "max_lr": 3e-4, "min_lr": 3e-5, "intra_warmup": 5},
                {"seq_len": 128, "pct": 0.3333, "batch_size": 2, "grad_accum": 2,
                 "max_lr": 2e-4, "min_lr": 2e-5, "intra_warmup": 3},
                {"seq_len": 256, "pct": 0.3334, "batch_size": 2, "grad_accum": 2,
                 "max_lr": 1.5e-4, "min_lr": 1.5e-5, "intra_warmup": 3},
            ]

    if args.synthetic:
        args.data_mode = "synthetic"

    # Resolve curriculum flag
    use_curriculum = getattr(args, "enable_curriculum", False) and not getattr(args, "no_curriculum", False)
    # Need curriculum list
    if use_curriculum and not hasattr(args, "curriculum"):
        print("WARNING: --enable_curriculum set but no curriculum config found. Using flat training.")
        use_curriculum = False

    pretokenized_shuffle = args.pretokenized_shuffle and not args.no_pretokenized_shuffle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx    = torch.amp.autocast(device_type="cuda", dtype=dtype)

    # TF32: ~10-15% faster matmuls on Ampere+ at negligible precision cost.
    # torch.compile exploits this heavily; harmless on older GPUs.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}  |  AMP dtype: {dtype}")

    # ── Tokenizer ────────────────────────────────────────────────────
    if args.tokenizer_json and not Path(args.tokenizer_json).exists():
        print(
            f"Custom tokenizer not found at {args.tokenizer_json}; "
            "falling back to tiktoken tokenizer"
        )
        args.tokenizer_json = None
        args.tokenizer_meta = None

    tokenizer = load_tokenizer(
        tokenizer_name=args.tokenizer,
        tokenizer_json=args.tokenizer_json,
        tokenizer_meta=args.tokenizer_meta,
    )
    tok_info = tokenizer.info()
    print(
        f"Tokenizer: {tok_info['backend']}:{tok_info['name']} "
        f"| vocab={tok_info['vocab_size']} | eot={tok_info['eot_token_id']}"
    )
    if args.vocab_size <= 0:
        args.vocab_size = tokenizer.n_vocab
    assert args.vocab_size == tokenizer.n_vocab, \
        f"vocab_size ({args.vocab_size}) != tokenizer ({tokenizer.n_vocab})"

    # ── Model config ─────────────────────────────────────────────────
    # When resuming, the checkpoint carries the saved architecture.
    # We read it here (before model creation) so the model is built
    # with exactly the right shape, then load_checkpoint() fills the
    # weights.  Training hyper-params (LR, schedule, batch) still come
    # from args / training_config.py so you can change them freely.
    # vocab_size always comes from the current tokenizer so the new
    # <|think|>/<|/think|> rows are automatically included.
    _ckpt_cfg: dict = {}
    if args.resume:
        _resume_path = Path(args.resume)
        if not _resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        print(f"Peeking at checkpoint config: {args.resume}")
        _peek = torch.load(_resume_path, map_location="cpu", weights_only=False)
        _ckpt_cfg = _peek.get("config", {})
        del _peek   # free RAM; load_checkpoint() re-reads the file later

    def _arch(key, fallback):
        """Return checkpoint value if resuming, else the fallback from args."""
        return _ckpt_cfg[key] if key in _ckpt_cfg else fallback

    config = HybridNSVSAConfig(
        # ── Architecture: from checkpoint when resuming ───────────
        d_model=_arch("d_model", args.d_model),
        num_layers=_arch("num_layers", args.num_layers),
        num_heads=_arch("num_heads", args.num_heads),
        max_seq_len=_arch("max_seq_len", args.max_seq_len),
        window_size=_arch("window_size", args.window_size),
        group_size=_arch("group_size", args.group_size),
        ffn_ratio=_arch("ffn_ratio", args.ffn_ratio),
        dropout=_arch("dropout", args.dropout),
        gate_init_bias=_arch("gate_init_bias", args.gate_init_bias),
        num_kv_heads=_arch("num_kv_heads", getattr(args, "num_kv_heads", 0)),
        qk_norm=_arch("qk_norm", getattr(args, "qk_norm", True)),
        learned_vsa_positions=_arch("learned_vsa_positions", getattr(args, "learned_vsa_positions", True)),
        # ── Vocabulary: always from current tokenizer (allows growth) ─
        vocab_size=args.vocab_size,
    )

    # ── Mode-conditioned training setup ──────────────────────────────
    # Resolve mode token IDs from tokenizer metadata → config
    if hasattr(tokenizer, "mode_token_ids"):
        tok_mode_ids = tokenizer.mode_token_ids()
        if tok_mode_ids:
            config.mode_token_ids = tok_mode_ids

    use_mode_training = (
        getattr(args, "enable_mode_training", False)
        and not getattr(args, "no_mode_training", False)
        and bool(config.mode_token_ids)
    )
    mode_mix = tuple(float(x) for x in getattr(args, "mode_mix", "0.50,0.30,0.20").split(","))
    assert len(mode_mix) == 3, f"mode_mix must have 3 values, got {len(mode_mix)}"
    mode_rng  = np.random.default_rng(42)
    blend_rng = np.random.default_rng(77)   # for CoT/pretok blending in Stage 3
    cot_path  = getattr(args, "cot_dataset", None)
    instruction_path = getattr(args, "instruction_dataset", None)
    preference_path = getattr(args, "preference_dataset", None)

    # Think token IDs for injecting think blocks into reason/deep batches
    think_token_id     = getattr(tokenizer, "think_token_id", None)
    end_think_token_id = getattr(tokenizer, "end_think_token_id", None)
    use_think_training = use_mode_training and think_token_id is not None and end_think_token_id is not None

    model = HybridNSVSA(config).to(device)
    n_params = model.num_parameters()
    print(f"Model: {n_params:,} parameters  ({n_params/1e6:.1f}M)")
    print(f"Config: d={config.d_model}, L={config.num_layers}, H={config.num_heads}, "
          f"Hkv={config.num_kv_heads}, W={config.window_size}, K={config.group_size}")
    print(f"Features: GQA={'on' if config.num_kv_heads < config.num_heads else 'off'}, "
          f"QK-Norm={'on' if config.qk_norm else 'off'}, "
          f"LearnedVSAPos={'on' if config.learned_vsa_positions else 'off'}, "
          f"RMSNorm=on")
    if use_mode_training:
        print(f"Mode training: ON  mix=(fast:{mode_mix[0]:.0%}, reason:{mode_mix[1]:.0%}, deep:{mode_mix[2]:.0%})")
        print(f"  Thinking levels: fast=none, reason=~{DEFAULT_THINK_FRACS[MODE_REASON]:.0%}, deep=~{DEFAULT_THINK_FRACS[MODE_DEEP]:.0%}")
        if use_think_training:
            print(f"  Think-block training: ON  think={think_token_id}  end_think={end_think_token_id}")
        else:
            print("  Think-block training: OFF (think tokens not in tokenizer)")
    else:
        print("Mode training: OFF")

    # ── VSA gradient scaling ─────────────────────────────────────────
    # Collect VSA params once; scale applied after clip via foreach_mul_
    # (one fused CUDA kernel) instead of a per-tensor hook per backward.
    vsa_grad_scale = getattr(args, "vsa_grad_scale", 1.0)
    vsa_params = [
        p for name, p in model.named_parameters()
        if "soft_vsa" in name and p.requires_grad
    ]
    if vsa_grad_scale != 1.0:
        print(f"VSA gradient scale: {vsa_grad_scale}x  ({len(vsa_params)} tensors, applied post-clip)")

    # ── Optimizer ────────────────────────────────────────────────────
    param_groups = make_param_groups(model, args.max_lr, args.weight_decay, args.vsa_lr_scale)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.beta1, args.beta2),
        fused=True,   # fused AdamW kernel — faster on CUDA
    )

    # bf16 doesn't overflow so GradScaler is unnecessary overhead.
    # fp16 path kept for completeness; on this GPU bf16 is always used.
    use_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # ── Raw model reference (pre-compile) ────────────────────────────
    # Must be kept before torch.compile so we can access nn.Module
    # interfaces for EMA updates and spectral norm application.
    raw_model = model

    # Spectral normalization on VSA Linear layers — prevents vanishing /
    # exploding gradient paths through the novel VSA mechanism.
    if getattr(args, "spectral_norm_vsa", False):
        from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm
        _sn_count = 0
        for _sn_name, _sn_mod in raw_model.named_modules():
            if "soft_vsa" in _sn_name and isinstance(_sn_mod, nn.Linear):
                _spectral_norm(_sn_mod, n_power_iterations=1)
                _sn_count += 1
        if _sn_count:
            print(f"Spectral norm: applied to {_sn_count} VSA Linear layers")

    # EMA model — exponential moving average of weights for smoother
    # evaluation and generation quality (used in most modern LMs).
    ema_model: "HybridNSVSA | None" = None
    ema_decay = getattr(args, "ema_decay", 0.0)
    ema_update_interval = getattr(args, "ema_update_interval", 10)
    if ema_decay > 0:
        from copy import deepcopy as _deepcopy
        ema_model = _deepcopy(raw_model).to(device).eval()
        for _p in ema_model.parameters():
            _p.requires_grad_(False)
        print(f"EMA model: decay={ema_decay}, update every {ema_update_interval} steps")

    # ── Curriculum stages ────────────────────────────────────────────
    if use_curriculum:
        stages = build_curriculum(args)
        curriculum_raw = args.curriculum
    else:
        stages = make_flat_stage(args)
        curriculum_raw = None

    print(f"\n{'Curriculum' if use_curriculum else 'Flat'} training: "
          f"{len(stages)} stage(s), {args.max_steps} total steps")
    for s in stages:
        eff_batch = s.batch_size * s.grad_accum
        tags = []
        if s.cot_mix > 0:
            tags.append(f"CoT {s.cot_mix*100:.0f}%")
        if s.instruction_mix > 0:
            tags.append(f"Instr {s.instruction_mix*100:.0f}%")
        if s.preference_mix > 0:
            tags.append(f"Pref {s.preference_mix*100:.0f}%")
        mix_tag = f"  [{' | '.join(tags)}]" if tags else ""
        print(f"  Stage {s.index}: steps [{s.start_step}, {s.end_step})  "
              f"seq_len={s.seq_len}  batch={s.batch_size}×{s.grad_accum}={eff_batch}  "
              f"lr={s.max_lr:.1e}→{s.min_lr:.1e}  warmup={s.warmup_steps}{mix_tag}")

    # ── Resume ───────────────────────────────────────────────────────
    start_step = 0
    best_val = float("inf")
    resumed_stage_idx = 0
    if args.resume:
        start_step, best_val, ckpt_tok, resumed_stage_idx = load_checkpoint(
            Path(args.resume), model, optimizer,
        )
        if ckpt_tok and not tokenizer_compatible(ckpt_tok, tok_info):
            msg = (
                "Checkpoint tokenizer metadata does not match runtime tokenizer. "
                f"checkpoint={ckpt_tok} runtime={tok_info}"
            )
            if args.strict_tokenizer_match:
                raise ValueError(msg)
            print(f"WARNING: {msg}")
        elif ckpt_tok:
            ckpt_vocab = int(ckpt_tok.get("vocab_size", 0))
            run_vocab  = int(tok_info.get("vocab_size", 0))
            if run_vocab > ckpt_vocab:
                print(f"  tokenizer vocab expanded: {ckpt_vocab} → {run_vocab} (OK)")
        print(f"Resumed from step {start_step}, best val loss {best_val:.4f}, stage {resumed_stage_idx}")

    # ── Compile ──────────────────────────────────────────────────────
    compile_requested = args.compile and not args.no_compile
    py314_or_newer = sys.version_info >= (3, 14)

    if compile_requested and py314_or_newer:
        print("Skipping torch.compile: not supported on Python 3.14+")
    elif compile_requested:
        try:
            # "default" mode: full kernel fusion + graph optimization, no CUDA graphs.
            # "reduce-overhead" (CUDA graphs) requires static tensor addresses but this
            # model has dynamic caches, conditional reasoning loops, and mask buffers
            # that alias into graph memory — causing corrupted view metadata crashes.
            print("Compiling model with torch.compile (default)...")
            model = torch.compile(model, mode="default")
        except RuntimeError as exc:
            print(f"Skipping torch.compile due to runtime error: {exc}")

    # ── Data iterator factories ──────────────────────────────────────
    data_mode = args.data_mode

    def make_train_iter(
        seq_len: int,
        batch_size: int,
        *,
        cot_mix: float = 0.0,
        instruction_mix: float = 0.0,
        preference_mix: float = 0.0,
    ):
        if data_mode == "synthetic":
            base = build_synthetic_iterator(
                vocab_size=args.vocab_size,
                seq_len=seq_len, batch_size=batch_size,
            )
        elif data_mode == "pretokenized":
            if use_curriculum:
                d = args.pretokenized_dir_pattern.format(
                    split="train", seq_len=seq_len,
                )
                if not Path(d).exists():
                    d = args.pretokenized_train_dir
                    tqdm.write(
                        f"  ⚠  Curriculum dir not found for seq_len={seq_len}, "
                        f"falling back to {d}"
                    )
            else:
                d = args.pretokenized_train_dir
            base = build_pretokenized_iterator(
                d, batch_size=batch_size,
                seed=42, shuffle=pretokenized_shuffle,
            )
        else:
            base = build_token_iterator(
                args.dataset, args.split, tokenizer,
                seq_len=seq_len, batch_size=batch_size,
                text_field=args.text_field,
            )
        iters: list[Iterator[torch.Tensor]] = []
        probs: list[float] = []

        base_weight = max(0.0, 1.0 - cot_mix - instruction_mix - preference_mix)
        if base_weight > 0.0:
            iters.append(base)
            probs.append(base_weight)

        if cot_mix > 0.0 and cot_path and Path(cot_path).exists():
            iters.append(build_cot_iterator(cot_path, tokenizer, seq_len, batch_size, seed=43))
            probs.append(cot_mix)

        if instruction_mix > 0.0 and instruction_path and Path(instruction_path).exists():
            iters.append(build_instruction_iterator(instruction_path, tokenizer, seq_len, batch_size, seed=44))
            probs.append(instruction_mix)

        if preference_mix > 0.0 and preference_path and Path(preference_path).exists():
            iters.append(build_preference_iterator(preference_path, tokenizer, seq_len, batch_size, seed=45))
            probs.append(preference_mix)

        if not iters:
            return base
        if len(iters) == 1:
            return iters[0]
        return _mix_iterators(iters, probs, blend_rng)

    def make_val_iter(seq_len: int, batch_size: int):
        if data_mode == "synthetic":
            return build_synthetic_iterator(
                vocab_size=args.vocab_size,
                seq_len=seq_len, batch_size=batch_size,
                seed=1337,
            )
        if data_mode == "pretokenized":
            if use_curriculum:
                d = args.pretokenized_dir_pattern.format(
                    split="val", seq_len=seq_len,
                )
                if not Path(d).exists():
                    d = args.pretokenized_val_dir
                    tqdm.write(
                        f"  ⚠  Curriculum dir not found for seq_len={seq_len}, "
                        f"falling back to {d}"
                    )
            else:
                d = args.pretokenized_val_dir
            return build_pretokenized_iterator(
                d, batch_size=batch_size,
                seed=1337, shuffle=pretokenized_shuffle,
            )
        val_ds = args.val_dataset or args.dataset
        return build_token_iterator(
            val_ds, args.split, tokenizer,
            seq_len=seq_len, batch_size=batch_size,
            text_field=args.text_field, seed=1337,
        )

    if data_mode == "synthetic":
        print("Using synthetic data (no network required)")
    elif data_mode == "pretokenized":
        if use_curriculum:
            print(f"Using pretokenized data (pattern: {args.pretokenized_dir_pattern})")
        else:
            print(
                "Using pretokenized data "
                f"(train={args.pretokenized_train_dir}, val={args.pretokenized_val_dir})"
            )

    # ── Initialise for first stage ───────────────────────────────────
    current_stage = get_stage_for_step(start_step, stages)
    train_iter = make_train_iter(
        current_stage.seq_len,
        current_stage.batch_size,
        cot_mix=current_stage.cot_mix,
        instruction_mix=current_stage.instruction_mix,
        preference_mix=current_stage.preference_mix,
    )

    # ── Training loop ────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    t0 = time.perf_counter()
    tokens_seen = 0
    ckpt_dir = Path(args.checkpoint_dir)

    # Loss-spike detection: skip optimizer step when step loss > 1.5 × running EMA
    _loss_ema: float | None = None
    _loss_ema_alpha: float = 0.98          # ≈50-step half-life at log_interval=10
    # LR cooldown: halve LR for 200 steps whenever val loss spikes above best × 1.15
    _lr_cooldown_steps: int = 0
    _lr_cooldown_scale: float = 1.0

    eff_batch = current_stage.batch_size * current_stage.grad_accum
    print(f"\nTraining for {args.max_steps} steps "
          f"(starting stage {current_stage.index}, "
          f"effective batch = {eff_batch} × {current_stage.seq_len} tokens)\n")

    _stage_total = current_stage.end_step - max(start_step, current_stage.start_step)
    step_iter = tqdm(
        range(start_step, args.max_steps),
        total=_stage_total,
        desc=f"stage {current_stage.index} (seq={current_stage.seq_len})",
        dynamic_ncols=True,
    )

    for step in step_iter:
        # ── Stage transition check ───────────────────────────────────
        stage = get_stage_for_step(step, stages)
        if stage.index != current_stage.index:
            old = current_stage
            current_stage = stage
            # Rebuild data iterator for new seq_len / batch_size
            train_iter = make_train_iter(
                current_stage.seq_len,
                current_stage.batch_size,
                cot_mix=current_stage.cot_mix,
                instruction_mix=current_stage.instruction_mix,
                preference_mix=current_stage.preference_mix,
            )
            # Reset throughput counters
            t0 = time.perf_counter()
            tokens_seen = 0
            accum_loss = 0.0

            eff_batch = current_stage.batch_size * current_stage.grad_accum
            data_tags = []
            if current_stage.cot_mix > 0.0:
                data_tags.append(f"CoT {current_stage.cot_mix*100:.0f}%")
            if current_stage.instruction_mix > 0.0:
                data_tags.append(f"Instruction {current_stage.instruction_mix*100:.0f}%")
            if current_stage.preference_mix > 0.0:
                data_tags.append(f"Preference {current_stage.preference_mix*100:.0f}%")
            data_info = f"  data:     {' + '.join(data_tags)}\n" if data_tags else ""
            tqdm.write(
                f"\n{'='*70}\n"
                f"  STAGE TRANSITION: {old.index} → {current_stage.index}\n"
                f"  seq_len:  {old.seq_len} → {current_stage.seq_len}\n"
                f"  batch:    {old.batch_size}×{old.grad_accum} → "
                f"{current_stage.batch_size}×{current_stage.grad_accum}  "
                f"(eff={eff_batch})\n"
                f"  lr:       {old.max_lr:.1e} → {current_stage.max_lr:.1e}\n"
                f"  warmup:   {current_stage.warmup_steps} steps\n"
                f"  steps:    [{current_stage.start_step}, {current_stage.end_step})\n"
                f"{data_info}"
                f"{'='*70}\n"
            )
            step_iter.reset(total=current_stage.end_step - current_stage.start_step)
            step_iter.set_description(
                f"stage {current_stage.index} (seq={current_stage.seq_len})"
            )

        # LR schedule (per-stage)
        lr = get_lr(step, current_stage)
        if _lr_cooldown_steps > 0:
            lr *= _lr_cooldown_scale
            _lr_cooldown_steps -= 1
        set_lr(optimizer, lr, args.vsa_lr_scale)

        # Gradient accumulation
        _micro_loss_sum = 0.0
        for micro in range(current_stage.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = make_train_iter(
                    current_stage.seq_len,
                    current_stage.batch_size,
                    cot_mix=current_stage.cot_mix,
                    instruction_mix=current_stage.instruction_mix,
                    preference_mix=current_stage.preference_mix,
                )
                batch = next(train_iter)

            batch = batch.to(device)

            # Prepend mode control token for mode-conditioned reasoning.
            # Skip whenever CoT is present in this stage (CoT rows already include control spans).
            if use_mode_training and current_stage.cot_mix == 0.0:
                batch = sample_mode_tokens(
                    batch, config.mode_token_ids,
                    mode_mix=mode_mix, rng=mode_rng,
                    think_token_id=think_token_id if use_think_training else None,
                    end_think_token_id=end_think_token_id if use_think_training else None,
                )

            input_ids = batch[:, :-1]

            with ctx:
                out = model(input_ids, labels=input_ids)
                loss = out["loss"] / current_stage.grad_accum

                # ── Entropy regularization (penalty for vocabulary collapse) ──
                # Subtracts α·H(p) from loss so the model is rewarded for
                # spreading probability mass across the vocabulary.
                if args.entropy_reg > 0.0:
                    _logits_reg = out.get("logits")
                    if _logits_reg is not None:
                        _log_p = torch.log_softmax(_logits_reg.float(), dim=-1)
                        _H_tok = -(_log_p.exp() * _log_p).sum(dim=-1).mean()
                        loss = loss - args.entropy_reg * _H_tok / current_stage.grad_accum

                # ── Embedding weight norm regularization ──────────────────────
                # Penalises RMS norm growth in the embedding matrix to prevent
                # the weight-tied projection from drifting too large.
                if args.embed_reg > 0.0:
                    _emb_rms = (raw_model.embedding.weight.float() ** 2).mean().sqrt()
                    loss = loss + args.embed_reg * _emb_rms / current_stage.grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            _lv = loss.item()
            # Min-loss floor detection — suspiciously small values suggest
            # a degenerate batch (all pad tokens, all identical, etc.)
            if _lv * current_stage.grad_accum < 0.01:
                tqdm.write(
                    f"  ⚠  Suspiciously low loss "
                    f"{_lv * current_stage.grad_accum:.6f} at step {step+1} "
                    f"(micro {micro+1}) — possible batch degeneration"
                )
            accum_loss += _lv
            _micro_loss_sum += _lv
            tokens_seen += input_ids.numel()

        # Loss spike detection — skip bad batches before they corrupt the model
        if _loss_ema is not None and _micro_loss_sum > 1.5 * _loss_ema:
            tqdm.write(
                f"  ⚡ Loss spike at step {step+1}: {_micro_loss_sum:.4f} > "
                f"1.5 × ema {_loss_ema:.4f} — skipping optimizer step"
            )
            optimizer.zero_grad(set_to_none=True)
            accum_loss -= _micro_loss_sum
            continue
        _loss_ema = _micro_loss_sum if _loss_ema is None else (
            _loss_ema_alpha * _loss_ema + (1 - _loss_ema_alpha) * _micro_loss_sum
        )

        # Gradient clipping
        if args.grad_clip > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = 0.0
        grad_norm_value = float(grad_norm)

        # VSA gradient scale — one fused foreach_mul_ after clip, before step
        if vsa_grad_scale != 1.0:
            vsa_grads = [p.grad for p in vsa_params if p.grad is not None]
            if vsa_grads:
                torch._foreach_mul_(vsa_grads, vsa_grad_scale)

        # Branch gradient norms (after unscale, before step).
        # Iterating all params + .norm(2) forces GPU sync — only do every 100 steps.
        branch_norms_str = ""
        if args.log_branch_grad_norms and (step + 1) % max(args.log_interval * 10, 100) == 0:
            bnorms = compute_branch_grad_norms(model)
            def _fmt_grad(v: float) -> str:
                if v == 0.0:
                    return "0"
                if v >= 0.01:
                    return f"{v:.2f}"
                return f"{v:.1e}"
            branch_norms_str = (
                f" | ∇attn {_fmt_grad(bnorms['attn'])}  "
                f"∇vsa {_fmt_grad(bnorms['vsa'])}  "
                f"∇ffn {_fmt_grad(bnorms['ffn'])}"
            )

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # EMA weight update — blend live weights into the EMA shadow copy.
        # Uses raw_model to bypass the torch.compile wrapper.
        if ema_model is not None and (step + 1) % ema_update_interval == 0:
            with torch.no_grad():
                for _p_ema, _p_live in zip(
                    ema_model.parameters(), raw_model.parameters()
                ):
                    _p_ema.data.mul_(ema_decay).add_(_p_live.data, alpha=1.0 - ema_decay)

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
                f"step {step+1:>6d}/{args.max_steps} "
                f"[S{current_stage.index}] | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm_value:.2f} | "
                f"{tok_per_sec:,.0f} tok/s"
                f"{branch_norms_str}"
            )
            accum_loss = 0.0

        # ── Eval ─────────────────────────────────────────────────────
        if (step + 1) % args.eval_interval == 0:
            val_iter = make_val_iter(current_stage.seq_len, current_stage.batch_size)
            val_loss = evaluate(model, val_iter, args.eval_steps, device, ctx)
            gate_str = ""
            if args.log_gate_values:
                gate_str = "\n" + log_gate_values(model)

            # EMA model validation — separate fresh iterator to avoid bias
            ema_val_str = ""
            ema_val_loss = None
            if ema_model is not None:
                val_iter_ema = make_val_iter(current_stage.seq_len, current_stage.batch_size)
                ema_val_loss = evaluate(ema_model, val_iter_ema, args.eval_steps, device, ctx)
                ema_val_str = f"  ema val: {ema_val_loss:.4f}"

            tqdm.write(
                f"  ── val loss: {val_loss:.4f}  (best: {best_val:.4f})  "
                f"[stage {current_stage.index}, seq={current_stage.seq_len}]"
                f"{ema_val_str}"
                f"{gate_str}"
            )
            if best_val < float("inf") and val_loss > best_val * 1.15:
                _lr_cooldown_steps = 200
                _lr_cooldown_scale = 0.5
                tqdm.write(
                    f"  ⚠  Val spike: {val_loss:.4f} > best {best_val:.4f} × 1.15"
                    f" — LR ×0.5 for 200 steps"
                )
            # Use the better of live-model and EMA-model loss for best checkpoint
            tracked_val = min(val_loss, ema_val_loss) if ema_val_loss is not None else val_loss
            if tracked_val < best_val:
                best_val = tracked_val
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model, optimizer, step + 1, config, best_val, tok_info,
                    stage_idx=current_stage.index,
                    curriculum_config=curriculum_raw,
                )
                update_ledger(ckpt_dir, "best.pt", step + 1, tracked_val, current_stage.index)
                # Also save EMA weights separately when EMA is better than live model
                if ema_val_loss is not None and ema_val_loss < val_loss:
                    save_checkpoint(
                        ckpt_dir / "best_ema.pt",
                        ema_model, None, step + 1, config, ema_val_loss, tok_info,
                        stage_idx=current_stage.index,
                        curriculum_config=curriculum_raw,
                    )
                    update_ledger(ckpt_dir, "best_ema.pt", step + 1, ema_val_loss, current_stage.index)

        # ── Periodic save ────────────────────────────────────────────
        if (step + 1) % args.save_interval == 0:
            ckpt_name = f"step_{step+1}.pt"
            save_checkpoint(
                ckpt_dir / ckpt_name,
                model, optimizer, step + 1, config, best_val, tok_info,
                stage_idx=current_stage.index,
                curriculum_config=curriculum_raw,
            )
            update_ledger(ckpt_dir, ckpt_name, step + 1, best_val, current_stage.index)
            manage_checkpoints(ckpt_dir, args.keep_best_k, args.keep_recent_k)

    step_iter.close()

    # ── Final save ───────────────────────────────────────────────────
    save_checkpoint(
        ckpt_dir / "final.pt",
        model, optimizer, args.max_steps, config, best_val, tok_info,
        stage_idx=current_stage.index,
        curriculum_config=curriculum_raw,
    )
    update_ledger(ckpt_dir, "final.pt", args.max_steps, best_val, current_stage.index)
    manage_checkpoints(ckpt_dir, args.keep_best_k, args.keep_recent_k)

    dt_total = time.perf_counter() - t0
    print(f"\nDone. {args.max_steps} steps in {dt_total/60:.1f} min. "
          f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
