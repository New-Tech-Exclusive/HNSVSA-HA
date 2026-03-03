#!/usr/bin/env python3
"""
Train a custom BPE tokenizer for HybridNSVSA-HA.

Design defaults (for this architecture):
- Vocab size: 48k (memory-quality balance for 8GB GPUs)
- Domain: English + code
- Byte-level BPE (robust for arbitrary text/code)
- Stable special tokens with explicit IDs in metadata

Outputs:
  <out_dir>/tokenizer.json
  <out_dir>/tokenizer_meta.json

Usage:
  python train_tokenizer.py \
    --out_dir tokenizers/vsa48k_en_code \
    --datasets HuggingFaceFW/fineweb-edu codeparrot/github-code-clean \
    --text_fields text code \
    --sample_docs 2000000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

# ── Corpus mix (mirrors CORPUS_CONFIG in pretokenize_dataset.py) ──────────────
CORPUS_CONFIG = [
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "text_field": "text",
        "split": "train",
        "pct": 0.45,
    },
    {
        "name": "bigcode/starcoderdata",
        "text_field": "content",
        "split": "train",
        "pct": 0.20,
    },
    {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "text_field": "text",
        "split": "train",
        "pct": 0.15,
    },
    {
        "name": "emozilla/pg19-test",
        "text_field": "text",
        "split": "test",
        "pct": 0.10,
    },
    {
        "name": "openai/gsm8k",
        "config": "main",
        "text_field": ["question", "answer"],
        "split": "train",
        "pct": 0.10,
    },
]
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFKC, Strip
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


@dataclass
class TokenizerMeta:
    name: str
    backend: str
    vocab_size: int
    eot_token: str
    eot_token_id: int
    bos_token: str
    bos_token_id: int
    pad_token: str
    pad_token_id: int
    unk_token: str
    unk_token_id: int
    fast_token: str
    fast_token_id: int
    reason_token: str
    reason_token_id: int
    deep_token: str
    deep_token_id: int
    think_token: str
    think_token_id: int
    end_think_token: str
    end_think_token_id: int
    datasets: list[str]
    text_fields: list[str]
    sample_docs: int
    min_frequency: int
    seed: int
    loaded_datasets: list[str]


def parse_args():
    p = argparse.ArgumentParser(description="Train custom tokenizer for NSVSA-HA")

    p.add_argument("--out_dir", default="tokenizers/vsa65k_mix")
    p.add_argument("--name", default="vsa65k_mix")
    p.add_argument("--vocab_size", type=int, default=65_000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--sample_docs", type=int, default=2_000_000,
                   help="Max total documents sampled across datasets")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no_streaming", action="store_true")
    p.add_argument("--shuffle_buffer", type=int, default=10000)

    return p.parse_args()


def build_dataset_iters(args) -> tuple[list[dict], list[str]]:
    streaming = args.streaming and not args.no_streaming
    
    iters_info = []
    loaded_names: list[str] = []
    
    for cfg in CORPUS_CONFIG:
        ds_name = cfg["name"]
        split = cfg.get("split", "train")
        config_name = cfg.get("config", None)
        pct = cfg.get("pct", 1.0)
        
        print(f"Loading dataset: {ds_name} [{config_name or "default"}, {split}] (pct={pct}, streaming={streaming})")
        try:
            if config_name:
                ds = load_dataset(ds_name, config_name, split=split, streaming=streaming)
            else:
                ds = load_dataset(ds_name, split=split, streaming=streaming)
                
            if streaming:
                ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            
            iters_info.append({
                "iter": iter(ds),
                "cfg": cfg,
                "weight": pct
            })
            loaded_names.append(f"{ds_name} ({config_name})")
        except Exception as exc:
            print(f"  ⚠ Skipping dataset '{ds_name}': {exc}")

    if not iters_info:
        raise RuntimeError("No datasets could be loaded.")

    return iters_info, loaded_names


def extract_text(example: dict, text_field: str | list[str]) -> str:
    if isinstance(text_field, list):
        parts = [example.get(f, "") for f in text_field]
        return "\n\n".join(p for p in parts if p and isinstance(p, str))
    val = example.get(text_field, "")
    return val if isinstance(val, str) else ""


def text_iterator(args, iters_info: list[dict]) -> Iterator[str]:
    random.seed(args.seed)
    
    weights = [info["weight"] for info in iters_info]
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    indices = list(range(len(iters_info)))
    
    produced = 0
    while produced < args.sample_docs and indices:
        idx = random.choices(indices, weights=probs, k=1)[0]
        actual_idx = indices.index(idx)
        info = iters_info[actual_idx]
        
        try:
            ex = next(info["iter"])
        except StopIteration:
            iters_info.pop(actual_idx)
            indices.pop(actual_idx)
            weights.pop(actual_idx)
            if not indices:
                break
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            continue
        except Exception:
            continue

        text = extract_text(ex, info["cfg"]["text_field"]).strip()
        if text:
            yield text
            produced += 1
            if produced % 100_000 == 0:
                print(f"  collected {produced:,} docs...")


def main():
    args = parse_args()

    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [
        "<|unk|>",
        "<|pad|>",
        "<|bos|>",
        "<|endoftext|>",
        "<|fast|>",
        "<|reason|>",
        "<|deep|>",
        "<|think|>",
        "<|/think|>",
    ]

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = Sequence([NFKC(), Strip()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    print(
        f"Training tokenizer: vocab={args.vocab_size:,}, min_freq={args.min_frequency}, "
        f"sample_docs={args.sample_docs:,}"
    )

    iters, loaded_names = build_dataset_iters(args)
    print(f"Using datasets: {loaded_names}")

    tokenizer.train_from_iterator(text_iterator(args, iters), trainer=trainer, length=args.sample_docs)

    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    unk_id = tokenizer.token_to_id("<|unk|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    fast_id = tokenizer.token_to_id("<|fast|>")
    reason_id = tokenizer.token_to_id("<|reason|>")
    deep_id = tokenizer.token_to_id("<|deep|>")
    think_id = tokenizer.token_to_id("<|think|>")
    end_think_id = tokenizer.token_to_id("<|/think|>")

    if any(x is None for x in [unk_id, pad_id, bos_id, eot_id, fast_id, reason_id, deep_id, think_id, end_think_id]):
        raise RuntimeError("Special token IDs missing after training")

    meta = TokenizerMeta(
        name=args.name,
        backend="hf_tokenizers",
        vocab_size=tokenizer.get_vocab_size(),
        eot_token="<|endoftext|>",
        eot_token_id=int(eot_id),
        bos_token="<|bos|>",
        bos_token_id=int(bos_id),
        pad_token="<|pad|>",
        pad_token_id=int(pad_id),
        unk_token="<|unk|>",
        unk_token_id=int(unk_id),
        fast_token="<|fast|>",
        fast_token_id=int(fast_id),
        reason_token="<|reason|>",
        reason_token_id=int(reason_id),
        deep_token="<|deep|>",
        deep_token_id=int(deep_id),
        think_token="<|think|>",
        think_token_id=int(think_id),
        end_think_token="<|/think|>",
        end_think_token_id=int(end_think_id),
        datasets=[cfg["name"] for cfg in CORPUS_CONFIG],
        text_fields=[str(cfg.get("text_field", "")) for cfg in CORPUS_CONFIG],
        sample_docs=args.sample_docs,
        min_frequency=args.min_frequency,
        seed=args.seed,
        loaded_datasets=loaded_names,
    )

    meta_path = out_dir / "tokenizer_meta.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2))

    dt = time.perf_counter() - t0
    print("\nDone.")
    print(f"  tokenizer: {tokenizer_json}")
    print(f"  metadata : {meta_path}")
    print(f"  final vocab size: {meta.vocab_size:,}")
    print(f"  eot token id: {meta.eot_token_id}")
    print(f"  elapsed: {dt/60:.1f} min")


if __name__ == "__main__":
    main()
