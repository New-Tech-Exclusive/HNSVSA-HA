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
    datasets: list[str]
    text_fields: list[str]
    sample_docs: int
    min_frequency: int
    seed: int
    loaded_datasets: list[str]


def parse_args():
    p = argparse.ArgumentParser(description="Train custom tokenizer for NSVSA-HA")

    p.add_argument("--out_dir", default="tokenizers/vsa48k_en_code")
    p.add_argument("--name", default="vsa48k_en_code")
    p.add_argument("--vocab_size", type=int, default=48_000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--sample_docs", type=int, default=2_000_000,
                   help="Max total documents sampled across datasets")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--datasets",
        nargs="+",
        default=["HuggingFaceFW/fineweb-edu"],
        help="HF dataset names, sampled round-robin",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="One split for all datasets, or one per dataset",
    )
    p.add_argument(
        "--text_fields",
        nargs="+",
        default=["text", "code"],
        help="Candidate text fields; first present field per example is used",
    )
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no_streaming", action="store_true")
    p.add_argument("--shuffle_buffer", type=int, default=10000)

    return p.parse_args()


def build_dataset_iters(args) -> tuple[list[Iterator[dict]], list[str]]:
    streaming = args.streaming and not args.no_streaming

    if len(args.splits) == 1:
        splits = args.splits * len(args.datasets)
    else:
        if len(args.splits) != len(args.datasets):
            raise ValueError("--splits must have length 1 or match --datasets length")
        splits = args.splits

    iters = []
    loaded_names: list[str] = []
    for ds_name, split in zip(args.datasets, splits):
        print(f"Loading dataset: {ds_name} [{split}] (streaming={streaming})")
        try:
            ds = load_dataset(ds_name, split=split, streaming=streaming)
            if streaming:
                ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            iters.append(iter(ds))
            loaded_names.append(ds_name)
        except Exception as exc:
            print(f"  ⚠ Skipping dataset '{ds_name}': {exc}")

    if not iters:
        raise RuntimeError(
            "No datasets could be loaded. Please pass supported HF parquet-style datasets "
            "via --datasets."
        )

    return iters, loaded_names


def pick_text(example: dict, text_fields: list[str]) -> str:
    for field in text_fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def text_iterator(args, iters: list[Iterator[dict]]) -> Iterator[str]:
    random.seed(args.seed)
    num_datasets = len(iters)

    produced = 0
    ds_index = 0
    while produced < args.sample_docs:
        it = iters[ds_index]
        try:
            ex = next(it)
        except StopIteration:
            ds_index = (ds_index + 1) % num_datasets
            continue
        except Exception:
            ds_index = (ds_index + 1) % num_datasets
            continue

        text = pick_text(ex, args.text_fields)
        if text:
            yield text
            produced += 1
            if produced % 100_000 == 0:
                print(f"  collected {produced:,} docs...")

        ds_index = (ds_index + 1) % num_datasets


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

    if any(x is None for x in [unk_id, pad_id, bos_id, eot_id]):
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
        datasets=args.datasets,
        text_fields=args.text_fields,
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
