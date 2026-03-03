#!/usr/bin/env python3
"""
Pretokenize a multi-dataset corpus into fixed-length shard files for fast training.

╔════════════════════════════════════════════════════════════════════════╗
║  Edit CORPUS_CONFIG and TOKENIZER_CONFIG below to control everything ║
╚════════════════════════════════════════════════════════════════════════╝

Output layout (curriculum):
  data/pretokenized/train_512/   shard_000000.npy … shard_NNNNNN.npy  manifest.json
  data/pretokenized/train_1024/  …
  data/pretokenized/train_2048/  …

Each shard stores shape [N, seq_len+1] int32 token IDs.

The script processes each dataset independently (into a temp staging dir),
then merges all shards into the final output directories with a unified
manifest, renumbering shards sequentially.

Usage:
  # Use the config below (default mode)
  python pretokenize_dataset.py

  # Override total sample budget from CLI
  python pretokenize_dataset.py --total_samples 2000000

  # Legacy single-dataset mode still works
  python pretokenize_dataset.py --dataset HuggingFaceFW/fineweb-edu --seq_len 512
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from nsvsa_ha.tokenizer import load_tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  TOKENIZER CONFIG — edit these to change the encoder
# ═══════════════════════════════════════════════════════════════════════

TOKENIZER_CONFIG = {
    "tokenizer_json": "tokenizers/vsa65k_mix/tokenizer.json",
    "tokenizer_meta": "tokenizers/vsa65k_mix/tokenizer_meta.json",
    "fallback_tiktoken": "cl100k_base",  # Used only if tokenizer_json missing
}


# ═══════════════════════════════════════════════════════════════════════
#  CORPUS CONFIG — add, remove, or re-weight datasets here
# ═══════════════════════════════════════════════════════════════════════
#
#  Each entry defines:
#    name         — HuggingFace dataset identifier
#    text_field   — column containing raw text (str or list[str] for multi-field)
#    config       — (optional) HF dataset config name, e.g. '20231101.en'
#    split        — HF dataset split to stream from
#    pct          — share of total_samples this source gets (must sum to 1.0)
#    streaming    — whether to stream (True) or download entire split
#
#  When text_field is a list, all fields are joined with "\n\n".
#
#  The script computes per-dataset sample counts as:
#    samples_for_dataset = round(total_samples_per_len × pct)

CORPUS_CONFIG = [
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "text_field": "text",
        "split": "train",
        "pct": 0.35,                       # High-quality educational web text
        "streaming": True,
    },
    {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "text_field": "text",
        "split": "train",
        "pct": 0.20,                       # Encyclopedia / factual grounding
        "streaming": True,
    },
    {
        "name": "allenai/c4",
        "config": "en",
        "text_field": "text",
        "split": "train",
        "pct": 0.15,                       # Broad web diversity (cleaned)
        "streaming": True,
    },
    {
        "name": "bigcode/starcoderdata",
        "text_field": "content",
        "split": "train",
        "pct": 0.15,                       # Programming/code robustness
        "streaming": True,
    },
    # ── Instruction-quality conversational data ─────────────────────
    {
        "name": "databricks/databricks-dolly-15k",
        "text_field": ["instruction", "context", "response"],
        "split": "train",
        "pct": 0.10,                       # Instruction following behavior
        "streaming": True,
    },
    # ── Compact high-quality math/reasoning supervision ────────────
    {
        "name": "openai/gsm8k",
        "config": "main",
        "text_field": ["question", "answer"],
        "split": "train",
        "pct": 0.05,                       # Reasoning depth signal
        "streaming": True,
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  SEQUENCE LENGTH CONFIG — curriculum lengths and budgets
# ═══════════════════════════════════════════════════════════════════════

SEQ_LENS = [512, 1024, 2048]
TOTAL_SAMPLES_PER_LEN = 1_500_000          # Per seq_len, split across datasets
SAMPLES_PER_SHARD = 20_000
OUT_DIR = "data/pretokenized"
SPLIT_NAME = "train"                       # Output split name
SEED = 42
SHUFFLE_BUFFER = 10_000


# ═══════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Multi-dataset pretokenizer with auto-merge")
    p.add_argument("--total_samples", type=int, default=None,
                   help=f"Override TOTAL_SAMPLES_PER_LEN (default: {TOTAL_SAMPLES_PER_LEN})")
    p.add_argument("--seq_lens", type=str, default=None,
                   help="Override SEQ_LENS, comma-separated (e.g. '512,1024,2048')")
    p.add_argument("--out_dir", default=None,
                   help=f"Override OUT_DIR (default: {OUT_DIR})")
    p.add_argument("--split", default=None,
                   help=f"Override SPLIT_NAME (default: {SPLIT_NAME})")
    p.add_argument("--seed", type=int, default=None,
                   help=f"Override SEED (default: {SEED})")

    # Legacy single-dataset mode (bypasses CORPUS_CONFIG)
    p.add_argument("--dataset", default=None,
                   help="Legacy: single HF dataset (bypasses CORPUS_CONFIG)")
    p.add_argument("--text_field", default="text",
                   help="Legacy: text column name")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Legacy: single seq_len")
    p.add_argument("--max_samples", type=int, default=500000,
                   help="Legacy: max samples for single-dataset mode")
    p.add_argument("--max_samples_per_len", type=str, default=None,
                   help="Legacy: comma-separated max samples per seq_len")
    p.add_argument("--output_split_name", default=None)
    p.add_argument("--samples_per_shard", type=int, default=None)
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no_streaming", action="store_true")
    p.add_argument("--shuffle_buffer", type=int, default=None)

    # Tokenizer overrides
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--tokenizer_json", default=None)
    p.add_argument("--tokenizer_meta", default=None)

    return p.parse_args()


def flush_shard(rows: list[list[int]], out_split_dir: Path, shard_id: int) -> tuple[str, int]:
    arr = np.asarray(rows, dtype=np.int32)
    shard_name = f"shard_{shard_id:06d}.npy"
    np.save(out_split_dir / shard_name, arr)
    return shard_name, int(arr.shape[0])


def pack_single_length(
    ds_iter,
    tokenizer,
    seq_len: int,
    max_samples: int,
    samples_per_shard: int,
    out_split_dir: Path,
    text_field: str = "text",
    desc_prefix: str = "",
) -> dict:
    """
    Pack documents from *ds_iter* into [N, seq_len+1] shards.
    Returns manifest dict.
    """
    out_split_dir.mkdir(parents=True, exist_ok=True)
    total_len = seq_len + 1
    eot = tokenizer.eot_token

    buf: list[int] = []
    rows: list[list[int]] = []
    shard_id = 0
    total_rows = 0
    total_docs = 0
    shard_meta: list[dict] = []

    label = f"{desc_prefix}seq={seq_len}" if desc_prefix else f"packing seq_len={seq_len}"
    progress = tqdm(total=max_samples, desc=label, dynamic_ncols=True)

    for ex in ds_iter:
        if isinstance(text_field, list):
            parts = [ex.get(f, "") for f in text_field]
            text = "\n\n".join(p for p in parts if p)
        else:
            text = ex.get(text_field, "")
        if not text:
            continue

        total_docs += 1
        tokens = tokenizer.encode(text)
        buf.extend(tokens)
        buf.append(eot)

        while len(buf) >= total_len and total_rows < max_samples:
            rows.append(buf[:total_len])
            del buf[:total_len]
            total_rows += 1
            progress.update(1)

            if len(rows) >= samples_per_shard:
                shard_name, n_rows = flush_shard(rows, out_split_dir, shard_id)
                shard_meta.append({"file": shard_name, "rows": n_rows})
                rows.clear()
                shard_id += 1

        if total_rows >= max_samples:
            break

    if rows:
        shard_name, n_rows = flush_shard(rows, out_split_dir, shard_id)
        shard_meta.append({"file": shard_name, "rows": n_rows})

    progress.close()

    tok_info = tokenizer.info()
    manifest = {
        "dataset": "(iterator)",
        "seq_len": seq_len,
        "sample_width": total_len,
        "max_samples": max_samples,
        "total_samples": total_rows,
        "total_docs_read": total_docs,
        "samples_per_shard": samples_per_shard,
        "num_shards": len(shard_meta),
        "shards": [s["file"] for s in shard_meta],
        "shard_rows": {s["file"]: s["rows"] for s in shard_meta},
        "tokenizer": tok_info,
    }

    manifest_path = out_split_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"  {out_split_dir}: {total_rows:,} samples, {len(shard_meta)} shards")
    return manifest


# ═══════════════════════════════════════════════════════════════════════
#  MERGE LOGIC — combines per-dataset staging dirs into final output
# ═══════════════════════════════════════════════════════════════════════

def merge_staged_shards(
    staging_dirs: list[tuple[str, Path]],
    final_dir: Path,
    seq_len: int,
    samples_per_shard: int,
    tokenizer_info: dict,
) -> dict:
    """
    Move & renumber .npy shards from multiple staging directories into
    a single final directory with a unified manifest.

    Args:
        staging_dirs: List of (dataset_name, staging_path) pairs.
        final_dir:    Target output directory.
        seq_len:      Sequence length for manifest.
        samples_per_shard: For manifest metadata.
        tokenizer_info: Tokenizer info dict for manifest.

    Returns:
        Unified manifest dict.
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    shard_counter = 0
    total_samples = 0
    total_docs = 0
    all_shards: list[str] = []
    shard_rows: dict[str, int] = {}
    source_breakdown: dict[str, int] = {}

    for ds_name, staging_path in staging_dirs:
        if not staging_path.exists():
            continue

        # Read the staging manifest
        staging_manifest_path = staging_path / "manifest.json"
        if staging_manifest_path.exists():
            sm = json.loads(staging_manifest_path.read_text())
            ds_samples = sm.get("total_samples", 0)
            ds_docs = sm.get("total_docs_read", 0)
        else:
            ds_samples = 0
            ds_docs = 0

        source_breakdown[ds_name] = ds_samples
        total_samples += ds_samples
        total_docs += ds_docs

        # Move + renumber shard files
        staged_shards = sorted(staging_path.glob("shard_*.npy"))
        for shard_file in staged_shards:
            new_name = f"shard_{shard_counter:06d}.npy"

            # Load to count rows
            arr = np.load(shard_file)
            n_rows = int(arr.shape[0])

            dest = final_dir / new_name
            shutil.move(str(shard_file), str(dest))

            all_shards.append(new_name)
            shard_rows[new_name] = n_rows
            shard_counter += 1

        # Clean up staging dir
        shutil.rmtree(staging_path, ignore_errors=True)

    manifest = {
        "datasets": source_breakdown,
        "seq_len": seq_len,
        "sample_width": seq_len + 1,
        "total_samples": total_samples,
        "total_docs_read": total_docs,
        "samples_per_shard": samples_per_shard,
        "num_shards": len(all_shards),
        "shards": all_shards,
        "shard_rows": shard_rows,
        "tokenizer": tokenizer_info,
    }

    manifest_path = final_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Resolve tokenizer ────────────────────────────────────────────
    tok_json = args.tokenizer_json or TOKENIZER_CONFIG["tokenizer_json"]
    tok_meta = args.tokenizer_meta or TOKENIZER_CONFIG["tokenizer_meta"]
    tok_fallback = args.tokenizer or TOKENIZER_CONFIG["fallback_tiktoken"]

    if tok_json and not Path(tok_json).exists():
        print(f"Custom tokenizer not found at {tok_json}; falling back to tiktoken")
        tok_json = None
        tok_meta = None

    tokenizer = load_tokenizer(
        tokenizer_name=tok_fallback,
        tokenizer_json=tok_json,
        tokenizer_meta=tok_meta,
    )
    tok_info = tokenizer.info()
    print(
        f"Tokenizer: {tok_info['backend']}:{tok_info['name']} "
        f"| vocab={tok_info['vocab_size']} | eot={tok_info['eot_token_id']}"
    )

    # ── Detect legacy single-dataset mode ────────────────────────────
    if args.dataset is not None:
        _run_legacy_mode(args, tokenizer)
        return

    # ── Multi-dataset mode (uses CORPUS_CONFIG) ──────────────────────
    seq_lens = [int(s) for s in args.seq_lens.split(",")] if args.seq_lens else SEQ_LENS
    total_per_len = args.total_samples or TOTAL_SAMPLES_PER_LEN
    out_dir = Path(args.out_dir or OUT_DIR)
    split_name = args.split or SPLIT_NAME
    seed = args.seed if args.seed is not None else SEED
    shard_size = args.samples_per_shard or SAMPLES_PER_SHARD
    shuffle_buf = args.shuffle_buffer or SHUFFLE_BUFFER

    # Validate corpus weights
    total_pct = sum(ds["pct"] for ds in CORPUS_CONFIG)
    if abs(total_pct - 1.0) > 0.01:
        print(f"⚠  CORPUS_CONFIG pct values sum to {total_pct:.3f}, expected 1.0")

    print(f"\n{'='*72}")
    print(f"  Multi-dataset pretokenization")
    print(f"  Seq lengths: {seq_lens}")
    print(f"  Total samples per length: {total_per_len:,}")
    print(f"  Output: {out_dir}/{split_name}_<seq_len>/")
    print(f"{'='*72}")

    print(f"\n  Corpus composition:")
    for ds in CORPUS_CONFIG:
        n_samples = round(total_per_len * ds["pct"])
        print(f"    {ds['pct']*100:5.1f}%  {ds['name']:<50s}  → {n_samples:>10,} samples")
    print()

    # ── Process each seq_len ─────────────────────────────────────────
    for sl in seq_lens:
        print(f"\n{'─'*72}")
        print(f"  Processing seq_len={sl}")
        print(f"{'─'*72}")

        staging_dirs: list[tuple[str, Path]] = []
        staging_root = out_dir / f"_staging_{split_name}_{sl}"

        for ds_cfg in CORPUS_CONFIG:
            ds_name = ds_cfg["name"]
            ds_text = ds_cfg["text_field"]
            ds_split = ds_cfg["split"]
            ds_stream = ds_cfg["streaming"]
            n_samples = round(total_per_len * ds_cfg["pct"])

            if n_samples == 0:
                continue

            # Create a staging subdirectory for this dataset
            safe_name = ds_name.replace("/", "__")
            staging_path = staging_root / safe_name
            staging_dirs.append((ds_name, staging_path))

            short = ds_name.split("/")[-1][:20]
            print(f"\n  [{short}] Loading {ds_name} [{ds_split}] → {n_samples:,} samples")

            ds_config = ds_cfg.get("config", None)
            ds = load_dataset(ds_name, ds_config, split=ds_split, streaming=ds_stream)
            if ds_stream:
                ds = ds.shuffle(seed=seed, buffer_size=shuffle_buf)

            pack_single_length(
                ds_iter=ds,
                tokenizer=tokenizer,
                seq_len=sl,
                max_samples=n_samples,
                samples_per_shard=shard_size,
                out_split_dir=staging_path,
                text_field=ds_text,
                desc_prefix=f"[{short}] ",
            )

        # ── Merge all staged shards into final dir ───────────────────
        final_dir = out_dir / f"{split_name}_{sl}"

        # Remove old data in the final dir (will be replaced)
        if final_dir.is_symlink():
            print(f"\n  Removing symlink {final_dir}")
            final_dir.unlink()
        elif final_dir.exists():
            print(f"\n  Removing old data in {final_dir}")
            shutil.rmtree(final_dir)

        print(f"\n  Merging → {final_dir}")
        manifest = merge_staged_shards(
            staging_dirs=staging_dirs,
            final_dir=final_dir,
            seq_len=sl,
            samples_per_shard=shard_size,
            tokenizer_info=tok_info,
        )

        total = manifest["total_samples"]
        n_shards = manifest["num_shards"]
        print(f"  ✓ {final_dir}: {total:,} samples, {n_shards} shards")
        for ds_name, count in manifest.get("datasets", {}).items():
            pct = count / total * 100 if total > 0 else 0
            print(f"      {pct:5.1f}%  {ds_name} ({count:,})")

        # Clean up staging root
        shutil.rmtree(staging_root, ignore_errors=True)

    print(f"\n{'='*72}")
    print(f"  Done. All lengths merged into {out_dir}/{split_name}_*/")
    print(f"{'='*72}\n")


# ═══════════════════════════════════════════════════════════════════════
#  LEGACY SINGLE-DATASET MODE (--dataset flag)
# ═══════════════════════════════════════════════════════════════════════

def _run_legacy_mode(args, tokenizer):
    """Original single-dataset behaviour for backward compatibility."""
    streaming = args.streaming and not args.no_streaming
    seed = args.seed if args.seed is not None else SEED
    shard_size = args.samples_per_shard or SAMPLES_PER_SHARD
    shuffle_buf = args.shuffle_buffer or SHUFFLE_BUFFER

    if args.seq_lens:
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
        if args.max_samples_per_len:
            max_per_len = [int(s) for s in args.max_samples_per_len.split(",")]
            if len(max_per_len) != len(seq_lens):
                raise ValueError(
                    f"--max_samples_per_len has {len(max_per_len)} entries "
                    f"but --seq_lens has {len(seq_lens)}"
                )
        else:
            max_per_len = [args.max_samples] * len(seq_lens)
    elif args.seq_len:
        seq_lens = [args.seq_len]
        max_per_len = [args.max_samples]
    else:
        seq_lens = [512]
        max_per_len = [args.max_samples]

    out_dir_root = Path(args.out_dir or OUT_DIR)
    split_name = args.split or SPLIT_NAME
    out_name = args.output_split_name or split_name
    multi_length = len(seq_lens) > 1

    print(f"\nLoading dataset: {args.dataset} [{split_name}] (streaming={streaming})")
    ds = load_dataset(args.dataset, split=split_name, streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buf)

    for sl, ms in zip(seq_lens, max_per_len):
        if multi_length:
            out_dir = out_dir_root / f"{out_name}_{sl}"
        else:
            out_dir = out_dir_root / out_name

        print(f"\nPacking seq_len={sl}, max_samples={ms:,} -> {out_dir}")

        if multi_length:
            ds_for_len = load_dataset(args.dataset, split=split_name, streaming=streaming)
            if streaming:
                ds_for_len = ds_for_len.shuffle(seed=seed, buffer_size=shuffle_buf)
        else:
            ds_for_len = ds

        m = pack_single_length(
            ds_iter=ds_for_len,
            tokenizer=tokenizer,
            seq_len=sl,
            max_samples=ms,
            samples_per_shard=shard_size,
            out_split_dir=out_dir,
            text_field=args.text_field,
        )
        m["dataset"] = args.dataset
        m["split"] = split_name
        m["output_split_name"] = out_name
        (out_dir / "manifest.json").write_text(json.dumps(m, indent=2))

    print("\nDone pretokenizing all lengths.")


if __name__ == "__main__":
    main()
