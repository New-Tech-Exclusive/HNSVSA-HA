#!/usr/bin/env python3
"""
scripts/data_config.py — ARNL Training Data Configuration
==========================================================

Centralised data configuration for train.py and populate_map.py.
Supports HuggingFace datasets as the primary source, with optional
local-file fallbacks and a JSON config file for reproducibility.

────────────────────────────────────────────────────────────────
Config file format  (save with --init-config / DataConfig.save())
────────────────────────────────────────────────────────────────
{
  "tokenizer_name": "gpt2",        // shared HF tokenizer (per-phase overrides allowed)
  "max_seq_len": 512,

  "phase1": {
    "hf_dataset":      "wikitext",                 // HF dataset repo id
    "hf_config_name":  "wikitext-2-raw-v1",        // dataset config/subset
    "hf_split":        "train",
    "hf_text_field":   "text",                     // column that contains raw text
    "hf_max_samples":  50000,                      // null = all
    "hf_streaming":    false,
    "local_file":      null,                       // fallback plain-text file
    "tokenizer_name":  null                        // override global tokenizer
  },

  "phase2": {
    "hf_dataset":      "my_org/arnl_facts",
    "hf_split":        "train",
    "col_input_ids":   "input_ids",                // column → input token IDs
    "col_label_ids":   "label_ids",                // column → label token IDs
    "col_target_id":   "target_id",                // column → single target token
    "col_anchor_ids":  "anchor_ids",               // column → anchor token list
    "col_text":        null,                       // tokenize this text column instead
    "hf_max_samples":  null,
    "local_file":      null
  },

  // Phase 3 removed in V1.1 — only 2 training phases

  "facts": {
    "hf_dataset":          "my_org/knowledge_base",
    "hf_split":            "train",
    "col_anchor_ids":      "anchor_ids",           // list[int] column
    "col_target_token_id": "target_token_id",      // int column
    "col_tier":            "tier",                 // "axiom"|"domain"|"user"
    "col_hard_lock":       "hard_lock",            // bool column (optional)
    "col_s_base":          "s_base",               // int column (optional)
    "col_text_target":     null,                   // text → tokenize → target
    "col_text_anchors":    null,                   // text → tokenize → anchors
    "hf_max_samples":      null,
    "local_file":          null
  }
}

────────────────────────────────────────────────────────────────
Quick-start — generate an example config then train:

    python scripts/data_config.py --init-config data/my_data.json
    # Edit data/my_data.json to point at your HF datasets
    python scripts/train.py --data-config data/my_data.json --checkpoint-dir ckpt/

Or for map population:
    python scripts/populate_map.py --data-config data/my_data.json \\
        --shade-file ckpt/shade.json --checkpoint-dir ckpt/
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import List, Optional

import torch
from torch.utils.data import Dataset

# Allow running as a module from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arnl.config import ARNLConfig
from arnl.training import EMLMMasker


# ════════════════════════════════════════════════════════════════
# Sub-config dataclasses
# ════════════════════════════════════════════════════════════════

@dataclass
class Phase1DataCfg:
    """Data source for Phase 1 (plain-text EMLM pretraining)."""

    # HuggingFace source
    hf_dataset: Optional[str] = None
    """Dataset repo id, e.g. 'wikitext' or 'EleutherAI/pile'."""
    hf_config_name: Optional[str] = None
    """Dataset config/subset name, e.g. 'wikitext-103-raw-v1'."""
    hf_split: str = "train"
    hf_text_field: str = "text"
    """Column that contains raw document text."""
    hf_streaming: bool = False
    """Use streaming mode (avoids downloading the full dataset)."""
    hf_max_samples: Optional[int] = None
    """Truncate to this many samples. null = use all."""

    # Local fallback
    local_file: Optional[str] = None
    """Plain-text file, one document per line."""

    # Per-phase tokenizer override
    tokenizer_name: Optional[str] = None


@dataclass
class Phase23DataCfg:
    """Data source for Phase 2 training (Phase 3 removed in V1.1)."""

    # HuggingFace source
    hf_dataset: Optional[str] = None
    hf_config_name: Optional[str] = None
    hf_split: str = "train"
    hf_streaming: bool = False
    hf_max_samples: Optional[int] = None

    # Column mapping (dataset → expected fields)
    col_input_ids: str = "input_ids"
    col_label_ids: str = "label_ids"
    col_target_id: str = "target_id"
    col_anchor_ids: str = "anchor_ids"
    col_text: Optional[str] = None
    """If set, tokenize this text column and use as input_ids (requires tokenizer)."""

    # Local fallback (JSONL)
    local_file: Optional[str] = None

    # Per-phase tokenizer override
    tokenizer_name: Optional[str] = None


@dataclass
class FactsDataCfg:
    """Data source for populate_map.py (System 2 knowledge injection)."""

    # HuggingFace source
    hf_dataset: Optional[str] = None
    hf_config_name: Optional[str] = None
    hf_split: str = "train"
    hf_streaming: bool = False
    hf_max_samples: Optional[int] = None

    # Column mapping
    col_anchor_ids: str = "anchor_ids"
    col_target_token_id: str = "target_token_id"
    col_tier: str = "tier"
    col_hard_lock: Optional[str] = "hard_lock"
    col_s_base: Optional[str] = "s_base"

    # Text-to-token columns (tokenizer required)
    col_text_target: Optional[str] = None
    """Text column → first token becomes the target token ID."""
    col_text_anchors: Optional[str] = None
    """Text column → all tokens become the anchor IDs."""

    # Local fallback (JSON array or JSONL)
    local_file: Optional[str] = None

    # Per-phase tokenizer override
    tokenizer_name: Optional[str] = None


# ════════════════════════════════════════════════════════════════
# Top-level DataConfig
# ════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """Top-level training data configuration.

    Keeps all data sources in one place so train.py, populate_map.py,
    and any future scripts share a single config file.
    """

    tokenizer_name: Optional[str] = None
    """Shared HuggingFace tokenizer name/path (e.g. 'gpt2', 'bert-base-uncased').
    Overridden by per-phase *tokenizer_name* settings."""

    max_seq_len: int = 2048
    """Maximum sequence length fed to the model."""

    phase1: Phase1DataCfg = field(default_factory=Phase1DataCfg)
    phase2: Phase23DataCfg = field(default_factory=Phase23DataCfg)
    # phase3 removed in V1.1 — kept for config compat but unused
    phase3: Phase23DataCfg = field(default_factory=Phase23DataCfg)
    facts: FactsDataCfg = field(default_factory=FactsDataCfg)

    # ── Serialisation ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write this config to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"  DataConfig saved → {os.path.abspath(path)}")

    @classmethod
    def load(cls, path: str) -> "DataConfig":
        """Load from a JSON file written by :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        def _sub(klass, key):
            raw = d.get(key) or {}
            valid = {fld.name for fld in dc_fields(klass)}
            return klass(**{k: v for k, v in raw.items() if k in valid})

        return cls(
            tokenizer_name=d.get("tokenizer_name"),
            max_seq_len=d.get("max_seq_len", 512),
            phase1=_sub(Phase1DataCfg, "phase1"),
            phase2=_sub(Phase23DataCfg, "phase2"),
            phase3=_sub(Phase23DataCfg, "phase3"),
            facts=_sub(FactsDataCfg, "facts"),
        )

    @classmethod
    def example(cls) -> "DataConfig":
        """Return an annotated example config (good starting point for editing)."""
        return cls(
            tokenizer_name="salt",
            max_seq_len=512,
            phase1=Phase1DataCfg(
                hf_dataset="wikitext",
                hf_config_name="wikitext-2-raw-v1",
                hf_split="train",
                hf_text_field="text",
                hf_max_samples=50_000,
                # local_file="data/corpus.txt",    # uncomment to use a local file instead
            ),
            phase2=Phase23DataCfg(
                # hf_dataset="my_org/my_facts",    # uncomment for HuggingFace source
                local_file="data/example_facts.jsonl",
            ),
            phase3=Phase23DataCfg(
                # hf_dataset="my_org/my_facts",
                local_file="data/example_facts.jsonl",
            ),
            facts=FactsDataCfg(
                # hf_dataset="my_org/knowledge_base",  # uncomment for HuggingFace source
                local_file="data/example_facts.json",
            ),
        )


# ════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════

def _require_datasets():
    """Import the `datasets` package or raise a clear error."""
    try:
        import datasets as _ds
        return _ds
    except ImportError:
        print(
            "\n[data_config] ERROR: The 'datasets' package is not installed.\n"
            "  Install it with:  pip install datasets\n"
        )
        sys.exit(1)


def _require_transformers():
    """Import AutoTokenizer from `transformers` or raise a clear error."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError:
        print(
            "\n[data_config] ERROR: The 'transformers' package is not installed.\n"
            "  Install it with:  pip install transformers\n"
        )
        sys.exit(1)


def _load_tokenizer(name: Optional[str], *, silent: bool = False):
    """Load a tokenizer by name or path.

    Supports:
      - ``"salt:<dir>"``  → SALTTokenizer from that directory
      - ``"salt"``        → SALTTokenizer from ``tokenizer/``
      - Any other string  → HuggingFace AutoTokenizer
    """
    if name is None:
        return None

    # SALT tokenizer
    if name.startswith("salt"):
        parts = name.split(":", 1)
        salt_dir = parts[1] if len(parts) > 1 else "tokenizer/"
        tok_path = os.path.join(salt_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            from arnl.salt_tokenizer import SALTTokenizer
            tok = SALTTokenizer(salt_dir)
            if not silent:
                print(f"  [data_config] Loaded SALT tokenizer from {salt_dir}")
            return tok
        else:
            if not silent:
                print(f"  [data_config] WARNING: SALT tokenizer not found at {salt_dir}")
            return None

    # HuggingFace tokenizer
    AutoTokenizer = _require_transformers()
    try:
        tok = AutoTokenizer.from_pretrained(name)
        if not silent:
            print(f"  [data_config] Loaded tokenizer: {name}")
        return tok
    except Exception as exc:
        print(f"  [data_config] WARNING: could not load tokenizer '{name}': {exc}")
        return None


def _effective_tokenizer(data_cfg: DataConfig, phase_tok: Optional[str]):
    """Return the per-phase tokenizer if set, else fall back to global."""
    name = phase_tok or data_cfg.tokenizer_name
    return _load_tokenizer(name)


def _clip_ids(ids: List[int], vocab_size: int) -> List[int]:
    return [min(i, vocab_size - 1) for i in ids]


def _pad(ids: List[int], max_len: int) -> List[int]:
    return (ids + [0] * max_len)[:max_len]


# ════════════════════════════════════════════════════════════════
# Phase 1 — plain-text EMLM dataset
# ════════════════════════════════════════════════════════════════

def _texts_from_hf(cfg: Phase1DataCfg) -> List[str]:
    """Download/stream text documents from a HuggingFace dataset."""
    ds_lib = _require_datasets()
    print(f"  [data_config] Loading HF dataset '{cfg.hf_dataset}'"
          + (f" / {cfg.hf_config_name}" if cfg.hf_config_name else "")
          + f" ({cfg.hf_split}) …")

    ds = ds_lib.load_dataset(
        cfg.hf_dataset,
        cfg.hf_config_name,
        split=cfg.hf_split,
        streaming=cfg.hf_streaming,
    )

    texts: List[str] = []
    for i, row in enumerate(ds):
        if cfg.hf_max_samples and i >= cfg.hf_max_samples:
            break
        t = row.get(cfg.hf_text_field, "")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())

    print(f"  [data_config] Loaded {len(texts):,} text documents from HF.")
    return texts


def _texts_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]


class HFPhase1Dataset(Dataset):
    """Phase 1 dataset that accepts a pre-built list of raw text strings.

    Applies EMLM masking and tokenizes.  Used for both HF-sourced and
    local-file-sourced text.
    """

    def __init__(
        self,
        texts: List[str],
        arnl_config: ARNLConfig,
        masker: EMLMMasker,
        max_len: int = 512,
        tokenizer=None,
    ):
        self.samples: List[List[int]] = []
        masked = masker.mask_batch(texts)

        if tokenizer is None:
            print("  [data_config] No tokenizer — using character-level fallback.")
            all_text = " ".join(masked)
            chars = sorted(set(all_text))
            char2id = {c: i + 3 for i, c in enumerate(chars)}
            char2id.update({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2})
            encode = lambda t: [1] + [char2id.get(c, 3) for c in t] + [2]
        else:
            encode = tokenizer.encode

        vs = arnl_config.vocab_size
        for txt in masked:
            ids = _clip_ids(encode(txt), vs)
            if len(ids) < 2:
                continue
            for start in range(0, max(1, len(ids) - max_len + 1), max_len // 2):
                chunk = ids[start: start + max_len]
                if len(chunk) >= 2:
                    self.samples.append(_pad(chunk, max_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": t, "labels": t.clone()}


def build_phase1_dataset(
    data_cfg: DataConfig,
    arnl_config: ARNLConfig,
    masker: EMLMMasker,
) -> Dataset:
    """Build a Phase 1 EMLM training dataset from *data_cfg*.

    Priority:
      1. HuggingFace dataset (if ``phase1.hf_dataset`` is set)
      2. Local file (if ``phase1.local_file`` is set)
      Returns ``None`` if neither source is configured.
    """
    cfg = data_cfg.phase1
    tok = _effective_tokenizer(data_cfg, cfg.tokenizer_name)

    if cfg.hf_dataset:
        texts = _texts_from_hf(cfg)
        return HFPhase1Dataset(texts, arnl_config, masker, data_cfg.max_seq_len, tok)

    if cfg.local_file:
        print(f"  [data_config] Phase 1 — loading local file: {cfg.local_file}")
        texts = _texts_from_file(cfg.local_file)
        return HFPhase1Dataset(texts, arnl_config, masker, data_cfg.max_seq_len, tok)

    return None


# ════════════════════════════════════════════════════════════════
# Phase 2 / 3 — structured fact-injection dataset
# ════════════════════════════════════════════════════════════════

def _process_phase23_row(
    row: dict,
    cfg: Phase23DataCfg,
    arnl_config: ARNLConfig,
    max_len: int,
    tokenizer=None,
) -> Optional[dict]:
    """Convert a raw row (dict from HF or JSONL) to a Phase 2/3 sample dict."""
    vs = arnl_config.vocab_size

    # input_ids
    if cfg.col_text and tokenizer and cfg.col_text in row:
        inp = _clip_ids(tokenizer.encode(row[cfg.col_text]), vs)
    elif cfg.col_input_ids in row:
        inp = _clip_ids(row[cfg.col_input_ids], vs)
    else:
        return None  # skip rows missing required fields

    inp = inp[:max_len]
    padded_inp = _pad(inp, max_len)

    # label_ids (fall back to input_ids)
    if cfg.col_label_ids in row:
        lbl = _clip_ids(row[cfg.col_label_ids], vs)[:max_len]
    else:
        lbl = inp
    padded_lbl = _pad(lbl, max_len)

    return {
        "input_ids": torch.tensor(padded_inp, dtype=torch.long),
        "labels":    torch.tensor(padded_lbl, dtype=torch.long),
        "target_id": int(row.get(cfg.col_target_id, 0)),
        "anchor_ids": list(row.get(cfg.col_anchor_ids, [])),
    }


class HFPhase23Dataset(Dataset):
    """Phase 2/3 dataset built from either a HuggingFace dataset or JSONL file."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _phase23_from_hf(cfg: Phase23DataCfg, arnl_config: ARNLConfig, max_len: int, tok) -> List[dict]:
    ds_lib = _require_datasets()
    print(f"  [data_config] Loading HF dataset '{cfg.hf_dataset}'"
          + (f" / {cfg.hf_config_name}" if cfg.hf_config_name else "")
          + f" ({cfg.hf_split}) …")
    ds = ds_lib.load_dataset(
        cfg.hf_dataset,
        cfg.hf_config_name,
        split=cfg.hf_split,
        streaming=cfg.hf_streaming,
    )
    samples = []
    for i, row in enumerate(ds):
        if cfg.hf_max_samples and i >= cfg.hf_max_samples:
            break
        s = _process_phase23_row(row, cfg, arnl_config, max_len, tok)
        if s is not None:
            samples.append(s)
    print(f"  [data_config] Loaded {len(samples):,} Phase 2/3 samples from HF.")
    return samples


def _phase23_from_file(path: str, cfg: Phase23DataCfg, arnl_config: ARNLConfig, max_len: int, tok) -> List[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            s = _process_phase23_row(row, cfg, arnl_config, max_len, tok)
            if s is not None:
                samples.append(s)
    return samples


def build_phase23_dataset(
    data_cfg: DataConfig,
    phase: int,
    arnl_config: ARNLConfig,
) -> Optional[Dataset]:
    """Build a Phase 2 dataset from *data_cfg*.

    Parameters
    ----------
    data_cfg : DataConfig
    phase : int
        2 (Phase 3 removed in V1.1).
    arnl_config : ARNLConfig

    Returns ``None`` if neither HF nor local source is configured.
    """
    cfg: Phase23DataCfg = data_cfg.phase2 if phase == 2 else data_cfg.phase3
    tok = _effective_tokenizer(data_cfg, cfg.tokenizer_name)
    max_len = data_cfg.max_seq_len

    if cfg.hf_dataset:
        samples = _phase23_from_hf(cfg, arnl_config, max_len, tok)
        return HFPhase23Dataset(samples) if samples else None

    if cfg.local_file:
        print(f"  [data_config] Phase {phase} — loading local file: {cfg.local_file}")
        samples = _phase23_from_file(cfg.local_file, cfg, arnl_config, max_len, tok)
        print(f"  [data_config] Loaded {len(samples):,} Phase {phase} samples.")
        return HFPhase23Dataset(samples) if samples else None

    return None


# ════════════════════════════════════════════════════════════════
# Facts — System 2 knowledge base
# ════════════════════════════════════════════════════════════════

def _process_fact_row(row: dict, cfg: FactsDataCfg, tokenizer=None) -> Optional[dict]:
    """Normalise a raw row into a facts dict for populate_map.py."""
    fact: dict = {}

    # anchor_ids
    if cfg.col_text_anchors and tokenizer and cfg.col_text_anchors in row:
        fact["anchor_ids"] = list(tokenizer.encode(row[cfg.col_text_anchors]))
    elif cfg.col_anchor_ids in row:
        fact["anchor_ids"] = list(row[cfg.col_anchor_ids])
    else:
        return None  # required field missing

    if len(fact["anchor_ids"]) < 2:
        return None  # hash requires ≥ 2 anchors

    # target_token_id
    if cfg.col_text_target and tokenizer and cfg.col_text_target in row:
        ids = tokenizer.encode(row[cfg.col_text_target])
        fact["target_token_id"] = int(ids[0]) if ids else 0
    elif cfg.col_target_token_id in row:
        fact["target_token_id"] = int(row[cfg.col_target_token_id])
    else:
        return None  # required field missing

    # optional fields
    if cfg.col_tier and cfg.col_tier in row:
        fact["tier"] = str(row[cfg.col_tier])
    if cfg.col_hard_lock and cfg.col_hard_lock in row:
        fact["hard_lock"] = bool(row[cfg.col_hard_lock])
    if cfg.col_s_base and cfg.col_s_base in row:
        fact["s_base"] = int(row[cfg.col_s_base])

    return fact


def build_facts_list(data_cfg: DataConfig) -> Optional[List[dict]]:
    """Return a list of fact dicts suitable for populate_map.py.

    Loads from HuggingFace if ``facts.hf_dataset`` is set, otherwise
    from ``facts.local_file``.  Returns ``None`` if neither is configured.
    """
    cfg = data_cfg.facts
    tok = _effective_tokenizer(data_cfg, cfg.tokenizer_name)

    rows: List[dict] = []

    if cfg.hf_dataset:
        ds_lib = _require_datasets()
        print(f"  [data_config] Loading facts from HF dataset '{cfg.hf_dataset}'"
              + (f" / {cfg.hf_config_name}" if cfg.hf_config_name else "")
              + f" ({cfg.hf_split}) …")
        ds = ds_lib.load_dataset(
            cfg.hf_dataset,
            cfg.hf_config_name,
            split=cfg.hf_split,
            streaming=cfg.hf_streaming,
        )
        for i, row in enumerate(ds):
            if cfg.hf_max_samples and i >= cfg.hf_max_samples:
                break
            rows.append(row)

    elif cfg.local_file:
        print(f"  [data_config] Facts — loading local file: {cfg.local_file}")
        with open(cfg.local_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            rows = json.loads(content)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    rows.append(json.loads(line))
    else:
        return None

    facts = []
    for i, row in enumerate(rows):
        fact = _process_fact_row(row, cfg, tok)
        if fact is not None:
            facts.append(fact)
        else:
            print(f"  [data_config] WARNING: Skipped fact row {i} (missing required fields).")

    print(f"  [data_config] Built {len(facts):,} facts from data config.")
    return facts


def build_facts_from_phase1(
    data_cfg: "DataConfig",
    arnl_config: "ARNLConfig",
    max_facts: Optional[int] = None,
) -> Optional[List[dict]]:
    """Extract SHADE node facts from the Phase 1 text dataset via sliding windows.

    Loads the same text corpus used for Phase 1 EMLM pretraining, tokenizes
    each document, then slides a window of ``(k_anchors + 1)`` tokens across
    the sequence.  The first ``k_anchors`` tokens become ``anchor_ids`` and
    the last token becomes ``target_token_id``.

    This lets you build a System 2 knowledge base from the same dataset you
    train System 1 on, without a separate structured facts file.

    Returns ``None`` if no Phase 1 source is configured or no tokenizer
    is available.
    """
    cfg = data_cfg.phase1
    if not cfg.hf_dataset and not cfg.local_file:
        return None

    tok = _effective_tokenizer(data_cfg, cfg.tokenizer_name)
    if tok is None:
        print("  [data_config] WARNING: no tokenizer — cannot extract facts from Phase 1 text.")
        return None

    if cfg.hf_dataset:
        texts = _texts_from_hf(cfg)
    else:
        print(f"  [data_config] Facts (phase1 fallback) — loading local file: {cfg.local_file}")
        texts = _texts_from_file(cfg.local_file)

    k = arnl_config.k_anchors       # number of anchors per edge
    vs = arnl_config.vocab_size
    window = k + 1
    limit = max_facts or data_cfg.facts.hf_max_samples

    print(f"  [data_config] Extracting SHADE facts from {len(texts):,} Phase 1 documents"
          f" (k_anchors={k}, limit={limit or 'none'}) …")

    facts: List[dict] = []
    for text in texts:
        ids = _clip_ids(tok.encode(text), vs)
        if len(ids) < window:
            continue
        # Non-overlapping windows to avoid near-duplicate edges
        for i in range(0, len(ids) - window + 1, window):
            anchor_ids = ids[i: i + k]
            target = ids[i + k]
            if len(anchor_ids) < 2:
                continue
            facts.append({
                "anchor_ids": anchor_ids,
                "target_token_id": target,
                "tier": "domain",
            })
            if limit and len(facts) >= limit:
                break
        if limit and len(facts) >= limit:
            break

    print(f"  [data_config] Extracted {len(facts):,} SHADE facts from Phase 1 dataset.")
    return facts


# ════════════════════════════════════════════════════════════════
# CLI — generate an example config file
# ════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="ARNL data configuration utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--init-config", metavar="PATH", default=None,
        help="Write an example DataConfig JSON to PATH and exit.",
    )
    parser.add_argument(
        "--show-config", metavar="PATH", default=None,
        help="Load and pretty-print a DataConfig JSON file.",
    )
    args = parser.parse_args()

    if args.init_config:
        cfg = DataConfig.example()
        cfg.save(args.init_config)
        print("  Edit the file to point at your datasets, then pass it to train.py / populate_map.py:")
        print(f"    python scripts/train.py --data-config {args.init_config} --checkpoint-dir ckpt/")
        print(f"    python scripts/populate_map.py --data-config {args.init_config} --shade-file ckpt/shade.json")
        return

    if args.show_config:
        cfg = DataConfig.load(args.show_config)
        print(json.dumps(asdict(cfg), indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
16