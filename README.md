# Arnold — ARNL V1.1

## NOTE

I am abandoning this arch in search of something simpler, but with a similar philosophy. Make Small language Models actually viable, safe, private and legal options in the AI space. This README is not up-to-date, please refer to the ARNL 1.1 spec sheet for how the arch really works.

**Arnold** is a neuro-symbolic language model implementing the **ARNL** (Axiomatic-Recurrent Neural Logic) architecture. It pairs a neural fluency backbone with a non-neural factual memory system to produce outputs that are both fluent and verifiably grounded.

> *System 1 determines **HOW** something is said.*
> *System 2 determines **WHAT** is said.*
> *The Reasoning Head determines **WHETHER** and **WHEN** System 2 influences System 1.*

---

## Architecture Overview

ARNL is a three-component heterogeneous system. The components are trained independently and fused at inference through the Injection Bridge.

```
┌─────────────────────────────────────┐
│           Arnold (ARNL V1.0)        │
│                                     │
│  System 1 (LLSU)    ~80% of params  │  ← Fluency Core (GLA/SSM)
│  Reasoning Head     ~15% of params  │  ← Control Logic + Classifier
│  Injection Bridge    ~5% of params  │  ← W_proj (semantic → syntactic)
│                                     │
│  System 2  ──── off-budget ────     │  ← Factual Memory (non-neural)
│  Decay Engine ─ off-budget ────     │  ← Memory Lifecycle (non-neural)
└─────────────────────────────────────┘
```

### System 1 — LLSU (Fluency Core)

A Gated Linear Attention (GLA) / SSM recurrent backbone. Determines fluency, grammar, and style. Trained on broad text corpora (Phase 1 EMLM). Parameters are frozen in Phase 2/3; LoRA adapters are grafted instead.

**Default config** (`arnold_7b`, reference scale):

| Hyperparameter | Value |
|---|---|
| `d_model` | 4096 |
| `n_layers` | 32 |
| `d_ffn` (SwiGLU) | 11008 |
| `d_state` | 512 |
| `n_heads` | 32 |

### System 2 — Hyper-Adjacency Map (Reliability Core)

A content-addressable hash map of discrete factual pathways (**Hyperedges**). Not a neural network — no gradients, no weights. Entirely deterministic.

**Hyperedge:**
```
H = { Hash(C_salient) → v_target,  S_base,  S_overflow }
```

**Tiers:**

| Tier | `S_base` Range | Behaviour |
|---|---|---|
| Incubation | 1 – 49 | No decay; `+1` on hit; promotes at 50 |
| Axiomatic | 50 – 1000 | 1-in-5 decay rule; hard-lockable |
| Overflow | `S_overflow` 0 → `S_max` | `+20` hit / `−20` miss (5-20 rule) |

`S_max` default: **2000**. Overflow ceiling prevents frequency-based dominance.

Supports **Hard Lock** (immutable pathway) and **Hard Delete** (refuted pathway removal).

### Reasoning Head (Control Logic)

A small Transformer that reads the current context and decides whether System 2 should inject a token. Gates new knowledge through tier-aware consistency thresholds:

| Tier | Threshold |
|---|---|
| Axiom (physical constants, definitions) | τ = 0.92 |
| Domain (scientific / technical claims) | τ = 0.80 |
| User (preferences, personal context) | τ = 0.65 |

Conflict detection uses inversion dead-zone δ = 0.25.

### Injection Bridge — W_proj

A single linear projection `d_model × d_semantic` that translates System 2's hyperspherical embedding into System 1's residual stream. Function-word suppression α = 0.2.

### Decay Engine

Deterministic counter logic managing Hyperedge lifetimes. No trainable parameters.

- **Incubation**: zero decay, `+1/hit`, promotes to Axiomatic at `S_base = 50`
- **Axiomatic**: 1-in-5 miss rule (`−1` per 5 consecutive misses)
- **Overflow**: `+20/hit`, `−20/miss`, capped at `S_max`

---

## Generative Loop (10 Steps)

Every autoregressive step executes:

1. Input ingestion
2. Semantic Saliency Filter
3. Hyperedge retrieval from System 2
4. Conflict scan
5. System 1 forward pass
6. Sigmoid ramp computation
7. Latent injection via W_proj
8. Token generation
9. Post-generation reinforcement (Decay Engine update)
10. Autoregressive append

---

## Model Presets

| Preset | Approx. Params | `d_model` | `n_layers` | `d_reasoning` |
|---|---|---|---|---|
| `arnold_tiny` | ~3.4M (dev) | 256 | 4 | 128 |
| `arnold_1b` | ~1B | 1536 | 16 | 768 |
| `arnold_3b` | ~3B | 2048 | 24 | 1024 |
| `arnold_7b` | ~7B (reference) | 4096 | 32 | 1536 |
| `arnold_13b` | ~13B | 5120 | 40 | 2048 |
| `arnold_24b` | ~24B | 6144 | 48 | 2560 |

Actual `arnold_tiny` parameter count (verified from demo): **3,434,373**
Budget split: System 1 83.9% · Reasoning Head 15.1% · W_proj 1.0%

---

## Project Structure

```
arnold/
├── arnl/                       # Core package
│   ├── __init__.py             # Public API
│   ├── config.py               # ARNLConfig + presets
│   ├── model.py                # Arnold (full model)
│   ├── system1.py              # LLSU — GLA/SSM backbone
│   ├── system2.py              # HyperAdjacencyMap + Hyperedge
│   ├── reasoning_head.py       # ReasoningHead + InjectionPlan
│   ├── injection_bridge.py     # InjectionBridge (W_proj)
│   ├── decay_engine.py         # DecayEngine
│   ├── training.py             # Training utilities
│   └── utils.py                # semantic_hash, apply_lora, etc.
├── scripts/
│   ├── train.py                # 3-phase training script
│   ├── populate_map.py         # Hyper-Adjacency Map population tool
│   └── data_config.py          # DataConfig + HF dataset helpers
├── data/
│   ├── my_data.json            # Example DataConfig (wikitext Phase 1)
│   ├── example_facts.json      # Example facts (JSON array)
│   └── example_facts.jsonl     # Example facts (JSONL)
├── examples/
│   └── demo.py                 # Full architecture demo (runs on CPU)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `datasets` (HuggingFace, for HF dataset support)
- `transformers` (HuggingFace, for GPT-2 tokenizer)

---

## Quick Start

### Run the demo (CPU, no data required)

```bash
python examples/demo.py
```

This creates an `arnold_tiny` model, inserts three axioms into System 2, runs the 10-step generative loop, and demonstrates Hard Lock / Hard Delete / Decay Engine operations.

### Programmatic usage

```python
from arnl import Arnold, ARNLConfig

# Create a model
config = ARNLConfig.arnold_7b()
model = Arnold(config)

# Insert a factual axiom
model.prepopulate_axiom(
    anchor_ids=[100, 200],   # token IDs for "capital", "France"
    target_token_id=300,     # token ID for "Paris"
    tier_label="axiom",
)

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 32))
output = model(input_ids)
logits = output["logits"]  # shape: [1, 32, vocab_size]
```

---

## Training

Training proceeds in three phases. All phases are managed by a single script with automatic resume.

```bash
# Full pipeline — synthetic data smoke test
python scripts/train.py --synthetic --checkpoint-dir checkpoints/

# Resume from wherever training left off
python scripts/train.py --synthetic --checkpoint-dir checkpoints/

# Phase 1 on a real text corpus
python scripts/train.py --phase 1 \
    --phase1-data data/corpus.txt \
    --checkpoint-dir checkpoints/ --epochs 3

# Train with a HuggingFace DataConfig
python scripts/train.py \
    --data-config data/my_data.json \
    --checkpoint-dir checkpoints/

# Use a specific model preset
python scripts/train.py --synthetic --preset arnold_1b \
    --checkpoint-dir checkpoints/1b/
```

### Phase Summary

| Phase | Trains | Data Format |
|---|---|---|
| 1 | System 1 (LLSU) — full weights | Plain text / HF dataset; EMLM masking applied automatically |
| 2 | LoRA adapters + Reasoning Head + W_proj | JSONL facts (`input_ids`, `target_id`, `anchor_ids`, `label_ids`) |
| 3 | Same as Phase 2, continued | Same as Phase 2 |

**EMLM entity tokens** used during Phase 1 masking: `[ENTITY_NOUN]`, `[ENTITY_NUM]`, `[ENTITY_CONCEPT]`, `[ENTITY_DOMAIN]`

**LoRA config** (Phase 2/3): rank = 16, alpha = 32.0, dropout = 0.05

### Checkpoint layout

```
checkpoints/
├── config.json
├── data_config.json            # Copy of the DataConfig used
├── phase1/
│   ├── system1.pt
│   ├── optimizer.pt
│   └── state.json              # {epoch, global_step, best_loss, done}
├── phase2/
│   ├── system1_lora.pt
│   ├── reasoning_head.pt
│   ├── injection_bridge.pt
│   ├── optimizer.pt
│   └── state.json
└── phase3/
    └── ...
```

---

## Populating System 2

`populate_map.py` ingests facts into the Hyper-Adjacency Map.

```bash
# Insert from a JSON facts file (idempotent — skips existing keys)
python scripts/populate_map.py \
    --facts-file data/facts.json \
    --checkpoint-dir checkpoints/

# Rebuild the map from scratch using the Phase 1 HF dataset
python scripts/populate_map.py \
    --checkpoint-dir checkpoints/ \
    --mode replace

# Validate new facts through the Reasoning Head gate
python scripts/populate_map.py \
    --facts-file data/facts.json \
    --checkpoint-dir checkpoints/ \
    --use-gate

# View map statistics without modifying anything
python scripts/populate_map.py \
    --checkpoint-dir checkpoints/ \
    --stats-only

# Export full belief state to JSON
python scripts/populate_map.py \
    --checkpoint-dir checkpoints/ \
    --export-beliefs beliefs_export.json
```

**Modes:**

| Mode | Behaviour |
|---|---|
| `insert` (default) | Insert new entries; skip duplicate keys |
| `upsert` | Insert new; update `S_base` / hard-lock on key collision |
| `replace` | Delete existing map, rebuild from scratch |

When `--facts-file` is not given, the script falls back to auto-detection in this order:
1. `facts.hf_dataset` in the DataConfig
2. `facts.local_file` in the DataConfig
3. Derives hyperedges from the Phase 1 HF corpus via a sliding token window

---

## DataConfig

`scripts/data_config.py` provides a portable configuration format for all data sources:

```json
{
  "tokenizer_name": "gpt2",
  "max_seq_len": 512,
  "phase1": {
    "hf_dataset": "wikitext",
    "hf_config_name": "wikitext-2-raw-v1",
    "hf_split": "train",
    "hf_max_samples": 50000
  },
  "phase2": { "local_file": "data/example_facts.jsonl" },
  "phase3": { "local_file": "data/example_facts.jsonl" },
  "facts": {
    "hf_dataset": null,
    "local_file": null
  }
}
```

When `facts.hf_dataset` and `facts.local_file` are both `null`, `populate_map.py` extracts hyperedges directly from the Phase 1 text corpus using a sliding `(k_anchors + 1)`-token window. With `k_anchors = 3` (tiny) or `5` (full configs), this produces dense factual pathways aligned with what System 1 was trained on.

Initialise a fresh config:

```bash
python scripts/data_config.py --init-config data/my_config.json
python scripts/data_config.py --show-config data/my_config.json
```

---

## System 2 Facts Format

**JSON array** (`--facts-file *.json`):
```json
[
  {
    "anchor_ids":      [100, 200],
    "target_token_id": 300,
    "tier":            "axiom",
    "hard_lock":       true,
    "s_base":          1000
  }
]
```

**JSONL** (`--facts-file *.jsonl`), one record per line:
```jsonl
{"anchor_ids": [100, 200], "target_token_id": 300, "tier": "axiom"}
{"anchor_ids": [101, 201], "target_token_id": 301, "tier": "domain"}
```

`anchor_ids` must contain at least 2 entries. Valid tiers: `"axiom"`, `"domain"`, `"user"`.

---

## Granular Semantic Auditability

Every factual claim is fully inspectable at runtime:

```python
beliefs = model.hyper_map.export_belief_state()
for b in beliefs:
    print(b["key"], b["target_token_id"], b["s_base"], b["tier"], b["alpha"])
```

Hard Lock / Hard Delete:

```python
model.hyper_map.hard_lock(key)    # Freeze — immune to decay and update
model.hyper_map.hard_delete(key)  # Refute — removes pathway permanently
```

---

## Public API

```python
from arnl import (
    Arnold,            # Full model
    ARNLConfig,        # Configuration + presets
    LLSU,              # System 1 backbone
    HyperAdjacencyMap, # System 2 hash map
    Hyperedge,         # Single factual pathway record
    ReasoningHead,     # Consistency gate + context encoder
    InjectionBridge,   # W_proj
    DecayEngine,       # Memory lifecycle manager
)
```

---

## Docs

See docs/ for any info that documentation of the model may contain

---

## License

See repository for license details.
