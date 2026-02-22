#!/usr/bin/env python3
"""
scripts/populate_map.py — Hyper-Adjacency Map Population Tool
==============================================================

Populates System 2 (the Hyper-Adjacency Map) with Hyperedges from a
structured facts file, then saves the updated map to disk.

On every run the script loads the existing map first (if the save file
exists), adds any new facts that are not already present, and saves back.
Re-running with the same file is safe and idempotent.

────────────────────────────────────────────────────────────────
Facts file format (JSON array, --facts-file)
────────────────────────────────────────────────────────────────
[
  {
    "anchor_ids":      [100, 200],    // token IDs of semantic anchors
    "target_token_id": 300,           // token ID of the target
    "tier":            "axiom",       // "axiom" | "domain" | "user"
    "hard_lock":       true,          // optional (default false)
    "s_base":          1000           // optional starting strength
  },
  ...
]

"anchor_ids" must contain at least 2 entries (the minimum for a valid hash).
"tier" controls which consistency-gate threshold is applied when the
Reasoning Head is asked to validate the entry (--use-gate mode).

────────────────────────────────────────────────────────────────
Facts file format (JSONL, one record per line, --facts-file *.jsonl)
────────────────────────────────────────────────────────────────
{"anchor_ids": [100, 200], "target_token_id": 300, "tier": "axiom"}
{"anchor_ids": [101, 202], "target_token_id": 301, "tier": "domain"}

────────────────────────────────────────────────────────────────
Operation modes
────────────────────────────────────────────────────────────────
  --mode insert   (default) Insert new entries; skip duplicates by key.
  --mode upsert   Update S_base / hard-lock on key collision.
  --mode replace  Delete the existing map and rebuild from scratch.

────────────────────────────────────────────────────────────────
Usage examples
────────────────────────────────────────────────────────────────
# Populate from a JSON facts file, save map alongside a checkpoint:
    python scripts/populate_map.py \\
        --facts-file data/facts.json \\
        --map-file   checkpoints/phase2/hyper_map.json \\
        --checkpoint-dir checkpoints/

# Load config from a checkpoint dir and validate with Reasoning Head:
    python scripts/populate_map.py \\
        --facts-file data/facts.json \\
        --map-file   checkpoints/hyper_map.json \\
        --checkpoint-dir checkpoints/ \\
        --use-gate

# Show current map stats without changing anything:
    python scripts/populate_map.py \\
        --map-file checkpoints/hyper_map.json \\
        --stats-only

# Hard-lock every existing axiom in the map:
    python scripts/populate_map.py \\
        --map-file checkpoints/hyper_map.json \\
        --lock-all-axioms

# Export the full belief state to a readable JSON file:
    python scripts/populate_map.py \\
        --map-file checkpoints/hyper_map.json \\
        --export-beliefs beliefs_export.json
────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
from dataclasses import fields as dc_fields
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arnl.config import ARNLConfig
from arnl.model import Arnold
from arnl.system2 import HyperAdjacencyMap, Hyperedge
from arnl.utils import semantic_hash
from scripts.data_config import DataConfig, build_facts_list, build_facts_from_phase1

_DATA_CONFIG_FILENAME = "data_config.json"


# ════════════════════════════════════════════════════════════════
# Fact loading
# ════════════════════════════════════════════════════════════════

def _load_facts(path: str) -> List[dict]:
    """Load facts from a JSON array file or a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        # JSON array
        facts = json.loads(content)
    else:
        # JSONL
        facts = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                facts.append(json.loads(line))

    return facts


def _validate_fact(rec: dict, idx: int) -> Optional[str]:
    """Return an error string if the record is malformed, else None."""
    if "anchor_ids" not in rec:
        return f"Record {idx}: missing 'anchor_ids'"
    if len(rec["anchor_ids"]) < 2:
        return f"Record {idx}: 'anchor_ids' must have at least 2 entries"
    if "target_token_id" not in rec:
        return f"Record {idx}: missing 'target_token_id'"
    return None


# ════════════════════════════════════════════════════════════════
# Config / model loading helpers
# ════════════════════════════════════════════════════════════════

def _load_config(checkpoint_dir: Optional[str]) -> ARNLConfig:
    if checkpoint_dir:
        cfg_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                d = json.load(f)
            valid = {fld.name for fld in dc_fields(ARNLConfig)}
            cfg = ARNLConfig(**{k: v for k, v in d.items() if k in valid})
            print(f"  Config loaded from {cfg_path}")
            return cfg
    print("  No config.json found — using arnold_tiny defaults.")
    return ARNLConfig.arnold_tiny()


def _build_model_from_checkpoint(
    checkpoint_dir: str,
    config: ARNLConfig,
    device: str,
) -> Arnold:
    """Build Arnold and load the best available weights from a checkpoint dir."""
    model = Arnold(config)

    # Load Reasoning Head (needed for semantic embeddings)
    for phase in [3, 2, 1]:
        pt = os.path.join(checkpoint_dir, f"phase{phase}", "reasoning_head.pt")
        if os.path.exists(pt):
            model.reasoning_head.load_state_dict(
                torch.load(pt, map_location=device)
            )
            print(f"  Loaded Reasoning Head from phase {phase} checkpoint")
            break
    else:
        print("  No Reasoning Head checkpoint found — using random weights.")

    model.to(device)
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════
# Core insertion loop
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def populate(
    model: Arnold,
    hyper_map: HyperAdjacencyMap,
    facts: List[dict],
    mode: str,
    use_gate: bool,
    device: str,
) -> dict:
    """Insert facts into *hyper_map*.

    Parameters
    ----------
    model : Arnold
        Used only for the semantic embeddings (reasoning_head._embed_semantic).
    hyper_map : HyperAdjacencyMap
    facts : list[dict]
        Validated fact records.
    mode : str
        'insert' | 'upsert' | 'replace'
    use_gate : bool
        If True, run each candidate through the Consistency Gate before inserting.
    device : str

    Returns
    -------
    dict with counts: inserted, skipped, updated, rejected
    """
    counts = {"inserted": 0, "skipped": 0, "updated": 0, "rejected": 0, "invalid": 0}

    for idx, rec in enumerate(facts):
        err = _validate_fact(rec, idx)
        if err:
            print(f"  [WARN] {err}")
            counts["invalid"] += 1
            continue

        anchor_ids: List[int] = rec["anchor_ids"]
        target_id: int = rec["target_token_id"]
        tier: str = rec.get("tier", "domain")
        hard_lock: bool = rec.get("hard_lock", False)
        s_base: int = rec.get("s_base", 1000 if hard_lock else 1)

        key = semantic_hash(anchor_ids)

        # ── Mode: replace cleared map before the loop; just insert ──
        existing = hyper_map._map.get(key)

        if existing is not None and mode == "insert":
            counts["skipped"] += 1
            continue

        # Get the semantic embedding for the target token
        target_tensor = torch.tensor([target_id], device=device, dtype=torch.long)
        v_target = model.reasoning_head._embed_semantic(target_tensor).squeeze(0)

        # Optional Reasoning Head consistency gate
        if use_gate:
            accepted, gate_score, init_sbase = model.reasoning_head.consistency_gate(
                v_target, hyper_map, tier
            )
            if not accepted:
                print(
                    f"  [GATE] Rejected: anchor_ids={anchor_ids} target={target_id} "
                    f"gate={gate_score:.3f}"
                )
                counts["rejected"] += 1
                continue
            # Use the gate's suggested init S_base only if not explicitly provided
            if "s_base" not in rec:
                s_base = max(s_base, init_sbase)

        if existing is not None and mode == "upsert":
            # Update the existing entry
            existing.s_base = s_base
            existing.s_base_original = max(existing.s_base_original, s_base)
            existing.v_target = v_target.detach().clone()
            existing.target_token_id = target_id
            existing.tier_label = tier
            if hard_lock:
                existing.hard_locked = True
            counts["updated"] += 1
            print(f"  [UPDATE] key={key} anchor_ids={anchor_ids} → token={target_id} "
                  f"S_base={s_base} tier={tier}")
        else:
            # Fresh insert
            edge = hyper_map.insert(
                anchor_ids=anchor_ids,
                v_target=v_target,
                target_token_id=target_id,
                s_base=s_base,
                tier_label=tier,
            )
            if hard_lock:
                edge.hard_locked = True
                edge.s_base_original = max(s_base, 1000)
                edge.s_base = max(s_base, 1000)
            counts["inserted"] += 1
            print(f"  [INSERT] key={key} anchor_ids={anchor_ids} → token={target_id} "
                  f"S_base={edge.s_base} tier={tier} locked={hard_lock}")

    return counts


# ════════════════════════════════════════════════════════════════
# Statistics display
# ════════════════════════════════════════════════════════════════

def _print_stats(hyper_map: HyperAdjacencyMap):
    all_edges = hyper_map.get_all_edges()
    incubation = [e for e in all_edges if e.s_base < 50]
    axiomatic  = [e for e in all_edges if 50 <= e.s_base < 1000]
    overflow   = [e for e in all_edges if e.s_base >= 1000]
    locked     = [e for e in all_edges if e.hard_locked]

    print("\n  ─── Hyper-Adjacency Map Statistics ───")
    print(f"  Total edges   : {len(all_edges):>6}")
    print(f"  Incubation    : {len(incubation):>6}  (S_base 1–49)")
    print(f"  Axiomatic     : {len(axiomatic):>6}  (S_base 50–999)")
    print(f"  Overflow      : {len(overflow):>6}  (S_base 1000)")
    print(f"  Hard-Locked   : {len(locked):>6}")
    diag = hyper_map.diagnostics
    print(f"  Lookups       : {diag['total_lookups']:>6}")
    print(f"  Hit rate      : {diag['hit_rate']:>6.1%}")
    print(f"  Semantic vacua: {diag['semantic_vacuum_events']:>6}")
    print(f"  Refuted       : {diag['refuted_pathways_count']:>6}")

    if all_edges:
        print("\n  Top-10 by S_base:")
        top = sorted(all_edges, key=lambda e: (e.s_base, e.s_overflow), reverse=True)[:10]
        for e in top:
            print(
                f"    key={e.key:<20d} token={e.target_token_id:<6d} "
                f"S_base={e.s_base:<5d} S_ov={e.s_overflow:<5d} "
                f"tier={e.tier:<12s} locked={str(e.hard_locked)}"
            )


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def build_args():
    p = argparse.ArgumentParser(
        description="Populate and manage the ARNL Hyper-Adjacency Map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Files ────────────────────────────────────────────────────
    p.add_argument("--data-config", default=None, metavar="PATH",
                   help="Path to a DataConfig JSON file (HuggingFace datasets). "
                        "Generated with: python scripts/data_config.py --init-config PATH")
    p.add_argument("--facts-file", "-f", default=None,
                   help="Path to JSON or JSONL facts file to ingest.")
    p.add_argument("--map-file", "-m", default=None,
                   help="Path to the hyper_map.json save file (default: {checkpoint-dir}/hyper_map.json).")
    p.add_argument("--checkpoint-dir", "-c", default=None,
                   help="Checkpoint directory to load config and model weights from.")

    # ── Mode ─────────────────────────────────────────────────────
    p.add_argument("--mode", default="insert",
                   choices=["insert", "upsert", "replace"],
                   help="insert=skip duplicates, upsert=overwrite, replace=rebuild (default: insert)")

    # ── Gate ─────────────────────────────────────────────────────
    p.add_argument("--use-gate", action="store_true",
                   help="Validate each fact through the Reasoning Head consistency gate before inserting.")

    # ── Utility operations ───────────────────────────────────────
    p.add_argument("--stats-only", action="store_true",
                   help="Print map statistics and exit (no modifications).")
    p.add_argument("--lock-all-axioms", action="store_true",
                   help="Hard-lock every axiom-tier entry (S_base=1000, decay disabled).")
    p.add_argument("--export-beliefs", default=None, metavar="PATH",
                   help="Export full belief state to a JSON file.")
    p.add_argument("--purge-dead", action="store_true",
                   help="Purge all entries with S_base <= 0 before saving.")

    # ── Model ────────────────────────────────────────────────────
    p.add_argument("--preset", default=None,
                   choices=["arnold_tiny","arnold_1b","arnold_3b",
                            "arnold_7b","arnold_13b","arnold_24b"],
                   help="Model preset to use if no config.json found.")
    p.add_argument("--device", default="cpu")

    return p.parse_args()


def main():
    args = build_args()

    print("=" * 60)
    print("  populate_map.py — ARNL Hyper-Adjacency Map Tool")
    print("=" * 60)

    device = args.device

    # ── Resolve defaults ─────────────────────────────────────────
    # Auto-detect data config saved by train.py inside the checkpoint dir
    data_config_path = args.data_config
    if data_config_path is None and args.checkpoint_dir:
        candidate = os.path.join(args.checkpoint_dir, _DATA_CONFIG_FILENAME)
        if os.path.exists(candidate):
            data_config_path = candidate
            print(f"  Auto-detected DataConfig: {candidate}")

    # Default map file: {checkpoint_dir}/hyper_map.json or ./hyper_map.json
    map_file = args.map_file
    if map_file is None:
        if args.checkpoint_dir:
            map_file = os.path.join(args.checkpoint_dir, "hyper_map.json")
        else:
            map_file = "hyper_map.json"
        print(f"  Default map file: {map_file}")

    # ── Config & model ───────────────────────────────────────────
    config = _load_config(args.checkpoint_dir)
    if args.preset and args.checkpoint_dir is None:
        config = getattr(ARNLConfig, args.preset)()
        print(f"  Using preset: {args.preset}")

    # Build model — needed for semantic embeddings
    if args.checkpoint_dir:
        model = _build_model_from_checkpoint(args.checkpoint_dir, config, device)
    else:
        print("  No --checkpoint-dir given — using randomly initialised semantic embeddings.")
        model = Arnold(config)
        model.to(device)
        model.eval()

    # ── Load existing map ────────────────────────────────────────
    hyper_map = HyperAdjacencyMap(config)
    if os.path.exists(map_file):
        hyper_map.load(map_file, device=torch.device(device))
        print(f"  Loaded existing map: {map_file}  ({hyper_map.size} edges)")
    else:
        print(f"  No existing map at {map_file} — starting fresh.")

    # ── Stats only ───────────────────────────────────────────────
    if args.stats_only:
        _print_stats(hyper_map)
        return

    # ── Replace mode — wipe the map ─────────────────────────────
    if args.mode == "replace":
        print(f"  Mode=replace: clearing {hyper_map.size} existing edges.")
        hyper_map._map.clear()

    # ── Populate from DataConfig (HuggingFace) or facts file ────
    facts = None
    facts_label = None

    if data_config_path:
        data_cfg = DataConfig.load(data_config_path)
        print(f"  DataConfig loaded from {data_config_path}")
        # 1. Try explicit facts section (HF dataset or local file)
        facts = build_facts_list(data_cfg)
        facts_label = f"DataConfig facts ({data_config_path})"
        # 2. Fall back: derive hyperedges from the Phase 1 HF text dataset
        if facts is None:
            print("  No explicit facts source — extracting hyperedges from Phase 1 dataset.")
            facts = build_facts_from_phase1(data_cfg, config)
            facts_label = f"Phase 1 dataset ({data_config_path})"

    if facts is None and args.facts_file:
        if not os.path.exists(args.facts_file):
            print(f"  ERROR: facts file not found: {args.facts_file}")
            sys.exit(1)
        facts = _load_facts(args.facts_file)
        facts_label = args.facts_file

    if facts is not None:
        n_facts = len(facts)
        print(f"\n  Ingesting {n_facts} facts from {facts_label}")
        print(f"  Mode: {args.mode}  |  Gate: {'on' if args.use_gate else 'off'}")
        print()

        t0 = time.time()
        counts = populate(model, hyper_map, facts, args.mode, args.use_gate, device)
        elapsed = time.time() - t0

        print(f"\n  ── Ingestion summary ({elapsed:.2f}s) ──")
        for k, v in counts.items():
            print(f"    {k:<10}: {v}")
    else:
        print("  No --facts-file or --data-config provided; skipping ingestion.")

    # ── Utility operations ───────────────────────────────────────
    if args.lock_all_axioms:
        n_locked = 0
        for edge in hyper_map.get_all_axioms(min_sbase=1000):
            if not edge.hard_locked:
                edge.hard_locked = True
                edge.s_base_original = 1000
                n_locked += 1
        print(f"\n  Hard-locked {n_locked} axiom edges.")

    if args.purge_dead:
        n_purged = hyper_map.purge_dead()
        print(f"\n  Purged {n_purged} dead edges (S_base <= 0).")

    # ── Statistics ───────────────────────────────────────────────
    _print_stats(hyper_map)

    # ── Save ─────────────────────────────────────────────────────
    save_dir = os.path.dirname(map_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    hyper_map.save(map_file)
    print(f"\n  Saved map → {os.path.abspath(map_file)}  ({hyper_map.size} edges)")

    # ── Export beliefs ───────────────────────────────────────────
    if args.export_beliefs:
        beliefs_json = hyper_map.export_belief_state_json()
        with open(args.export_beliefs, "w", encoding="utf-8") as f:
            f.write(beliefs_json)
        print(f"  Belief state exported → {os.path.abspath(args.export_beliefs)}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
