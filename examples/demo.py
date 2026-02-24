#!/usr/bin/env python3
"""
Arnold — ARNL V1.1 Demo (SHADE Edition)
=========================================

Demonstrates the full architecture:
    1. Create an Arnold model (tiny config for CPU demo)
    2. Pre-populate SHADE nodes
    3. Run the V1.1 generative loop (gap detection → SHADE inject)
    4. Inspect the belief state (Granular Semantic Auditability)
    5. Show Hard Lock / Hard Delete operations
    6. Run a quick EMLM training loop

Usage:
    python examples/demo.py
"""

import sys
import os

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from arnl import Arnold, ARNLConfig, SHADE, ShadeNode


def main():
    print("=" * 60)
    print("  Arnold — ARNL V1.1 Architecture Demo (SHADE Edition)")
    print("=" * 60)

    # ── 1. Create a tiny model ──────────────────────────────────
    config = ARNLConfig.arnold_tiny()
    model = Arnold(config)

    print(f"\n{model}")
    report = model.param_report()
    print(f"\nParameter breakdown:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # ── 2. Pre-populate SHADE nodes ─────────────────────────────
    print("\n" + "─" * 60)
    print("Pre-populating SHADE with knowledge nodes...")

    # Simulate: "capital of France" → "Paris" (token IDs are arbitrary)
    paris_emb = np.random.randint(-128, 127, size=config.d_semantic, dtype=np.int8)
    node_paris = model.shade.insert_node(
        concept_emb=paris_emb,
        target_dist={300: 0.9, 301: 0.05, 302: 0.05},  # Paris=300
        context_window=[100, 200],  # capital=100, France=200
        s_base=1500,
        tier_label="axiom",
    )
    model.shade.hard_lock(node_paris.node_id)
    print(f"  Inserted: Paris node (id={node_paris.node_id})")
    print(f"  tier={node_paris.tier}, s_base={node_paris.s_base}, hard_locked={node_paris.hard_locked}")

    # Add more nodes
    h2o_emb = np.random.randint(-128, 127, size=config.d_semantic, dtype=np.int8)
    node_h2o = model.shade.insert_node(
        concept_emb=h2o_emb,
        target_dist={400: 0.85, 401: 0.1, 402: 0.05},  # H2O=400
        context_window=[110, 210],
        s_base=1500,
    )

    arnl_emb = np.random.randint(-128, 127, size=config.d_semantic, dtype=np.int8)
    node_arnl = model.shade.insert_node(
        concept_emb=arnl_emb,
        target_dist={500: 0.8, 501: 0.15, 502: 0.05},  # ARNL=500
        context_window=[120, 220],
        s_base=800,
    )

    print(f"  Total SHADE nodes: {model.shade.size}")

    # Add an edge between Paris and H2O (for demo)
    model.shade.add_or_strengthen_edge(node_paris.node_id, node_h2o.node_id)
    print(f"  Added edge: Paris ↔ H2O")

    # ── 3. Inspect belief state ─────────────────────────────────
    print("\n" + "─" * 60)
    print("Belief State (Granular Semantic Auditability):")
    beliefs = model.shade.export_belief_state()
    for b in beliefs[:5]:  # Show first 5
        print(f"  id={b['node_id']} | s_base={b['s_base']} | tier={b['tier']} | "
              f"α={b['alpha']:.4f} | locked={b['hard_locked']}")

    # ── 4. Run a forward pass ───────────────────────────────────
    print("\n" + "─" * 60)
    print("Running System 1 forward pass...")

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        logits, states = model(input_ids)
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # ── 5. Gap detection demo ───────────────────────────────────
    print("\n" + "─" * 60)
    print("Gap detection (H_SAC entropy)...")

    with torch.no_grad():
        logits_single, _ = model(input_ids)
        h_sac = model.compute_h_sac(logits_single[0, -1])
        print(f"  H_SAC = {h_sac:.4f} (threshold h_gap = {config.h_gap})")
        print(f"  Gap detected: {'YES' if h_sac > config.h_gap else 'NO'}")

    # ── 6. Run generation (autoregressive loop) ─────────────────
    print("\n" + "─" * 60)
    print("Running autoregressive generation...")

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
    )
    print(f"  Prompt tokens:    {prompt[0].tolist()}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    print(f"  Total length:     {generated.shape[1]}")

    # ── 7. Hard Lock / Hard Delete demo ─────────────────────────
    print("\n" + "─" * 60)
    print("Hard Lock / Hard Delete operations:")

    # Hard Lock
    model.shade.hard_lock(node_paris.node_id)
    print(f"  Hard Lock (Paris): node_id={node_paris.node_id} locked={node_paris.hard_locked}")

    # Hard Delete (override to refuted)
    # Hard Override (sets s_base=0, schedules for deletion)
    model.shade.hard_override(node_arnl.node_id)
    print(f"  Hard Override (ARNL): s_base now {node_arnl.s_base}")

    # Delete
    model.shade.delete_node(node_arnl.node_id)
    print(f"  Deleted ARNL node. Remaining: {model.shade.size}")

    # ── 8. Decay Engine demo ────────────────────────────────────
    print("\n" + "─" * 60)
    print("Decay Engine demo:")

    for i in range(10):
        if i % 3 == 0:
            value = "high"
        elif i % 3 == 1:
            value = "low"
        else:
            value = "miss"
        status = model.decay.update_node(node_h2o, value)
        print(f"  Step {i}: {value:4s} → {status} "
              f"(s_base={node_h2o.s_base}, s_overflow={node_h2o.s_overflow})")

    # ── 9. Edge management ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("Edge management:")

    # Strengthen edge
    for _ in range(5):
        model.shade.add_or_strengthen_edge(node_paris.node_id, node_h2o.node_id)
    print(f"  Edges: {model.shade.edge_count}")

    # Decay edges
    model.shade.step_edges([node_paris.node_id, node_h2o.node_id], step=100)
    print(f"  Edges after decay: {model.shade.edge_count}")

    # ── 10. Save/Load demo ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("Save/Load test:")

    save_dir = "/tmp/arnold_demo_v1.1"
    os.makedirs(save_dir, exist_ok=True)

    # Save SHADE
    shade_path = os.path.join(save_dir, "shade.json")
    model.shade.save(shade_path)
    print(f"  SHADE saved to {shade_path}")

    # Reload
    shade2 = SHADE(config)
    shade2.load(shade_path)
    print(f"  SHADE loaded: {shade2.size} nodes, {shade2.edge_count} edges")

    # Conflict scan (check a random embedding against axioms)
    test_emb = np.random.randint(-128, 127, size=config.d_semantic, dtype=np.int8)
    conflicts = model.shade.conflict_scan(test_emb)
    print(f"  Conflict scan: {len(conflicts)} issues found")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
