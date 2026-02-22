#!/usr/bin/env python3
"""
Arnold — ARNL V1.0 Demo
========================

Demonstrates the full architecture:
    1. Create an Arnold model (tiny config for CPU demo)
    2. Pre-populate some axioms into System 2
    3. Run the 10-step generative loop
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
from arnl import Arnold, ARNLConfig


def main():
    print("=" * 60)
    print("  Arnold — ARNL V1.0 Architecture Demo")
    print("=" * 60)

    # ── 1. Create a tiny model ──────────────────────────────────
    config = ARNLConfig.arnold_tiny()
    model = Arnold(config)

    print(f"\n{model}")
    report = model.param_report()
    print(f"\nParameter breakdown:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # ── 2. Pre-populate axioms ──────────────────────────────────
    print("\n" + "─" * 60)
    print("Pre-populating System 2 with axioms...")

    # Simulate: "capital" + "France" → "Paris" (token IDs are arbitrary here)
    capital_id, france_id, paris_id = 100, 200, 300

    edge = model.prepopulate_axiom(
        anchor_ids=[capital_id, france_id],
        target_token_id=paris_id,
        tier_label="axiom",
    )
    print(f"  Inserted: Hash(capital, France) → Paris")
    print(f"  Edge: S_base={edge.s_base}, tier={edge.tier}, hard_locked={edge.hard_locked}")

    # Add a few more facts
    model.prepopulate_axiom([101, 201], target_token_id=301, tier_label="axiom")  # water→H2O
    model.prepopulate_axiom([102, 202], target_token_id=302, tier_label="domain")  # ARNL→architecture

    print(f"  Total edges in System 2: {model.hyper_map.size}")

    # ── 3. Inspect belief state ─────────────────────────────────
    print("\n" + "─" * 60)
    print("Belief State (Granular Semantic Auditability):")
    beliefs = model.hyper_map.export_belief_state()
    for b in beliefs:
        print(f"  key={b['key']} | token={b['target_token_id']} | "
              f"S_base={b['s_base']} | tier={b['tier']} | "
              f"α={b['alpha']:.4f} | locked={b['hard_locked']}")

    # ── 4. Run a forward pass (no injection) ────────────────────
    print("\n" + "─" * 60)
    print("Running System 1 forward pass (no injection)...")

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        result = model(input_ids)
    logits = result["logits"]
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # ── 5. Run generation (autoregressive loop) ─────────────────
    print("\n" + "─" * 60)
    print("Running autoregressive generation...")

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated, gen_log = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True,
    )
    print(f"  Prompt tokens:    {prompt[0].tolist()}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    print(f"  Steps logged:     {len(gen_log)}")

    # Show generation diagnostics
    vacuum_count = sum(1 for s in gen_log if s["is_vacuum"])
    injection_count = sum(1 for s in gen_log if s["alpha"] > 0.01)
    print(f"  Semantic vacuum events: {vacuum_count}")
    print(f"  Injection events:       {injection_count}")

    # ── 6. Hard Lock / Hard Delete demo ─────────────────────────
    print("\n" + "─" * 60)
    print("Hard Lock / Hard Delete operations:")

    # Hard Lock
    success = model.hard_lock_fact([capital_id, france_id])
    print(f"  Hard Lock (capital, France): {'OK' if success else 'FAIL'}")

    # Hard Delete another fact
    success = model.hard_delete_fact([101, 201])
    print(f"  Hard Delete (water, H2O):    {'OK' if success else 'FAIL'}")

    print(f"  Edges remaining: {model.hyper_map.size}")
    print(f"  Refuted pathways: {len(model.hyper_map.refuted_pathways)}")

    # ── 7. Decay Engine demo ────────────────────────────────────
    print("\n" + "─" * 60)
    print("Decay Engine demo:")

    # Simulate hits and misses on the France edge
    france_edge = model.hyper_map.lookup([capital_id, france_id])
    if france_edge:
        for i in range(10):
            is_hit = (i % 3 == 0)  # alternating pattern
            status = model.decay_engine.update(france_edge, is_hit, model.hyper_map)
            print(f"  Step {i}: {'HIT' if is_hit else 'MISS'} → {status} "
                  f"(S_base={france_edge.s_base}, S_overflow={france_edge.s_overflow})")

    # ── 8. Knowledge ingestion demo ─────────────────────────────
    print("\n" + "─" * 60)
    print("Knowledge ingestion (via Reasoning Head):")

    context = torch.randint(3, config.vocab_size, (10,))  # random context
    target_id = 500
    new_edge = model.ingest_knowledge(context, target_id)
    if new_edge:
        print(f"  Ingested: key={new_edge.key}, S_base={new_edge.s_base}, tier={new_edge.tier}")
    else:
        print("  Rejected by consistency gate (expected for random data)")

    # ── 9. System 2 diagnostics ─────────────────────────────────
    print("\n" + "─" * 60)
    print("System 2 Diagnostics:")
    diag = model.hyper_map.diagnostics
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # ── 10. Save/Load demo ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("Save/Load test:")

    save_dir = "/tmp/arnold_demo"
    model.save_pretrained(save_dir)
    print(f"  Saved to {save_dir}")

    loaded = Arnold.from_pretrained(save_dir)
    print(f"  Loaded: {loaded}")
    print(f"  Edges preserved: {loaded.hyper_map.size}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
