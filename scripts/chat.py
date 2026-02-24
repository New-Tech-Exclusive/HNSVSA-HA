#!/usr/bin/env python3
"""
scripts/chat.py — Interactive Chat with ARNL V1.1
===================================================

Loads the model from a checkpoint directory and provides an
interactive REPL for generation with SHADE integration.

Usage:
    python scripts/chat.py --checkpoint-dir checkpoints/
"""

import argparse
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arnl import Arnold, ARNLConfig
from arnl.salt_tokenizer import SALTTokenizer


def load_arnl_from_checkpoint(checkpoint_dir: str, device: str) -> Arnold:
    from dataclasses import fields

    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    valid_keys = {f.name for f in fields(ARNLConfig)}
    config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = ARNLConfig(**config_dict)

    model = Arnold(config)

    # Load Phase 1 (Base System 1)
    p1_path = os.path.join(checkpoint_dir, "phase1", "system1.pt")
    if os.path.exists(p1_path):
        print(f"  Loading Phase 1 System 1 from {p1_path}")
        model.system1.load_state_dict(
            torch.load(p1_path, map_location=device, weights_only=True)
        )

    # Check for Phase 2 LoRA weights
    p2_path = os.path.join(checkpoint_dir, "phase2", "system1_lora.pt")
    if os.path.exists(p2_path):
        from arnl.utils import apply_lora
        apply_lora(model.system1, config)
        print(f"  Loading Phase 2 LoRA weights from {p2_path}")
        model.system1.load_state_dict(
            torch.load(p2_path, map_location=device, weights_only=True)
        )

    # Load SHADE database
    shade_path = os.path.join(checkpoint_dir, "shade.json")
    if os.path.exists(shade_path):
        print(f"  Loading SHADE from {shade_path}")
        model.shade.load(shade_path)

    model.to(device)
    model.init_shade()  # Give SHADE access to embedding table
    print("  Model loaded successfully.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with ARNL V1.1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--preset", type=str, default="arnold_tiny")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  Arnold — ARNL V1.1 Interactive Chat (SHADE Edition)")
    print("=" * 60)

    # ── 1. Load Tokenizer ───────────────────────────────────────
    tokenizer_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tokenizer",
    )
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print(f"  Loading SALT tokenizer from {tokenizer_dir}")
        tokenizer = SALTTokenizer(tokenizer_dir)
        _use_salt = True
    else:
        print("  SALT tokenizer not found — falling back to GPT-2.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _use_salt = False

    # ── 2. Load Model ───────────────────────────────────────────
    if os.path.exists(os.path.join(args.checkpoint_dir, "config.json")):
        print(f"  Loading model from {args.checkpoint_dir}")
        model = load_arnl_from_checkpoint(args.checkpoint_dir, args.device)
    else:
        print(f"  No checkpoint found. Creating fresh {args.preset} model.")
        config = getattr(ARNLConfig, args.preset)()
        model = Arnold(config).to(args.device)
        model.init_shade()

    model.eval()
    model.param_report()

    # Load SALT token classes for gap detection
    token_classes_path = os.path.join(tokenizer_dir, "token_classes.npy")
    if os.path.exists(token_classes_path):
        import numpy as np
        tc = np.load(token_classes_path)
        model.load_salt_classes(torch.from_numpy(tc).long())
        print(f"  Loaded token classes for gap detection")

    print(f"\n  SHADE: {model.shade.size} nodes, {model.shade.edge_count} edges")
    print(f"  Device: {args.device}")
    print("\n  Commands:")
    print("    /quit, /exit  — Exit")
    print("    /beliefs      — Show SHADE belief state")
    print("    /stats        — Show SHADE diagnostics")
    print("    /save         — Save SHADE to checkpoint dir")
    print("-" * 60)

    # ── 3. Chat Loop ────────────────────────────────────────────
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            break

        if user_input.lower() == "/beliefs":
            beliefs = model.shade.export_belief_state()
            if not beliefs:
                print("  [SHADE is empty]")
            else:
                for b in beliefs[:20]:
                    token_id = b.get("primary_target", -1)
                    token_str = tokenizer.decode([token_id]) if token_id >= 0 else "?"
                    print(
                        f"  id={b['node_id']} | '{token_str}' ({token_id}) | "
                        f"S_base={b['s_base']} | tier={b['tier']} | "
                        f"α={b['alpha']:.4f} | locked={b['hard_locked']}"
                    )
                if len(beliefs) > 20:
                    print(f"  ... and {len(beliefs) - 20} more nodes")
            continue

        if user_input.lower() == "/stats":
            diag = model.shade.diagnostics
            for k, v in diag.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            continue

        if user_input.lower() == "/save":
            shade_path = os.path.join(args.checkpoint_dir, "shade.json")
            model.shade.save(shade_path)
            print(f"  SHADE saved → {shade_path}")
            continue

        # ── Generate Response ───────────────────────────────────
        input_ids_list = tokenizer.encode(user_input)
        input_ids = torch.tensor([input_ids_list], device=args.device)
        input_ids = torch.clamp(input_ids, max=model.config.vocab_size - 1)

        with torch.no_grad():
            eos_id = tokenizer.eos_id if _use_salt else tokenizer.eos_token_id
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_token_id=eos_id,
            )

        response_ids = generated[0][input_ids.shape[1]:]
        if _use_salt:
            response_text = tokenizer.decode(response_ids.tolist())
        else:
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        print(f"\nArnold: {response_text}")

        # Print SHADE diagnostics for this turn
        diag = model.shade.diagnostics
        if diag["total_hits"] > 0:
            print(f"  [SHADE: {diag['total_hits']} hits, "
                  f"{diag['total_abstentions']} abstentions, "
                  f"hit_rate={diag['hit_rate']:.2%}]")


if __name__ == "__main__":
    main()
