#!/usr/bin/env python3
"""
Interactive chat / text-completion script for HybridNSVSA-HA.

Usage:
  # Interactive REPL
  python chat.py --checkpoint checkpoints/best.pt

  # Single prompt
  python chat.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

  # Tune sampling
  python chat.py --checkpoint checkpoints/best.pt --temperature 0.8 --top_p 0.95

  # Without a checkpoint (random weights, for smoke-testing the pipeline)
  python chat.py --no_checkpoint --d_model 256 --num_layers 4 --num_heads 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig
from nsvsa_ha.tokenizer import BaseTokenizer, load_tokenizer, tokenizer_compatible


DEFAULT_CUSTOM_TOKENIZER_JSON = "tokenizers/vsa65k_mix/tokenizer.json"
DEFAULT_CUSTOM_TOKENIZER_META = "tokenizers/vsa65k_mix/tokenizer_meta.json"


# ─── Load ────────────────────────────────────────────────────────────────────

def load_model(args, tok_info: dict) -> tuple[HybridNSVSA, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint and not args.no_checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        config = HybridNSVSAConfig(**cfg_dict)
        model = HybridNSVSA(config)
        model.load_state_dict(ckpt["model"])
        ckpt_tok = ckpt.get("tokenizer")
        if ckpt_tok and not tokenizer_compatible(ckpt_tok, tok_info):
            msg = (
                "Checkpoint tokenizer metadata does not match runtime tokenizer. "
                f"checkpoint={ckpt_tok} runtime={tok_info}"
            )
            if args.strict_tokenizer_match:
                raise ValueError(msg)
            print(f"WARNING: {msg}")
        step = ckpt.get("step", "?")
        print(f"Loaded checkpoint: {args.checkpoint}  (step {step})")
    else:
        config = HybridNSVSAConfig(
            d_model=args.d_model,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
            window_size=args.window_size,
            group_size=args.group_size,
        )
        model = HybridNSVSA(config)
        print("No checkpoint — using random weights.")

    model = model.to(device).eval()
    n = model.num_parameters()
    print(f"Model: {n:,} params  |  device: {device}")
    return model, device


# ─── Generation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_streaming(
    model: HybridNSVSA,
    tokenizer: BaseTokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Cached autoregressive generation with token-by-token streaming to stdout.

    Phase 1 (prefill):  Full forward on the prompt → builds KV + VSA cache.
    Phase 2 (decode):   One token at a time through the cached forward.
                        Cost per token: O(W) attention + O(1) VSA update.

    Returns the full generated text (prompt + completion).
    """
    input_ids = tokenizer.encode(prompt)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    eot = tokenizer.eot_token

    generated_tokens: list[int] = []
    t0 = time.perf_counter()

    # ── Prefill: process entire prompt, build cache ──────────────────
    out = model(ids, use_cache=True)
    logits = out["logits"][:, -1, :]  # [1, V]
    cache = out["cache"]

    for _ in range(max_tokens):
        # Repetition penalty
        if repetition_penalty != 1.0:
            seen = ids[0].unique()
            logits[:, seen] /= repetition_penalty

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Top-k
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        # Top-p (nucleus)
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Sample
        probs = torch.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1)  # [1, 1]
        tok_id = tok.item()

        # Stop on EOT
        if tok_id == eot:
            break

        ids = torch.cat([ids, tok], dim=1)
        generated_tokens.append(tok_id)

        # Stream decode — decode just the new token
        text = tokenizer.decode([tok_id])
        print(text, end="", flush=True)

        # ── Cached decode step ───────────────────────────────────────
        out = model(tok, cache=cache, use_cache=True)
        logits = out["logits"][:, -1, :]
        cache = out["cache"]

    dt = time.perf_counter() - t0
    tok_s = len(generated_tokens) / max(dt, 1e-6)
    print(f"\n\n[{len(generated_tokens)} tokens, {tok_s:.1f} tok/s]")

    return tokenizer.decode(input_ids + generated_tokens)


# ─── REPL ────────────────────────────────────────────────────────────────────

def repl(model, tokenizer, device, args):
    print("\n── HybridNSVSA-HA Chat ──")
    print(f"   temperature={args.temperature}  top_k={args.top_k}  "
          f"top_p={args.top_p}  rep_penalty={args.repetition_penalty}")
    print("   Type a prompt and press Enter. Ctrl-C or 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        generate_streaming(
            model, tokenizer, prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print()  # blank line between generations


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Chat / generate with HybridNSVSA-HA")

    # Model loading
    p.add_argument("--checkpoint",  default="checkpoints/best.pt",
                   help="Path to .pt checkpoint")
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Skip loading checkpoint (random weights)")

    # Model config (only used with --no_checkpoint)
    p.add_argument("--d_model",     type=int, default=512)
    p.add_argument("--num_layers",  type=int, default=12)
    p.add_argument("--num_heads",   type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--group_size",  type=int, default=64)
    p.add_argument("--vocab_size",  type=int, default=100277)
    p.add_argument("--tokenizer",    default="cl100k_base",
                   help="tiktoken encoding name (used when --tokenizer_json is not set)")
    p.add_argument("--tokenizer_json", default=DEFAULT_CUSTOM_TOKENIZER_JSON,
                   help="Path to custom HF tokenizers tokenizer.json")
    p.add_argument("--tokenizer_meta", default=DEFAULT_CUSTOM_TOKENIZER_META,
                   help="Path to tokenizer metadata JSON (optional)")
    p.add_argument("--strict_tokenizer_match", action="store_true",
                   help="Fail if checkpoint tokenizer metadata mismatches runtime tokenizer")

    # Sampling
    p.add_argument("--temperature",        type=float, default=0.8)
    p.add_argument("--top_k",              type=int,   default=50)
    p.add_argument("--top_p",              type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--max_tokens",         type=int,   default=256)

    # Single-shot mode
    p.add_argument("--prompt", default=None,
                   help="If provided, generate once and exit (no REPL)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.tokenizer_json and not os.path.exists(args.tokenizer_json):
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

    if args.no_checkpoint:
        args.vocab_size = tokenizer.n_vocab

    model, device = load_model(args, tok_info)

    if args.prompt:
        generate_streaming(
            model, tokenizer, args.prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        repl(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
