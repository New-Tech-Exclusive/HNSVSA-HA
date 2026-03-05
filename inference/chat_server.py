#!/usr/bin/env python3
"""
Web chat server for HybridNSVSA-HA (Arnold).

Serves a ChatGPT-style web UI and a streaming generation API.

Usage:
  python chat_server.py --checkpoint checkpoints/best.pt
  python chat_server.py --checkpoint checkpoints/best.pt --port 8080
  python chat_server.py --no_checkpoint --d_model 256 --num_layers 4 --num_heads 4

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig
from nsvsa_ha.tokenizer import BaseTokenizer, load_tokenizer, tokenizer_compatible
from nsvsa_ha.modes import MODE_FAST, MODE_REASON, MODE_DEEP

_MODE_NAME_MAP = {"fast": MODE_FAST, "reason": MODE_REASON, "deep": MODE_DEEP}


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_TOKENIZER_JSON = "tokenizers/vsa65k_mix/tokenizer.json"
DEFAULT_TOKENIZER_META = "tokenizers/vsa65k_mix/tokenizer_meta.json"

STATIC_DIR = Path(__file__).parent / "static"


# ═══════════════════════════════════════════════════════════════════════
#  Global state (set in main)
# ═══════════════════════════════════════════════════════════════════════

model: HybridNSVSA | None = None
tokenizer: BaseTokenizer | None = None
device: torch.device | None = None
loaded_checkpoint: str = ""
gen_lock = asyncio.Lock()  # Serialize generation (1 GPU)


# ═══════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(title="Arnold Chat", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=50, ge=1, le=1000)
    top_p: float | None = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=3.0)
    mode: str = Field(
        default="deep",
        description="Reasoning mode: 'fast', 'reason', or 'deep'",
    )


class ModelInfo(BaseModel):
    name: str
    parameters: int
    d_model: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    window_size: int
    group_size: int
    vocab_size: int
    device: str
    tokenizer: str
    checkpoint: str


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "chat.html"
    if not html_path.exists():
        return HTMLResponse("<h1>chat.html not found in static/</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@app.get("/api/info")
async def info():
    cfg = model.config
    tok_info = tokenizer.info()
    return ModelInfo(
        name="Arnold (HybridNSVSA-HA)",
        parameters=model.num_parameters(),
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        max_seq_len=cfg.max_seq_len,
        window_size=cfg.window_size,
        group_size=cfg.group_size,
        vocab_size=cfg.vocab_size,
        device=str(device),
        tokenizer=f"{tok_info['backend']}:{tok_info['name']}",
        checkpoint=loaded_checkpoint,
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream generated tokens as Server-Sent Events."""
    return StreamingResponse(
        generate_sse(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def generate_sse(req: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Run autoregressive generation in a thread (torch is sync),
    yielding SSE events for each token.
    """
    token_queue: asyncio.Queue[dict | None] = asyncio.Queue()
    stats: dict = {}

    def _generate():
        """Synchronous generation — runs in a thread."""
        nonlocal stats

        def emit(event: dict):
            token_queue.put_nowait(event)

        def _to_float(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                return float(value.detach().float().mean().item())
            return float(value)

        def _sample_logits(logits: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
            """Apply repetition penalty, temperature, top-k/p then sample 1 token."""
            lgt = logits.clone()
            if req.repetition_penalty != 1.0:
                seen = context_ids[0].unique()
                lgt[:, seen] /= req.repetition_penalty
            lgt = lgt / max(req.temperature, 1e-8)
            if req.top_k is not None:
                k = min(req.top_k, lgt.size(-1))
                kth = torch.topk(lgt, k).values[:, -1, None]
                lgt = lgt.masked_fill(lgt < kth, float("-inf"))
            if req.top_p is not None:
                sl, si = torch.sort(lgt, descending=True)
                cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
                rm = cum > req.top_p
                rm[:, 1:] = rm[:, :-1].clone()
                rm[:, 0] = False
                sl[rm] = float("-inf")
                lgt = sl.scatter(1, si, sl)
            return torch.multinomial(torch.softmax(lgt, dim=-1), 1)

        with torch.no_grad():
            input_ids = tokenizer.encode(req.message)

            # Prepend mode control token if available
            mode_name = getattr(req, "mode", "deep") or "deep"
            mode_id = _MODE_NAME_MAP.get(mode_name, MODE_DEEP)
            mode_tok_ids = model.config.mode_token_ids
            if mode_tok_ids and mode_id in mode_tok_ids:
                input_ids = [mode_tok_ids[mode_id]] + input_ids

            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            eot = tokenizer.eot_token

            # Think-token IDs from tokenizer (None if not in vocab)
            think_id     = getattr(tokenizer, "think_token_id", None)
            end_think_id = getattr(tokenizer, "end_think_token_id", None)

            # Thinking budget: deep=256 tokens, reason=128, fast=0
            if think_id is not None and mode_id != MODE_FAST:
                max_think = 256 if mode_id == MODE_DEEP else 128
            else:
                max_think = 0

            generated_tokens: list[int] = []
            t0 = time.perf_counter()
            decode_reason_step_sum = 0.0
            decode_reason_step_count = 0

            # ── Prefill ──────────────────────────────────────────────
            out = model(ids, use_cache=True)
            logits = out["logits"][:, -1, :]
            cache = out["cache"]

            # ── Thinking panel header ─────────────────────────────────
            mode_label = mode_name.upper()
            emit({"thinking": f"[{mode_label} mode · think budget: {max_think} tokens]\n\n"})

            # ── Thinking phase (reason / deep modes only) ────────────
            if max_think > 0 and think_id is not None:
                # Inject <|think|> into context and take one decode step
                think_tok = torch.tensor([[think_id]], dtype=torch.long, device=device)
                ids = torch.cat([ids, think_tok], dim=1)
                out = model(think_tok, cache=cache, use_cache=True)
                logits = out["logits"][:, -1, :]
                cache  = out["cache"]

                think_count = 0
                while think_count < max_think:
                    tok = _sample_logits(logits, ids)
                    tok_id = int(tok.item())
                    ids = torch.cat([ids, tok], dim=1)
                    if tok_id == eot:
                        break
                    text = tokenizer.decode([tok_id])
                    emit({"thinking": text})
                    out    = model(tok, cache=cache, use_cache=True)
                    logits = out["logits"][:, -1, :]
                    cache  = out["cache"]
                    if tok_id == end_think_id:
                        # Model closed its think block voluntarily
                        break
                    think_count += 1
                else:
                    # Budget exhausted — force-close the think block
                    end_tok = torch.tensor([[end_think_id]], dtype=torch.long, device=device)
                    ids = torch.cat([ids, end_tok], dim=1)
                    emit({"thinking": tokenizer.decode([end_think_id])})
                    out    = model(end_tok, cache=cache, use_cache=True)
                    logits = out["logits"][:, -1, :]
                    cache  = out["cache"]

                emit({"thinking": "\n\n"})  # visual separator before the answer

            # ── Answer generation ────────────────────────────────────
            for _ in range(req.max_tokens):
                tok = _sample_logits(logits, ids)
                tok_id = int(tok.item())

                if tok_id == eot:
                    break

                ids = torch.cat([ids, tok], dim=1)
                generated_tokens.append(tok_id)

                text = tokenizer.decode([tok_id])
                emit({"token": text})

                out    = model(tok, cache=cache, use_cache=True)
                logits = out["logits"][:, -1, :]
                cache  = out["cache"]

            dt    = time.perf_counter() - t0
            tok_s = len(generated_tokens) / max(dt, 1e-6)
            stats.update({
                "tokens":    len(generated_tokens),
                "time_s":    round(dt, 2),
                "tok_per_s": round(tok_s, 1),
            })
            token_queue.put_nowait(None)  # Signal done

    async with gen_lock:
        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, _generate)

        while True:
            try:
                event = await asyncio.wait_for(token_queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Generation timed out'})}\n\n"
                break

            if event is None:
                # Send completion stats
                yield f"data: {json.dumps({'done': True, **stats})}\n\n"
                break

            yield f"data: {json.dumps(event)}\n\n"

        await gen_task


# ═══════════════════════════════════════════════════════════════════════
#  Model loading (reused from chat.py)
# ═══════════════════════════════════════════════════════════════════════

def load_model_from_args(args, tok_info: dict) -> tuple[HybridNSVSA, torch.device]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint and not args.no_checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        config = HybridNSVSAConfig(**cfg_dict)
        mdl = HybridNSVSA(config)
        # Checkpoints saved from torch.compile'd models have keys prefixed
        # with "_orig_mod." — strip it so they load into a plain model.
        state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        # Expand vocabulary if tokenizer has grown since checkpoint was saved
        ckpt_vocab = state["embedding.weight"].shape[0]
        tok_vocab = tok_info.get("vocab_size", config.vocab_size)
        if ckpt_vocab < tok_vocab:
            def _pad_w(w: torch.Tensor, new_rows: int) -> torch.Tensor:
                padded = torch.zeros(new_rows, w.shape[1], dtype=w.dtype)
                padded[:w.shape[0]] = w
                return padded
            state = dict(state)
            state["embedding.weight"] = _pad_w(state["embedding.weight"], tok_vocab)
            if "lm_head.weight" in state:
                state["lm_head.weight"] = _pad_w(state["lm_head.weight"], tok_vocab)
            config.vocab_size = tok_vocab
            mdl = HybridNSVSA(config)  # rebuild with correct vocab
            print(f"  vocab expanded: {ckpt_vocab} → {tok_vocab}")
        mdl.load_state_dict(state)
        # Re-tie lm_head → embedding after load_state_dict (which breaks the tie)
        if config.tie_weights:
            mdl.lm_head.weight = mdl.embedding.weight
        ckpt_tok = ckpt.get("tokenizer")
        if ckpt_tok and not tokenizer_compatible(ckpt_tok, tok_info):
            print(
                f"WARNING: Checkpoint tokenizer mismatch. "
                f"ckpt={ckpt_tok} runtime={tok_info}"
            )
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
        mdl = HybridNSVSA(config)
        print("No checkpoint — using random weights.")

    mdl = mdl.to(dev).eval()
    n = mdl.num_parameters()
    print(f"Model: {n:,} params  |  device: {dev}")
    return mdl, dev


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Arnold web chat server")

    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument(
        "--checkpoint_kind",
        choices=["best", "final"],
        default=None,
        help=(
            "Shortcut for SFT checkpoints in checkpoints/sft: "
            "best -> checkpoints/sft/best_sft.pt, "
            "final -> checkpoints/sft/sft_final.pt"
        ),
    )
    p.add_argument("--no_checkpoint", action="store_true")

    # Model config (only with --no_checkpoint)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--window_size", type=int, default=256)
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=48000)

    # Tokenizer
    p.add_argument("--tokenizer", default="cl100k_base")
    p.add_argument("--tokenizer_json", default=DEFAULT_TOKENIZER_JSON)
    p.add_argument("--tokenizer_meta", default=DEFAULT_TOKENIZER_META)

    # Server
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)

    return p.parse_args()


def main():
    global model, tokenizer, device, loaded_checkpoint

    args = parse_args()

    # Tokenizer
    tok_json = args.tokenizer_json
    tok_meta = args.tokenizer_meta
    if tok_json and not Path(tok_json).exists():
        print(f"Custom tokenizer not found at {tok_json}; falling back to tiktoken")
        tok_json = None
        tok_meta = None

    tokenizer = load_tokenizer(
        tokenizer_name=args.tokenizer,
        tokenizer_json=tok_json,
        tokenizer_meta=tok_meta,
    )
    tok_info = tokenizer.info()
    print(
        f"Tokenizer: {tok_info['backend']}:{tok_info['name']} "
        f"| vocab={tok_info['vocab_size']} | eot={tok_info['eot_token_id']}"
    )

    if args.no_checkpoint:
        args.vocab_size = tokenizer.n_vocab
    elif args.checkpoint_kind is not None:
        ckpt_map = {
            "best": "checkpoints/sft/best_sft.pt",
            "final": "checkpoints/sft/sft_final.pt",
        }
        args.checkpoint = ckpt_map[args.checkpoint_kind]

    loaded_checkpoint = args.checkpoint if not args.no_checkpoint else "<random-init>"

    model, device = load_model_from_args(args, tok_info)

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
