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


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_TOKENIZER_JSON = "tokenizers/vsa48k_en_code/tokenizer.json"
DEFAULT_TOKENIZER_META = "tokenizers/vsa48k_en_code/tokenizer_meta.json"

STATIC_DIR = Path(__file__).parent / "static"


# ═══════════════════════════════════════════════════════════════════════
#  Global state (set in main)
# ═══════════════════════════════════════════════════════════════════════

model: HybridNSVSA | None = None
tokenizer: BaseTokenizer | None = None
device: torch.device | None = None
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
    token_queue: asyncio.Queue[str | None] = asyncio.Queue()
    stats: dict = {}

    def _generate():
        """Synchronous generation — runs in a thread."""
        nonlocal stats
        with torch.no_grad():
            input_ids = tokenizer.encode(req.message)
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            eot = tokenizer.eot_token

            generated_tokens: list[int] = []
            t0 = time.perf_counter()

            # Prefill
            out = model(ids, use_cache=True)
            logits = out["logits"][:, -1, :]
            cache = out["cache"]

            for _ in range(req.max_tokens):
                # Repetition penalty
                if req.repetition_penalty != 1.0:
                    seen = ids[0].unique()
                    logits[:, seen] /= req.repetition_penalty

                # Temperature
                logits = logits / max(req.temperature, 1e-8)

                # Top-k
                if req.top_k is not None:
                    k = min(req.top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))

                # Top-p (nucleus)
                if req.top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cum_probs > req.top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits[remove] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                # Sample
                probs = torch.softmax(logits, dim=-1)
                tok = torch.multinomial(probs, 1)
                tok_id = tok.item()

                if tok_id == eot:
                    break

                ids = torch.cat([ids, tok], dim=1)
                generated_tokens.append(tok_id)

                # Decode token text and push to queue
                text = tokenizer.decode([tok_id])
                token_queue.put_nowait(text)

                # Cached decode step
                out = model(tok, cache=cache, use_cache=True)
                logits = out["logits"][:, -1, :]
                cache = out["cache"]

            dt = time.perf_counter() - t0
            tok_s = len(generated_tokens) / max(dt, 1e-6)
            stats.update({
                "tokens": len(generated_tokens),
                "time_s": round(dt, 2),
                "tok_per_s": round(tok_s, 1),
            })
            token_queue.put_nowait(None)  # Signal done

    async with gen_lock:
        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, _generate)

        while True:
            try:
                text = await asyncio.wait_for(token_queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Generation timed out'})}\n\n"
                break

            if text is None:
                # Send completion stats
                yield f"data: {json.dumps({'done': True, **stats})}\n\n"
                break

            yield f"data: {json.dumps({'token': text})}\n\n"

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
        mdl.load_state_dict(ckpt["model"])
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
    global model, tokenizer, device

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

    model, device = load_model_from_args(args, tok_info)

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
