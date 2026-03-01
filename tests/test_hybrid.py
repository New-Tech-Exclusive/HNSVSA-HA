#!/usr/bin/env python3
"""
Smoke test for the Hybrid NSVSA-HA model.
Tests each new component individually, then the full model end-to-end.
"""

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

PASS = "[OK]"
FAIL = "[FAIL]"


# ── 1. RoPE ─────────────────────────────────────────────────────────────────

def test_rope():
    from nsvsa_ha.rope import RotaryEmbedding, build_rope_cache, apply_rope
    rope = RotaryEmbedding(head_dim=64, max_seq_len=512)

    # cos/sin cache shape
    cos, sin = build_rope_cache(512, 64)
    assert cos.shape == (512, 64), f"Expected (512,64) got {cos.shape}"

    # apply to Q/K
    B, H, L, Dh = 2, 4, 32, 64
    q = torch.randn(B, H, L, Dh)
    k = torch.randn(B, H, L, Dh)
    q_r, k_r = rope(q, k)
    assert q_r.shape == q.shape, "RoPE changed shape"

    # position vectors are unit-norm
    pos = torch.arange(16)
    vecs = rope.get_position_vectors(pos, 512)
    norms = vecs.norm(dim=-1)
    assert (norms - 1.0).abs().max() < 1e-4, f"Position vecs not unit norm: {norms[:4]}"

    # relative-position property: similarity decays with distance
    # sim(0,1) should be greater than sim(0,16)
    pos2 = torch.arange(32)
    v = rope.get_position_vectors(pos2, 512)
    sim_near = (v[0] * v[1]).sum().item()         # adjacent
    sim_far  = (v[0] * v[16]).sum().item()        # half-sequence away
    sim_very_far = (v[0] * v[31]).sum().item()    # max distance
    assert sim_near > sim_far, \
        f"Similarity should decay: sim(0,1)={sim_near:.3f} sim(0,16)={sim_far:.3f}"
    assert sim_far > sim_very_far, \
        f"Similarity should decay: sim(0,16)={sim_far:.3f} sim(0,31)={sim_very_far:.3f}"

    print(f"  {PASS} RoPE: shapes, unit-norm, relative decorrelation")


# ── 2. Soft bundling ─────────────────────────────────────────────────────────

def test_soft_bundle():
    from nsvsa_ha.soft_vsa import SoftBundle
    bundle = SoftBundle()

    # Unit-norm output
    vecs = torch.randn(8, 32, 1000)       # [B, n, d]
    out = bundle(vecs, dim=1)
    norms = out.norm(dim=-1)
    assert (norms - 1.0).abs().max() < 1e-4, "SoftBundle output not unit-norm"

    # Similarity to members > similarity to random
    unit_vecs = F.normalize(vecs, dim=-1)
    member_sim = (out * unit_vecs[:, 0]).sum(-1).mean().item()
    random_sims = (out * F.normalize(torch.randn_like(out), dim=-1)).sum(-1).mean().item()
    assert member_sim > random_sims, "Bundled vector not closer to members than random"

    # Real gradients
    vecs2 = torch.randn(4, 8, 512, requires_grad=True)
    out2 = bundle(vecs2, dim=1)
    out2.sum().backward()
    assert vecs2.grad is not None, "No gradient through SoftBundle"
    assert (vecs2.grad != 0).any(), "Zero gradient through SoftBundle"

    print(f"  {PASS} SoftBundle: unit-norm, similarity, real gradients")


# ── 3. SoftVSAStateUpdate ────────────────────────────────────────────────────

def test_soft_vsa_state():
    from nsvsa_ha.soft_vsa import SoftVSAStateUpdate
    from nsvsa_ha.rope import RotaryEmbedding

    d, K, G = 256, 8, 32
    rope = RotaryEmbedding(head_dim=32, max_seq_len=512)
    local_pos = rope.get_position_vectors(torch.arange(K), d)
    macro_pos = rope.get_position_vectors(torch.arange(G) + 10_000, d)

    vsa = SoftVSAStateUpdate(d=d, group_size=K, max_groups=G)

    B, L = 2, 24
    tokens = F.normalize(torch.randn(B, L, d), dim=-1)

    queries, state, _ = vsa(tokens, local_pos, macro_pos)

    assert queries.shape == (B, L, d),  f"Query shape wrong: {queries.shape}"
    assert state.shape   == (B, d),     f"State shape wrong: {state.shape}"
    # State is unit-norm
    assert (state.norm(dim=-1) - 1.0).abs().max() < 1e-3, "State not unit-norm"

    # Gradients flow to decay parameter
    loss = queries.sum() + state.sum()
    loss.backward()
    assert vsa.decay_logit.grad is not None, "No grad on decay_logit"

    print(f"  {PASS} SoftVSAStateUpdate: shapes, unit-norm state, gradients")


# ── 4. Local windowed attention ───────────────────────────────────────────────

def test_local_attention():
    from nsvsa_ha.local_attention import LocalWindowedAttention

    d, H, W = 256, 4, 16
    attn = LocalWindowedAttention(d_model=d, num_heads=H, window_size=W)

    B, L = 2, 32
    x = torch.randn(B, L, d)
    out, _ = attn(x)
    assert out.shape == (B, L, d), f"Attn output shape wrong: {out.shape}"

    # Causal: output at t should not depend on t+1
    # Perturb last token, check first token output unchanged
    x2 = x.clone()
    x2[:, -1] += 10.0
    out2, _ = attn(x2)
    diff_first = (out[:, 0] - out2[:, 0]).abs().max().item()
    assert diff_first < 1e-4, f"Attention is not causal: diff={diff_first:.4f}"

    # Gradients
    x3 = torch.randn(B, L, d, requires_grad=True)
    attn(x3)[0].sum().backward()
    assert x3.grad is not None

    print(f"  {PASS} LocalWindowedAttention: shape, causality, gradients")


# ── 5. SwiGLU FFN ─────────────────────────────────────────────────────────────

def test_ffn():
    from nsvsa_ha.ffn import build_ffn

    for variant in ("swiglu", "geglu"):
        ffn = build_ffn(d_model=256, expansion_ratio=4.0, variant=variant)
        x = torch.randn(2, 16, 256, requires_grad=True)
        out = ffn(x)
        assert out.shape == x.shape, f"{variant} output shape wrong"
        out.sum().backward()
        assert x.grad is not None

    print(f"  {PASS} SwiGLU / GEGLU FFN: shapes, gradients")


# ── 6. HybridNSVSALayer ───────────────────────────────────────────────────────

def test_hybrid_layer():
    from nsvsa_ha.hybrid_layer import HybridNSVSALayer
    from nsvsa_ha.rope import RotaryEmbedding

    d, H, W, K, G = 256, 4, 16, 8, 32
    rope = RotaryEmbedding(head_dim=d // H, max_seq_len=512)
    local_pos = rope.get_position_vectors(torch.arange(K), d)
    macro_pos = rope.get_position_vectors(torch.arange(G) + 10_000, d)

    layer = HybridNSVSALayer(
        d_model=d, num_heads=H, window_size=W,
        group_size=K, max_groups=G
    )

    B, L = 2, 24
    x = torch.randn(B, L, d, requires_grad=True)
    out, state, _ = layer(x, local_pos, macro_pos)

    assert out.shape   == (B, L, d)
    assert state.shape == (B, d)

    out.sum().backward()
    assert x.grad is not None

    print(f"  {PASS} HybridNSVSALayer: shapes, gradients")


# ── 7. Full HybridNSVSA model ─────────────────────────────────────────────────

def test_hybrid_model():
    from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig

    cfg = HybridNSVSAConfig(
        d_model=256,
        vocab_size=200,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        window_size=16,
        group_size=8,
        ffn_ratio=4.0,
    )

    model = HybridNSVSA(cfg).to(DEVICE)

    B, L = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, L), device=DEVICE)

    # Forward
    out = model(ids)
    assert out["logits"].shape == (B, L, cfg.vocab_size), \
        f"Logits shape wrong: {out['logits'].shape}"

    # Loss
    out_loss = model(ids, labels=ids)
    assert "loss" in out_loss
    assert out_loss["loss"].item() > 0

    # Backward with real gradients (no STE)
    out_loss["loss"].backward()
    total_params = model.num_parameters()
    params_with_grad = sum(
        p.numel() for p in model.parameters() if p.grad is not None
    )
    grad_coverage = params_with_grad / total_params
    assert grad_coverage > 0.95, \
        f"Only {grad_coverage:.1%} of params have gradients"

    print(f"  {PASS} HybridNSVSA: forward, loss, {grad_coverage:.1%} gradient coverage")
    print(f"         Parameters: {total_params:,}")


# ── 8. Generation ─────────────────────────────────────────────────────────────

def test_generation():
    from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig

    cfg = HybridNSVSAConfig(
        d_model=256, vocab_size=100, num_layers=2, num_heads=4,
        max_seq_len=32, window_size=8, group_size=4,
    )
    model = HybridNSVSA(cfg).to(DEVICE)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)
    generated = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)
    assert generated.shape == (1, 12), f"Generated shape wrong: {generated.shape}"

    print(f"  {PASS} Generation: {prompt[0].tolist()} → {generated[0].tolist()}")


# ── 9. Parameter groups (differential LR) ─────────────────────────────────────

def test_param_groups():
    from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig

    cfg = HybridNSVSAConfig(d_model=128, vocab_size=100, num_layers=2,
                            num_heads=4, max_seq_len=32, window_size=8, group_size=4)
    model = HybridNSVSA(cfg)
    groups = model.parameter_groups(base_lr=1e-4, vsa_lr_scale=0.1)
    assert len(groups) == 2
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == model.num_parameters()
    print(f"  {PASS} Parameter groups: {len(groups)} groups, VSA LR = 1e-5")


# ── 10. Soft VSA has no STE (verify real gradients vs original) ───────────────

def test_no_ste_needed():
    """Confirm SoftVSA gradient magnitude is non-trivially large compared to STE."""
    from nsvsa_ha.soft_vsa import SoftVSAStateUpdate
    from nsvsa_ha.rope import RotaryEmbedding

    d, K, G = 256, 8, 16
    rope = RotaryEmbedding(head_dim=32, max_seq_len=512)
    local_pos = rope.get_position_vectors(torch.arange(K), d)
    macro_pos = rope.get_position_vectors(torch.arange(G) + 10_000, d)

    vsa = SoftVSAStateUpdate(d=d, group_size=K, max_groups=G)

    tokens = torch.randn(2, 16, d, requires_grad=True)
    queries, state, _ = vsa(F.normalize(tokens, dim=-1), local_pos, macro_pos)

    # Compute gradient of loss w.r.t. input tokens
    (queries.sum() + state.sum()).backward()

    grad_norm = tokens.grad.norm().item()
    # If STE was being silently zeroed, this would be 0 or near-0
    assert grad_norm > 0.1, f"Gradient through soft VSA is suspiciously small: {grad_norm:.4f}"

    print(f"  {PASS} No STE needed: input gradient norm = {grad_norm:.4f} (real gradients)")


# ── 11. Cached vs uncached generation consistency ────────────────────────────

def test_cached_generation():
    """Verify that cached generation produces identical logits to uncached."""
    from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig

    cfg = HybridNSVSAConfig(
        d_model=128, vocab_size=100, num_layers=2, num_heads=4,
        max_seq_len=32, window_size=8, group_size=4,
    )
    model = HybridNSVSA(cfg).to(DEVICE).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE)
    n_decode = 4

    # Uncached: full forward on growing sequence
    ids = prompt.clone()
    uncached_logits = []
    for _ in range(n_decode):
        out = model(ids)
        logit = out["logits"][:, -1, :]  # [1, V]
        uncached_logits.append(logit)
        tok = logit.argmax(-1, keepdim=True)
        ids = torch.cat([ids, tok], dim=1)

    # Cached: prefill then single-token steps
    ids_c = prompt.clone()
    cached_logits = []
    out = model(ids_c, use_cache=True)
    cached_logits.append(out["logits"][:, -1, :])
    cache = out["cache"]
    tok = cached_logits[0].argmax(-1, keepdim=True)
    ids_c = torch.cat([ids_c, tok], dim=1)

    for _ in range(n_decode - 1):
        out = model(tok, cache=cache, use_cache=True)
        cached_logits.append(out["logits"][:, -1, :])
        cache = out["cache"]
        tok = cached_logits[-1].argmax(-1, keepdim=True)
        ids_c = torch.cat([ids_c, tok], dim=1)

    # Compare logits
    for i, (uc, cc) in enumerate(zip(uncached_logits, cached_logits)):
        diff = (uc - cc).abs().max().item()
        assert diff < 1e-3, f"Step {i}: cached vs uncached logit diff = {diff:.6f}"

    # Same generated sequence
    assert torch.equal(ids, ids_c), (
        f"Sequences differ: uncached={ids[0].tolist()} cached={ids_c[0].tolist()}"
    )

    print(f"  {PASS} Cached generation: logits match uncached (max diff < 1e-3)")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("RoPE",                  test_rope),
        ("SoftBundle",            test_soft_bundle),
        ("SoftVSAStateUpdate",    test_soft_vsa_state),
        ("LocalWindowedAttention",test_local_attention),
        ("FFN",                   test_ffn),
        ("HybridNSVSALayer",      test_hybrid_layer),
        ("HybridNSVSA full model",test_hybrid_model),
        ("Generation",            test_generation),
        ("Parameter groups",      test_param_groups),
        ("No-STE real gradients", test_no_ste_needed),
        ("Cached generation",     test_cached_generation),
    ]

    failed = []
    for name, fn in tests:
        print(f"Testing {name}...")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  {FAIL} {name}: {e}")
            traceback.print_exc()
            failed.append(name)

    print()
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
