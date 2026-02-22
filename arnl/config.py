"""
ARNL Configuration
==================

All hyperparameters for the Arnold architecture. Includes preset
configurations for different parameter scales (1B through 24B+).

Parameter Budget Philosophy (80/15/5):
    System 1 (LLSU):        ~80% — fluency backbone
    Reasoning Head:         ~15% — control logic + classifier
    Injection Bridge W_proj: ~5% — semantic-to-syntactic projection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ARNLConfig:
    """Full configuration for the ARNL architecture."""

    # ── Model Identity ──────────────────────────────────────────────
    model_name: str = "Arnold"
    arch_name: str = "ARNL"
    version: str = "1.0"

    # ── Vocabulary ──────────────────────────────────────────────────
    vocab_size: int = 32_000
    max_seq_len: int = 4_096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Entity-mask special tokens (EMLM training)
    entity_tokens: List[str] = field(default_factory=lambda: [
        "[ENTITY_NOUN]",     # Proper nouns (people, places)
        "[ENTITY_NUM]",      # Numerical data (dates, quantities)
        "[ENTITY_CONCEPT]",  # Named concepts (products, events)
        "[ENTITY_DOMAIN]",   # Domain-specific terms
    ])

    # ── System 1 — LLSU (Gated Linear Attention / SSM) ──────────
    # ~80 % of total parameter budget
    d_model: int = 2048        # Hidden dimension
    n_layers: int = 24         # Number of LLSU blocks
    d_ffn: int = 5504          # SwiGLU intermediate dimension
    d_state: int = 256         # GLA/SSM recurrent state width
    n_heads: int = 16          # Number of GLA heads

    # ── System 2 — Hyper-Adjacency Map ──────────────────────────
    # Off-budget (non-neural hash map)
    d_semantic: int = 768      # Frozen semantic embedding dimension
    k_anchors: int = 5         # Semantic anchors per lookup
    s_max: int = 2000          # Overflow ceiling (S_overflow cap)

    # ── Reasoning Head ──────────────────────────────────────────
    # ~15 % of total parameter budget
    d_reasoning: int = 1024    # Context-encoder hidden dim
    n_reasoning_layers: int = 8
    d_reasoning_ffn: int = 2816
    n_reasoning_heads: int = 8

    # Consistency gate thresholds (τ_tiered)
    tau_axiom: float = 0.92    # Physical constants, definitions
    tau_domain: float = 0.80   # Scientific / technical claims
    tau_user: float = 0.65     # User preferences, personal context

    # Conflict detection
    delta: float = 0.25        # Inversion dead-zone threshold

    # ── Decay Engine ────────────────────────────────────────────
    beta: float = 1.0          # Logarithmic overflow scaling coeff
    miss_decay_interval: int = 5   # 1-in-5 rule
    overflow_hit_delta: int = 20   # +20 on hit
    overflow_miss_delta: int = 20  # -20 on miss

    # ── Injection Bridge (W_proj) ───────────────────────────────
    # ~5 % of total parameter budget
    # Dimensions: d_model × d_semantic (derived)
    function_word_suppression: float = 0.2   # α multiplier on function words

    # Which LLSU layers receive injection (None ⇒ last layer only)
    injection_layers: Optional[List[int]] = None

    # ── LoRA (Phase 2/3 adapters on System 1) ───────────────────
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # ── Training ────────────────────────────────────────────────
    dropout: float = 0.1
    weight_tying: bool = True  # Tie input/output embeddings

    # Saliency filter
    syntactic_filter_cosine: float = 0.3     # Proximity to syntactic centroids
    semantic_vacuum_min_anchors: int = 2     # Min anchors before skipping S2

    # Paraphrase pass
    paraphrase_cosine_radius: float = 0.15
    paraphrase_init_sbase: int = 25

    # ── Preset Configurations ───────────────────────────────────

    @staticmethod
    def arnold_tiny() -> "ARNLConfig":
        """Tiny config for unit tests and development (~5M params)."""
        return ARNLConfig(
            vocab_size=1_000,
            max_seq_len=512,
            d_model=256,
            n_layers=4,
            d_ffn=512,
            d_state=64,
            n_heads=4,
            d_semantic=128,
            k_anchors=3,
            d_reasoning=128,
            n_reasoning_layers=2,
            d_reasoning_ffn=256,
            n_reasoning_heads=4,
        )

    @staticmethod
    def arnold_1b() -> "ARNLConfig":
        """~1 B total parameters."""
        return ARNLConfig(
            d_model=1536,
            n_layers=16,
            d_ffn=4096,
            d_state=192,
            n_heads=12,
            d_reasoning=768,
            n_reasoning_layers=4,
            d_reasoning_ffn=2048,
            n_reasoning_heads=8,
        )

    @staticmethod
    def arnold_3b() -> "ARNLConfig":
        """~3 B total parameters."""
        return ARNLConfig(
            d_model=2048,
            n_layers=24,
            d_ffn=5504,
            d_state=256,
            n_heads=16,
            d_reasoning=1024,
            n_reasoning_layers=6,
            d_reasoning_ffn=2816,
            n_reasoning_heads=8,
        )

    @staticmethod
    def arnold_7b() -> "ARNLConfig":
        """~7 B total parameters (reference scale from spec)."""
        return ARNLConfig(
            d_model=4096,
            n_layers=32,
            d_ffn=11008,
            d_state=512,
            n_heads=32,
            d_reasoning=1536,
            n_reasoning_layers=12,
            d_reasoning_ffn=4096,
            n_reasoning_heads=16,
        )

    @staticmethod
    def arnold_13b() -> "ARNLConfig":
        """~13 B total parameters."""
        return ARNLConfig(
            d_model=5120,
            n_layers=40,
            d_ffn=13824,
            d_state=640,
            n_heads=40,
            d_reasoning=2048,
            n_reasoning_layers=16,
            d_reasoning_ffn=5504,
            n_reasoning_heads=16,
        )

    @staticmethod
    def arnold_24b() -> "ARNLConfig":
        """~24 B total parameters."""
        return ARNLConfig(
            d_model=6144,
            n_layers=48,
            d_ffn=16384,
            d_state=768,
            n_heads=48,
            d_reasoning=2560,
            n_reasoning_layers=20,
            d_reasoning_ffn=6912,
            n_reasoning_heads=20,
        )
