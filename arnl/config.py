"""
ARNL V1.1 Configuration
========================

All hyperparameters for the Arnold architecture (SHADE Edition).

V1.1 eliminates the Reasoning Head and W_proj Injection Bridge.
Gap detection is driven by System 1's output entropy over SAC tokens.
Factual injection operates directly at the logit layer via SHADE.

Parameter Budget Philosophy (~95 / ~5):
    System 1 (LLSU):  ~95% of total params — fluency backbone + gap detection
    Infrastructure:    ~5% — SALT embeddings
    SHADE:            Off-budget (database, no trainable parameters)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ARNLConfig:
    """Full configuration for the ARNL V1.1 architecture."""

    # ── Model Identity ──────────────────────────────────────────────
    model_name: str = "Arnold"
    arch_name: str = "ARNL"
    version: str = "1.1"

    # ── Vocabulary ──────────────────────────────────────────────────
    vocab_size: int = 32_000
    max_seq_len: int = 4_096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer_type: str = "salt"
    tokenizer_dir: str = "tokenizer/"

    # Entity-mask special tokens (EMLM training)
    entity_tokens: List[str] = field(default_factory=lambda: [
        "[ENTITY_NOUN]",
        "[ENTITY_NUM]",
        "[ENTITY_CONCEPT]",
        "[ENTITY_DOMAIN]",
    ])

    # ── System 1 — LLSU (Gated Linear Attention / SSM) ──────────
    d_model: int = 2048
    n_layers: int = 24
    d_ffn: int = 5504
    d_state: int = 256
    n_heads: int = 16

    # ── Gap Detection (replaces Reasoning Head) ─────────────────
    h_gap: float = 2.0         # Shannon entropy threshold (nats) over SAC tokens

    # ── System 2 — SHADE ────────────────────────────────────────
    # Off-budget (non-neural database)
    d_semantic: int = 768      # Frozen SALT semantic embedding dimension
    shade_top_m: int = 16      # Nearest-neighbor candidates per retrieval
    shade_top_k: int = 8       # Top-K tokens stored per node target_dist
    shade_context_k: int = 8   # Recent SAC tokens for context fingerprint
    shade_sim_min: float = 0.65  # Min similarity for retrieval (else abstain)
    s_max: int = 2000          # Overflow ceiling (S_overflow cap)

    # Consistency gate thresholds (τ_tiered)
    tau_axiom: float = 0.92
    tau_domain: float = 0.80
    tau_user: float = 0.65

    # Conflict detection
    delta: float = 0.25        # Inversion dead-zone threshold

    # ── Decay Engine ────────────────────────────────────────────
    beta: float = 1.0
    miss_decay_interval: int = 5
    overflow_hit_delta: int = 20
    overflow_miss_delta: int = 20

    # ── Function Word Suppression ───────────────────────────────
    function_word_suppression: float = 0.2

    # ── LoRA (Phase 2 adapters on System 1) ─────────────────────
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # ── Training ────────────────────────────────────────────────
    dropout: float = 0.1
    weight_tying: bool = True

    # Paraphrase pass
    paraphrase_cosine_radius: float = 0.15
    paraphrase_init_sbase: int = 25

    # ── Preset Configurations ───────────────────────────────────

    @staticmethod
    def arnold_tiny() -> "ARNLConfig":
        """Tiny config for unit tests and development (~12M params).

        System 1 commands ~95% of the parameter budget.
        SHADE is off-budget (database, not neural).
        """
        return ARNLConfig(
            vocab_size=32_000,
            max_seq_len=2048,
            d_model=256,
            n_layers=4,
            d_ffn=512,
            d_state=64,
            n_heads=4,
            d_semantic=128,
            shade_top_m=8,
            shade_top_k=8,
            shade_context_k=4,
            h_gap=2.0,
        )

    @staticmethod
    def arnold_1b() -> "ARNLConfig":
        """~1 B total parameters."""
        return ARNLConfig(
            d_model=1536, n_layers=16, d_ffn=4096,
            d_state=192, n_heads=12, d_semantic=768,
        )

    @staticmethod
    def arnold_3b() -> "ARNLConfig":
        """~3 B total parameters."""
        return ARNLConfig(
            d_model=2048, n_layers=24, d_ffn=5504,
            d_state=256, n_heads=16, d_semantic=768,
        )

    @staticmethod
    def arnold_7b() -> "ARNLConfig":
        """~7 B total parameters (reference scale from spec)."""
        return ARNLConfig(
            d_model=4096, n_layers=32, d_ffn=11008,
            d_state=512, n_heads=32, d_semantic=768,
        )

    @staticmethod
    def arnold_13b() -> "ARNLConfig":
        """~13 B total parameters."""
        return ARNLConfig(
            d_model=5120, n_layers=40, d_ffn=13824,
            d_state=640, n_heads=40, d_semantic=768,
        )

    @staticmethod
    def arnold_24b() -> "ARNLConfig":
        """~24 B total parameters."""
        return ARNLConfig(
            d_model=6144, n_layers=48, d_ffn=16384,
            d_state=768, n_heads=48, d_semantic=768,
        )
