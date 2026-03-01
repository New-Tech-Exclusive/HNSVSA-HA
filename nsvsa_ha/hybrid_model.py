"""
Hybrid NSVSA-HA Model – complete architecture

Full per-token forward pass:

    Token IDs  →  Embedding + LayerNorm  →  [HybridNSVSALayer × N]  →  LM Head

vs. the original model, changes are:
  1. Continuous embeddings – no STE binarization of the embedding layer
     (the SoftVSA L2-normalizes internally so vectors approach the hypersphere)
  2. RoPE position vectors instead of frozen random bipolar P
  3. SoftBundle instead of hard sgn – real gradients everywhere
  4. Local windowed attention alongside VSA state update
  5. SwiGLU FFN after each VSA operation
  6. Pre-norm residuals with LayerNorm throughout
  7. No Clean-Up Memory cosine-similarity pass – a simple weight-tied LM head
     is sufficient because the FFN already does the nonlinear mapping

Training stability:
  - VSA gates are initialized to 0, so the model starts as a pure local
    attention model and gradually learns to use the VSA state.
  - Separate learning-rate groups are exposed so the VSA parameters can
    be warmed up at a lower LR before joint training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .hybrid_layer import HybridNSVSALayerStack
from .rope import RotaryEmbedding
from .cache import ModelCache
from .reasoning import ReasoningBlock, ponder_loss_fn
from .rmsnorm import RMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridNSVSAConfig:
    """Configuration for the Hybrid NSVSA-HA model."""

    # Dimensions
    d_model: int = 512          # Model/embedding dimension
    vocab_size: int = 50_257    # Vocabulary size
    num_layers: int = 6         # Number of hybrid layers
    num_heads: int = 8          # Attention heads per layer
    max_seq_len: int = 2048     # Maximum sequence length

    # Local attention
    window_size: int = 128      # W – local attention window
    num_kv_heads: int = 0       # GQA KV heads (0 = MHA, same as num_heads)
    qk_norm: bool = True        # L2-normalize Q/K with learned temperature

    # VSA
    group_size: int = 64        # K – tokens per VSA group
    learned_vsa_positions: bool = True  # Learnable VSA position codebook

    # FFN
    ffn_ratio: float = 4.0      # FFN expansion ratio
    ffn_variant: str = "swiglu" # "swiglu" or "geglu"

    # Regularization
    dropout: float = 0.0
    embed_dropout: float = 0.0

    # VSA gate
    gate_init_bias: float = 0.0  # sigmoid(bias) = initial VSA contribution

    # Reasoning (PonderNet-style adaptive computation)
    reasoning_layers: int = 0       # Top R layers form the reasoning block (0 = off)
    max_reason_steps: int = 8       # Maximum pondering iterations
    reason_hidden_dim: int = 256    # ReasoningController MLP hidden size
    ponder_lambda: float = 0.01     # Weight for ponder KL loss
    ponder_p_geo: float = 0.5       # Geometric prior parameter
    reason_epsilon: float = 0.01    # Halt threshold for inference early-stop

    # Misc
    layer_norm_eps: float = 1e-6
    tie_weights: bool = True    # Tie input embedding ↔ LM head weights
    rope_base: float = 10_000.0 # RoPE frequency base

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.d_model % 2 == 0, "d_model must be even for RoPE"
        assert 0 <= self.reasoning_layers <= self.num_layers, \
            f"reasoning_layers ({self.reasoning_layers}) must be in [0, num_layers ({self.num_layers})]"
        # Resolve num_kv_heads: 0 means same as num_heads (standard MHA)
        if self.num_kv_heads == 0:
            object.__setattr__(self, 'num_kv_heads', self.num_heads)
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

    @property
    def reasoning_enabled(self) -> bool:
        return self.reasoning_layers > 0

    @property
    def base_layers(self) -> int:
        """Number of non-reasoning (base) layers."""
        return self.num_layers - self.reasoning_layers

    @property
    def num_macro_positions(self) -> int:
        return (self.max_seq_len + self.group_size - 1) // self.group_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridNSVSA(nn.Module):
    """
    Hybrid Neuro-Symbolic VSA Language Model.

    Differentiators from vanilla transformer:
    - VSA recurrent state provides O(1)-per-token long-range memory
    - Attention is local-only (window W), making it sub-quadratic
    - VSA algebraic structure gives interpretable memory operations
    - No KV-cache needed for long-range context (state replaces it)
    """

    def __init__(self, config: HybridNSVSAConfig):
        super().__init__()
        self.config = config

        # ── Token embedding ──────────────────────────────────────────
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.embed_drop = nn.Dropout(config.embed_dropout)

        # ── RoPE for position vectors (used by both attention and VSA) ─
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        # ── Hybrid layer stack ───────────────────────────────────────
        self.layers = HybridNSVSALayerStack(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            window_size=config.window_size,
            group_size=config.group_size,
            max_groups=config.num_macro_positions,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            ffn_variant=config.ffn_variant,
            layer_norm_eps=config.layer_norm_eps,
            gate_init_bias=config.gate_init_bias,
            num_kv_heads=config.num_kv_heads,
            qk_norm=config.qk_norm,
        )

        # ── LM head ──────────────────────────────────────────────────
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ── Reasoning block (PonderNet) ─────────────────────────────
        self.reasoning_block: Optional[ReasoningBlock] = None
        if config.reasoning_enabled:
            self.reasoning_block = ReasoningBlock(
                d_model=config.d_model,
                max_steps=config.max_reason_steps,
                hidden_dim=config.reason_hidden_dim,
                p_geometric=config.ponder_p_geo,
                epsilon=config.reason_epsilon,
            )

        # Weight tying (halves parameters, improves generalization)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        # ── Learned VSA position codebook ───────────────────────────
        if config.learned_vsa_positions:
            # Initialize from RoPE-derived vectors (warm start),
            # then let them train end-to-end.
            with torch.no_grad():
                K = config.group_size
                idx = torch.arange(K)
                local_init = self.rope.get_position_vectors(idx, config.d_model)

                max_macro = config.num_macro_positions
                offset = K * 1000
                macro_idx = torch.arange(max_macro) + offset
                macro_init = self.rope.get_position_vectors(
                    macro_idx % (config.max_seq_len + offset), config.d_model
                )
            self.local_pos_embed = nn.Parameter(local_init)      # [K, d]
            self.macro_pos_embed = nn.Parameter(macro_init)      # [max_groups, d]

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Scaled normal initialisation (GPT-2 style)."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # Don't re-init the learned position embeddings (already warm-started)
        if self.config.learned_vsa_positions:
            pass  # local_pos_embed and macro_pos_embed keep their RoPE init

    # ------------------------------------------------------------------
    # Position vector helpers
    # ------------------------------------------------------------------

    def _get_local_positions(self, device: torch.device) -> torch.Tensor:
        """
        Position vectors for intra-group binding.  Shape [K, d_model].

        If learned_vsa_positions: returns L2-normalized learned codebook.
        Otherwise: RoPE-derived unit vectors (original behavior).
        """
        if self.config.learned_vsa_positions:
            return F.normalize(self.local_pos_embed.to(device), p=2, dim=-1)
        K = self.config.group_size
        idx = torch.arange(K, device=device)
        return self.rope.get_position_vectors(idx, self.config.d_model)

    def _get_macro_positions(self, num_groups: int, device: torch.device) -> torch.Tensor:
        """
        Position vectors for group-level binding.  Shape [num_groups, d_model].

        If learned_vsa_positions: slices from learned macro codebook.
        Otherwise: RoPE-derived with offset (original behavior).
        """
        if self.config.learned_vsa_positions:
            return F.normalize(self.macro_pos_embed[:num_groups].to(device), p=2, dim=-1)
        offset = self.config.group_size * 1000
        idx = torch.arange(num_groups, device=device) + offset
        return self.rope.get_position_vectors(idx % (self.config.max_seq_len + offset),
                                              self.config.d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                      # [B, L]
        labels: Optional[torch.Tensor] = None,        # [B, L]
        cache: Optional[ModelCache] = None,            # generation cache
        use_cache: bool = False,                       # build / update cache?
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids:           Token IDs [B, L].
            labels:              Target IDs for LM loss [B, L].
                                 Cross-entropy is computed internally
                                 (shift by 1 applied here).
            cache:               ModelCache from a previous forward call.
                                 When provided, input_ids are the NEW tokens
                                 only (positions are offset automatically).
            use_cache:           If True, return a "cache" key in the output
                                 dict for subsequent cached forward calls.
            return_hidden_states: If True, include intermediate tensors.

        Returns dict with keys:
            logits:         [B, L, vocab_size]
            loss:           scalar (if labels provided, includes ponder_cost)
            ponder_cost:    scalar (if reasoning enabled)
            mean_steps:     scalar mean reasoning steps (if reasoning enabled)
            cache:          ModelCache (if use_cache)
            vsa_states:     list of per-layer final states (if return_hidden_states)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── Embed ────────────────────────────────────────────────────
        x = self.embed_drop(self.embed_norm(self.embedding(input_ids)))  # [B, L, d]

        # ── Position indices for RoPE inside local attention ────────
        seq_offset = cache.seq_len if cache is not None else 0
        seq_positions = torch.arange(
            seq_offset, seq_offset + L, device=device
        )  # [L]

        # ── Position vectors for VSA binding ────────────────────────
        local_pos = self._get_local_positions(device)                     # [K, d]
        total_len = seq_offset + L
        num_groups = (total_len + self.config.group_size - 1) // self.config.group_size
        macro_pos = self._get_macro_positions(num_groups, device)         # [G, d]

        # ── Common kwargs for layer calls ────────────────────────────
        layer_kwargs = dict(
            local_positions=local_pos,
            macro_positions=macro_pos,
            token_seq_positions=seq_positions,
        )

        # ── Decide path: reasoning or standard ──────────────────────
        cfg = self.config
        if cfg.reasoning_enabled and self.reasoning_block is not None:
            # Split: base layers [0, B) + reasoning layers [B, L)
            base_end = cfg.base_layers
            reason_start = base_end
            reason_end = cfg.num_layers

            # Base layer caches
            base_caches = None
            if cache is not None and cache.layers is not None:
                base_caches = cache.layers[:base_end]

            # Run base layers
            x, base_vsa, new_base_caches = self.layers.forward_partial(
                x, 0, base_end, **layer_kwargs,
                layer_caches=base_caches, use_cache=use_cache,
            )

            # Build reasoning_fn — a closure that runs reasoning layers
            # Note: during reasoning loop, we don't use/build KV cache
            # (the loop re-processes the same positions with updated states)
            def reasoning_fn(h, **_kw):
                return self.layers.forward_partial(
                    h, reason_start, reason_end, **layer_kwargs,
                    layer_caches=None, use_cache=False,
                )

            # Run reasoning block (PonderNet loop)
            hidden, ponder_cost, mean_steps = self.reasoning_block(
                x, reasoning_fn,
            )

            # Apply final norm
            hidden = self.layers.final_norm(hidden)

            vsa_states = base_vsa  # Reasoning VSA states vary per step
            new_layer_caches = new_base_caches

            # For cached generation, also run reasoning layers once (no loop)
            # to build proper caches for the decode path
            if use_cache:
                reason_caches_in = None
                if cache is not None and cache.layers is not None:
                    reason_caches_in = cache.layers[base_end:]
                _, reason_vsa, new_reason_caches = self.layers.forward_partial(
                    x, reason_start, reason_end, **layer_kwargs,
                    layer_caches=reason_caches_in, use_cache=True,
                )
                vsa_states = base_vsa + reason_vsa
                if new_layer_caches is not None and new_reason_caches is not None:
                    new_layer_caches = new_layer_caches + new_reason_caches

        else:
            # Standard path: all layers in one pass
            layer_caches = cache.layers if cache is not None else None
            hidden, vsa_states, new_layer_caches = self.layers(
                x, local_pos, macro_pos, seq_positions,
                layer_caches=layer_caches, use_cache=use_cache,
            )
            ponder_cost = None
            mean_steps = None

        # ── LM head ─────────────────────────────────────────────────
        logits = self.lm_head(hidden)  # [B, L, vocab_size]

        output: Dict[str, Any] = {"logits": logits}

        # ── Loss ─────────────────────────────────────────────────────
        if labels is not None:
            # Shift: predict token t from context 0…t-1
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add ponder regularization when reasoning is active
            if ponder_cost is not None:
                output["ponder_cost"] = ponder_cost
                output["mean_reason_steps"] = mean_steps
                loss = loss + self.config.ponder_lambda * ponder_cost
            output["loss"] = loss

        # ── Cache ────────────────────────────────────────────────────
        if use_cache:
            output["cache"] = ModelCache(
                layers=new_layer_caches,
                seq_len=seq_offset + L,
            )

        if return_hidden_states:
            output["vsa_states"] = vsa_states
            output["hidden"] = hidden

        return output

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,       # [B, prompt_len]
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Cached autoregressive generation.

        Phase 1 (prefill):  Full forward pass on the prompt, builds KV cache
                            + VSA state cache.
        Phase 2 (decode):   One token at a time through cached forward.
                            Cost per token: O(W) attention + O(1) VSA.
        """
        self.eval()
        generated = input_ids.clone()

        # ── Prefill ──────────────────────────────────────────────────
        out = self.forward(generated, use_cache=True)
        cache = out["cache"]
        logits = out["logits"][:, -1, :]  # [B, V]

        for _ in range(max_new_tokens):
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0].unique():
                    logits[:, token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-p (nucleus)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits.scatter_(1, sorted_idx, logits.masked_fill(remove, float("-inf")).gather(1, sorted_idx))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            # ── Cached decode step ───────────────────────────────────
            out = self.forward(next_token, cache=cache, use_cache=True)
            cache = out["cache"]
            logits = out["logits"][:, -1, :]

        return generated

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else \
                 (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_groups(
        self,
        base_lr: float = 1e-4,
        vsa_lr_scale: float = 0.1,
    ) -> list:
        """
        Separate parameter groups for differential learning rates.

        VSA decay parameters are warmed up at a lower LR initially
        to avoid state collapse before the gates have opened.
        """
        vsa_params, other_params = [], []
        for name, param in self.named_parameters():
            if "soft_vsa" in name:
                vsa_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": other_params, "lr": base_lr},
            {"params": vsa_params,   "lr": base_lr * vsa_lr_scale,
             "name": "vsa_params"},
        ]
