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
from dataclasses import dataclass, field  # noqa: F811 (field used in config)

from .hybrid_layer import HybridNSVSALayerStack
from .rope import RotaryEmbedding
from .cache import ModelCache
from .rmsnorm import RMSNorm
from .modes import MODE_FAST, MODE_REASON, MODE_DEEP, NUM_MODES


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

    # Misc
    layer_norm_eps: float = 1e-6
    tie_weights: bool = True    # Tie input embedding ↔ LM head weights
    rope_base: float = 10_000.0 # RoPE frequency base

    # Mode tokens (thinking level control)
    mode_token_ids: Dict[int, int] = field(default_factory=dict)  # mode_id → token_id
    default_mode: int = MODE_FAST

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.d_model % 2 == 0, "d_model must be even for RoPE"
        # Resolve num_kv_heads: 0 means same as num_heads (standard MHA)
        if self.num_kv_heads == 0:
            object.__setattr__(self, 'num_kv_heads', self.num_heads)
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

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

        # ── Mode token detection lookup ─────────────────────────────
        # Reverse map: token_id → mode_id for fast scanning
        self._token_to_mode: Dict[int, int] = {
            tok_id: mode_id for mode_id, tok_id in config.mode_token_ids.items()
        }
        self._mode_token_set = set(config.mode_token_ids.values())

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
    # Mode detection
    # ------------------------------------------------------------------

    def _detect_mode_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Scan input_ids for mode control tokens (<|fast|>, <|reason|>, <|deep|>).

        Returns [B] tensor of mode IDs.  If a sequence has no mode token,
        it defaults to config.default_mode.  If multiple are present, the
        *first* one wins.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        default = self.config.default_mode
        mode_ids = torch.full((B,), default, dtype=torch.long, device=device)

        if not self._mode_token_set:
            return mode_ids

        for b in range(B):
            row = input_ids[b]
            for tok_id_int in self._mode_token_set:
                mask = (row == tok_id_int)
                if mask.any():
                    mode_ids[b] = self._token_to_mode[tok_id_int]
                    break  # first mode token wins

        return mode_ids

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
        mode_ids: Optional[torch.Tensor] = None,      # [B] mode per element
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
            mode_ids:            [B] int tensor of reasoning modes.
                                 None = auto-detect from input_ids control tokens.

        Returns dict with keys:
            logits:         [B, L, vocab_size]
            loss:           scalar (if labels provided)
            mode_ids:       [B] detected or provided mode IDs
            cache:          ModelCache (if use_cache)
            vsa_states:     list of per-layer final states (if return_hidden_states)
        """
        B, L = input_ids.shape
        device = input_ids.device
        cfg = self.config

        # ── Embed ────────────────────────────────────────────────────
        x = self.embed_drop(self.embed_norm(self.embedding(input_ids)))  # [B, L, d]

        # ── Detect thinking mode from control tokens ────────────────
        if mode_ids is None:
            mode_ids = self._detect_mode_ids(input_ids)           # [B]

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

        # ── All layers in one pass ───────────────────────────────────
        layer_caches = cache.layers if cache is not None else None
        hidden, vsa_states, new_layer_caches = self.layers(
            x, local_pos, macro_pos, seq_positions,
            layer_caches=layer_caches, use_cache=use_cache,
        )

        # ── LM head ─────────────────────────────────────────────────
        logits = self.lm_head(hidden)  # [B, L, vocab_size]

        output: Dict[str, Any] = {"logits": logits}
        if mode_ids is not None:
            output["mode_ids"] = mode_ids

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
    # Vocabulary resizing
    # ------------------------------------------------------------------

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Expand the token embedding table and LM head to ``new_vocab_size``.

        New rows are zero-initialised so the new tokens start as neutral.
        Tied-weight mode is preserved: after resizing the LM head weight is
        re-pointed at the (new) embedding weight tensor.

        Args:
            new_vocab_size: Target vocabulary size.  Must be >= current size.
        """
        old_vocab = self.config.vocab_size
        if new_vocab_size == old_vocab:
            return
        if new_vocab_size < old_vocab:
            raise ValueError(
                f"resize_token_embeddings: new_vocab_size ({new_vocab_size}) "
                f"must be >= current vocab_size ({old_vocab})."
            )

        d = self.config.d_model
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype

        # ── Embed table ─────────────────────────────────────────────
        new_emb = nn.Embedding(new_vocab_size, d).to(device=device, dtype=dtype)
        nn.init.zeros_(new_emb.weight)
        with torch.no_grad():
            new_emb.weight[:old_vocab] = self.embedding.weight
        self.embedding = new_emb

        # ── LM head ─────────────────────────────────────────────────
        new_lm = nn.Linear(d, new_vocab_size, bias=False).to(device=device, dtype=dtype)
        nn.init.zeros_(new_lm.weight)
        with torch.no_grad():
            new_lm.weight[:old_vocab] = self.lm_head.weight
        self.lm_head = new_lm

        # Re-tie weights if originally tied
        if self.config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        self.config.vocab_size = new_vocab_size

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
