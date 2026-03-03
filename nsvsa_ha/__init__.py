"""
NSVSA-HA: Neuro-Symbolic VSA with Hierarchical Attention

Hybrid architecture: continuous soft-VSA global state + local windowed attention
+ RoPE positional encodings + SwiGLU FFN, with full real-gradient flow (no STE).

  from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig
"""

from .hybrid_model import HybridNSVSA, HybridNSVSAConfig
from .soft_vsa import SoftVSAStateUpdate, SoftBundle
from .rope import RotaryEmbedding
from .local_attention import LocalWindowedAttention, repeat_kv
from .ffn import SwiGLUFFN, build_ffn
from .hybrid_layer import HybridNSVSALayer, HybridNSVSALayerStack
from .tokenizer import BaseTokenizer, load_tokenizer, tokenizer_compatible
from .modes import (
    MODE_FAST, MODE_REASON, MODE_DEEP, NUM_MODES, DEFAULT_THINK_FRACS,
)
from .rmsnorm import RMSNorm

__version__ = "0.4.0"
__all__ = [
    "HybridNSVSAConfig",
    "HybridNSVSA",
    "SoftVSAStateUpdate",
    "SoftBundle",
    "RotaryEmbedding",
    "LocalWindowedAttention",
    "repeat_kv",
    "SwiGLUFFN",
    "build_ffn",
    "HybridNSVSALayer",
    "HybridNSVSALayerStack",
    "BaseTokenizer",
    "load_tokenizer",
    "tokenizer_compatible",
    "MODE_FAST",
    "MODE_REASON",
    "MODE_DEEP",
    "NUM_MODES",
    "DEFAULT_THINK_FRACS",
    "RMSNorm",
]
