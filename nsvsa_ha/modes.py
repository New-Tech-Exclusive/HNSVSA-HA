"""
Thinking-mode constants for NSVSA-HA.

Mode tokens control the level of chain-of-thought reasoning injected
during training and available during inference:

    <|fast|>   → no thinking  (direct answer)
    <|reason|> → some thinking (short inner monologue)
    <|deep|>   → lots of thinking (extended reasoning chain)
"""

from __future__ import annotations
from typing import Dict

# Mode IDs (match token semantics, NOT token IDs)
MODE_FAST   = 0
MODE_REASON = 1
MODE_DEEP   = 2
NUM_MODES   = 3

# Fraction of sequence budget devoted to <|think|>…<|/think|> per mode
# (used by the training script when injecting think blocks)
DEFAULT_THINK_FRACS: Dict[int, float] = {
    MODE_FAST:   0.0,    # no thinking
    MODE_REASON: 0.25,   # ~25% of tokens are thinking
    MODE_DEEP:   0.50,   # ~50% of tokens are thinking
}
