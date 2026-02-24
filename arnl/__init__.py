"""
ARNL — Axiomatic-Recurrent Neural Logic
========================================

A Neuro-Symbolic Replacement Architecture for Small Language Models.

Model Name: Arnold
Architecture: ARNL V1.1 — SHADE Edition
Parameter Range: 1B – 24B+

Core Design Principle (V1.1):
    System 1 determines HOW something is said and SIGNALS when it
    does not know WHAT.
    SHADE (System 2) determines WHAT is said whenever System 1
    signals uncertainty.
    No controller mediates between them.
"""

from arnl.config import ARNLConfig
from arnl.model import Arnold
from arnl.system1 import LLSU, LLSUBlock, GLALayer
from arnl.system2 import SHADE, ShadeNode, ShadeEdge
from arnl.decay_engine import DecayEngine
from arnl.salt_tokenizer import SALTTokenizer, SALTEncoding, TokenClass

__version__ = "1.1.0"
__model_name__ = "Arnold"
__arch_name__ = "ARNL"
