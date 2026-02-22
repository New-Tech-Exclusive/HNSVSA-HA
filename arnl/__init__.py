"""
ARNL — Axiomatic-Recurrent Neural Logic
========================================

A Neuro-Symbolic Replacement Architecture for Small Language Models.

Model Name: Arnold
Architecture: ARNL V1.0
Parameter Range: 1B – 24B+
Target: High-Stakes Factual Domains and General-Purpose Factual AI

Core Design Principle:
    System 1 determines HOW something is said.
    System 2 determines WHAT is said.
    The Reasoning Head determines WHETHER and WHEN System 2 influences System 1.
"""

from arnl.config import ARNLConfig
from arnl.model import Arnold
from arnl.system1 import LLSU, LLSUBlock, GLALayer
from arnl.system2 import HyperAdjacencyMap, Hyperedge
from arnl.reasoning_head import ReasoningHead
from arnl.injection_bridge import InjectionBridge
from arnl.decay_engine import DecayEngine

__version__ = "1.0.0"
__model_name__ = "Arnold"
__arch_name__ = "ARNL"

__all__ = [
    "ARNLConfig",
    "Arnold",
    "LLSU",
    "LLSUBlock",
    "GLALayer",
    "HyperAdjacencyMap",
    "Hyperedge",
    "ReasoningHead",
    "InjectionBridge",
    "DecayEngine",
]
