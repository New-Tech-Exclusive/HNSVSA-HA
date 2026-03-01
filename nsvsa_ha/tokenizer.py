"""
Tokenizer abstraction for NSVSA-HA.

Supports:
- `tiktoken` encodings (default: cl100k_base)
- Custom Hugging Face `tokenizers` JSON artifacts

Unified interface used by training/chat scripts:
- encode(text) -> list[int]
- decode(ids) -> str
- n_vocab
- eot_token / eot_token_text
- info() metadata for checkpoint compatibility
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json


@dataclass
class TokenizerInfo:
    backend: str
    name: str
    vocab_size: int
    eot_token: str
    eot_token_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "eot_token": self.eot_token,
            "eot_token_id": self.eot_token_id,
        }


class BaseTokenizer:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    @property
    def n_vocab(self) -> int:
        raise NotImplementedError

    @property
    def eot_token(self) -> int:
        raise NotImplementedError

    @property
    def eot_token_text(self) -> str:
        return "<|endoftext|>"

    def info(self) -> Dict[str, Any]:
        raise NotImplementedError


class TiktokenTokenizer(BaseTokenizer):
    def __init__(self, name: str = "cl100k_base"):
        import tiktoken

        self._tok = tiktoken.get_encoding(name)
        self._name = name

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, allowed_special={self.eot_token_text})

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids))

    @property
    def n_vocab(self) -> int:
        return self._tok.n_vocab

    @property
    def eot_token(self) -> int:
        return int(self._tok.eot_token)

    def info(self) -> Dict[str, Any]:
        return TokenizerInfo(
            backend="tiktoken",
            name=self._name,
            vocab_size=self.n_vocab,
            eot_token=self.eot_token_text,
            eot_token_id=self.eot_token,
        ).to_dict()


class HFJsonTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer_json: str,
        *,
        metadata_json: Optional[str] = None,
        eot_token_text: str = "<|endoftext|>",
    ):
        from tokenizers import Tokenizer

        self._path = str(tokenizer_json)
        self._tok = Tokenizer.from_file(self._path)
        self._meta: Dict[str, Any] = {}

        if metadata_json and Path(metadata_json).exists():
            self._meta = json.loads(Path(metadata_json).read_text())

        self._name = self._meta.get("name", Path(tokenizer_json).stem)
        self._eot_text = self._meta.get("eot_token", eot_token_text)

        eot_id = self._meta.get("eot_token_id")
        if eot_id is None:
            eot_id = self._tok.token_to_id(self._eot_text)
        if eot_id is None:
            raise ValueError(
                f"EOT token '{self._eot_text}' not found in tokenizer vocab. "
                "Provide metadata JSON with eot_token_id or include token in vocab."
            )

        self._eot_id = int(eot_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids))

    @property
    def n_vocab(self) -> int:
        vocab_size = self._meta.get("vocab_size")
        if vocab_size is not None:
            return int(vocab_size)
        return int(self._tok.get_vocab_size())

    @property
    def eot_token(self) -> int:
        return self._eot_id

    @property
    def eot_token_text(self) -> str:
        return self._eot_text

    def info(self) -> Dict[str, Any]:
        return TokenizerInfo(
            backend="hf_tokenizers",
            name=self._name,
            vocab_size=self.n_vocab,
            eot_token=self.eot_token_text,
            eot_token_id=self.eot_token,
        ).to_dict()


def load_tokenizer(
    *,
    tokenizer_name: str = "cl100k_base",
    tokenizer_json: Optional[str] = None,
    tokenizer_meta: Optional[str] = None,
    eot_token_text: str = "<|endoftext|>",
) -> BaseTokenizer:
    """
    Load tokenizer according to CLI/config.

    Priority:
      1) If `tokenizer_json` is provided, load custom HF JSON tokenizer.
      2) Otherwise load `tiktoken` encoding by `tokenizer_name`.
    """
    if tokenizer_json:
        return HFJsonTokenizer(
            tokenizer_json,
            metadata_json=tokenizer_meta,
            eot_token_text=eot_token_text,
        )
    return TiktokenTokenizer(tokenizer_name)


def tokenizer_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Compatibility check for checkpoint vs runtime tokenizer.

    We require vocab size and EOT ID to match exactly.
    Name/backend mismatches are warnings, not fatal, if IDs and size match.
    """
    if not a or not b:
        return True
    return (
        int(a.get("vocab_size", -1)) == int(b.get("vocab_size", -2))
        and int(a.get("eot_token_id", -1)) == int(b.get("eot_token_id", -2))
    )
