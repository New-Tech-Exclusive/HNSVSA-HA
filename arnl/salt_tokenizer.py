"""
SALT — Semantic-Anchor-Linked Tokenizer
=========================================

Custom ARNL-native tokenizer that replaces generic BPE tokenizers
(e.g. GPT-2) with a domain-aware encoding pipeline.

Architecture
────────────
1. **Byte-Level BPE** (via HuggingFace ``tokenizers`` Rust library) —
   compact vocab of ~12 000 sub-words, trained on the ARNL corpus.

2. **Token Classification Layer** — each vocab entry is statically
   assigned one of three classes, stored in a flat uint8 numpy array:

       SYN  — syntactic / function tokens  (determiners, prepositions,
              punctuation, common closed-class items)
       SAC  — semantic-anchor candidates   (nouns, proper nouns, numbers,
              capitalised words — likely to be System 2 anchors)
       AMB  — ambiguous / context-dependent (verbs, adjectives, adverbs
              — may or may not carry factual weight)

3. **SALTEncoding** — the encode() method returns a dataclass that carries
   token IDs *plus* classification metadata consumed by:
       • EMLMMasker   — deterministic masking from class_ids
       • SHADE        — concept embedding from anchor_mask
       • Gap Detection — entropy-based triggering from SAC tokens

Special Tokens
──────────────
    [PAD]            = 0
    [BOS]            = 1
    [EOS]            = 2
    [UNK]            = 3
    [ENTITY_NOUN]    = 4
    [ENTITY_NUM]     = 5
    [ENTITY_CONCEPT] = 6
    [ENTITY_DOMAIN]  = 7
    [ANCHOR]         = 8
    [SYN]            = 9

Performance Notes
─────────────────
• Token classes stored as ``np.ndarray(uint8, vocab_size)`` — 12 KB, not 600 KB.
• anchor_mask / syntactic_mask in SALTEncoding are uint8 numpy arrays (1 B/token).
• entity_spans stored as ``np.ndarray(int16, (N, 3))`` — no string allocation.
• encode_batch() calls the HF Rust encode_batch (releases GIL, thread-pool).
• emlm_mask() is fully vectorised — no Python loop, no per-token string calls.
• _skip_ids built once in __init__; _etype_map at module level.
• Token classes saved/loaded as binary .npy (μs load) alongside .json (human).
• Classification uses a static closed-class frozenset — no NLTK on the hot path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ════════════════════════════════════════════════════════════════
# Token Class Enum
# ════════════════════════════════════════════════════════════════

class TokenClass(IntEnum):
    """Static class assigned to every token in the vocabulary."""
    SYN = 0   # syntactic / closed-class
    SAC = 1   # semantic-anchor candidate
    AMB = 2   # ambiguous / context-dependent


# ════════════════════════════════════════════════════════════════
# Module-level constants  (zero-cost at call sites)
# ════════════════════════════════════════════════════════════════

# Special token definitions
SPECIAL_TOKENS: List[str] = [
    "[PAD]",            # 0
    "[BOS]",            # 1
    "[EOS]",            # 2
    "[UNK]",            # 3
    "[ENTITY_NOUN]",    # 4
    "[ENTITY_NUM]",     # 5
    "[ENTITY_CONCEPT]", # 6
    "[ENTITY_DOMAIN]",  # 7
    "[ANCHOR]",         # 8
    "[SYN]",            # 9
]

# Entity token IDs
ENTITY_NOUN_ID    = 4
ENTITY_NUM_ID     = 5
ENTITY_CONCEPT_ID = 6
ENTITY_DOMAIN_ID  = 7

_ALL_ENTITY_IDS = frozenset({
    ENTITY_NOUN_ID, ENTITY_NUM_ID,
    ENTITY_CONCEPT_ID, ENTITY_DOMAIN_ID,
})

# Entity ID → int type code  (column 2 in entity_spans arrays)
_ETYPE_MAP: Dict[int, int] = {
    ENTITY_NOUN_ID:    ENTITY_NOUN_ID,
    ENTITY_NUM_ID:     ENTITY_NUM_ID,
    ENTITY_CONCEPT_ID: ENTITY_CONCEPT_ID,
    ENTITY_DOMAIN_ID:  ENTITY_DOMAIN_ID,
}
# Reverse: type int → short string for display
_ETYPE_NAMES: Dict[int, str] = {
    ENTITY_NOUN_ID:    "NOUN",
    ENTITY_NUM_ID:     "NUM",
    ENTITY_CONCEPT_ID: "CONCEPT",
    ENTITY_DOMAIN_ID:  "DOMAIN",
}

# Closed-class English words → SYN
# ~260 entries covering determiners, prepositions, conjunctions, pronouns,
# auxiliaries, modals, and common function words.
# More accurate for BPE sub-words than NLTK alone (which expects full-sentence
# context and performs poorly on isolated fragments).
_CLOSED_CLASS_WORDS: frozenset = frozenset({
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "all", "both", "each", "every", "few", "many", "much", "no", "other",
    "such", "what", "which", "whatever", "whichever", "whoever",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "by", "despite", "down", "during", "except",
    "for", "from", "in", "inside", "into", "like", "near", "of", "off",
    "on", "onto", "out", "outside", "over", "past", "since", "through",
    "throughout", "to", "toward", "towards", "under", "until", "up",
    "upon", "via", "with", "within", "without",
    # Coordinating / subordinating conjunctions
    "and", "but", "or", "nor", "so", "yet", "although", "because",
    "if", "since", "though", "unless", "until", "when",
    "where", "while", "whether", "whereas",
    # Personal pronouns
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "one", "oneself",
    # Wh- words
    "who", "whom", "whose", "which", "that",
    # Auxiliaries / modals
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "need", "dare", "ought",
    # Common function adverbs / particles
    "not", "n't", "so", "too", "also", "just", "only", "even",
    "very", "quite", "rather", "more", "most", "less", "least",
    "as", "than", "then", "there", "here", "now", "already", "still",
    "yet", "never", "always", "often", "usually", "sometimes",
    "again", "once", "twice", "well",
    # Existential / discourse
    "hence", "thus", "therefore", "however", "moreover",
    "furthermore", "nevertheless", "nonetheless", "meanwhile",
    # Punctuation tokens (as surface strings)
    ".", ",", "!", "?", ";", ":", "'", '"', "`",
    "--", "...", "\u2026", "-", "\u2013", "\u2014",
    "(", ")", "[", "]", "{", "}",
    "'s", "'re", "'ve", "'ll", "'d", "'t",
})

# Penn Treebank POS tags → SYN (used when NLTK is available as tiebreaker)
_SYN_POS_TAGS: frozenset = frozenset({
    "DT", "IN", "CC", "TO", "EX", "MD",
    "PRP", "PRP$", "WDT", "WP", "WP$", "WRB",
    "RP", "UH", "PDT", "FW",
    ".", ",", ":", "``", "''", "-LRB-", "-RRB-", "#", "$", "SYM", "LS",
})
# POS tags → SAC
_SAC_POS_TAGS: frozenset = frozenset({
    "NN", "NNS", "NNP", "NNPS", "CD",
})

# Module-level NLTK readiness flag — avoids nltk.data.find() overhead on
# every classify_vocabulary() call after the first.
_NLTK_READY: bool = False


# ════════════════════════════════════════════════════════════════
# SALTEncoding — output of encode()
# ════════════════════════════════════════════════════════════════

@dataclass
class SALTEncoding:
    """Rich encoding returned by :meth:`SALTTokenizer.encode`.

    Attributes
    ----------
    input_ids : list[int]
        Token IDs (sub-word indices into SALT vocabulary).
    class_ids : list[int]
        Per-token class int (0=SYN, 1=SAC, 2=AMB).
    anchor_mask : np.ndarray(uint8)
        1 where the token is a semantic-anchor candidate (SAC), else 0.
    syntactic_mask : np.ndarray(uint8)
        1 where the token is purely syntactic (SYN), else 0.
    entity_spans : np.ndarray(int16, shape=(N, 3))
        Rows are [start_idx, end_idx, entity_type_id].
        entity_type_id matches the entity token IDs: 4=NOUN, 5=NUM,
        6=CONCEPT, 7=DOMAIN.  Adjacent spans of *different* types are
        kept separate (Q1 correctness fix).  Empty = shape (0, 3).
    """
    input_ids:      List[int]  = field(default_factory=list)
    class_ids:      List[int]  = field(default_factory=list)
    anchor_mask:    np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    syntactic_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    entity_spans:   np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int16))

    # ── Convenience properties ──────────────────────────────────

    @property
    def anchor_ids(self) -> List[int]:
        """Token IDs marked as anchors (SAC)."""
        ids_arr = np.array(self.input_ids, dtype=np.int32)
        return ids_arr[self.anchor_mask.astype(bool)].tolist()

    def to_tensor(self, max_len: Optional[int] = None, pad_id: int = 0) -> torch.Tensor:
        """Return input_ids as a padded/truncated 1-D LongTensor (zero-copy via numpy)."""
        ids = np.array(self.input_ids, dtype=np.int64)
        if max_len is not None:
            if len(ids) < max_len:
                ids = np.pad(ids, (0, max_len - len(ids)), constant_values=pad_id)
            else:
                ids = ids[:max_len]
        return torch.from_numpy(ids.copy())

    def labels_tensor(
        self,
        max_len: Optional[int] = None,
        pad_id: int = 0,
        mask_entities: bool = True,
    ) -> torch.Tensor:
        """Return a labels tensor with entity spans and padding set to -100.

        Padding is masked **by position** (everything beyond the real sequence
        length), not by value, which avoids incorrectly masking any legitimate
        occurrence of token ID 0 mid-sequence.
        """
        real_len = len(self.input_ids)
        ids = np.array(self.input_ids, dtype=np.int64)

        if max_len is not None:
            if len(ids) < max_len:
                ids = np.pad(ids, (0, max_len - len(ids)), constant_values=pad_id)
            else:
                ids = ids[:max_len]
                real_len = min(real_len, max_len)

        labels = torch.from_numpy(ids.copy())

        if mask_entities and len(self.entity_spans) > 0:
            for row in self.entity_spans:
                s = max(int(row[0]), 0)
                e = min(int(row[1]), len(labels))
                if s < e:
                    labels[s:e] = -100

        # Mask padding by position, not by token value (Q2 fix)
        if real_len < len(labels):
            labels[real_len:] = -100
        return labels

    def entity_spans_as_tuples(self) -> List[Tuple[int, int, str]]:
        """Return entity_spans as (start, end, type_str) tuples (for display)."""
        return [
            (int(s), int(e), _ETYPE_NAMES.get(int(t), str(int(t))))
            for s, e, t in self.entity_spans
        ]

    def __len__(self) -> int:
        return len(self.input_ids)





# ════════════════════════════════════════════════════════════════
# Token classification helpers
# ════════════════════════════════════════════════════════════════

def classify_token_text(text: str, pos_tag: Optional[str] = None) -> TokenClass:
    """Classify a single token given its surface text.

    Uses a fast frozenset lookup against *_CLOSED_CLASS_WORDS* first so no
    NLTK context is required on the hot path.  If *pos_tag* is provided it
    is used as an additional signal (useful when NLTK has already been run
    over the full batch).

    Parameters
    ----------
    text : str
        Surface form of the sub-word / word (may have BPE prefix bytes).
    pos_tag : str | None
        Optional Penn Treebank POS tag.

    Returns
    -------
    TokenClass
    """
    # Strip BPE prefix bytes (Sentencepiece ▁, HF Ġ, ASCII space)
    stripped_raw = text.replace("Ġ", "").replace("▁", "").strip()
    stripped = stripped_raw.lower()

    # Fast closed-class lookup (no NLTK needed)
    if stripped in _CLOSED_CLASS_WORDS:
        return TokenClass.SYN

    # NLTK tiebreaker when available
    if pos_tag is not None:
        if pos_tag in _SYN_POS_TAGS:
            return TokenClass.SYN
        if pos_tag in _SAC_POS_TAGS:
            return TokenClass.SAC

    # Heuristic: initial uppercase in a non-trivially-short token → SAC
    if stripped_raw and stripped_raw[0].isupper() and len(stripped_raw) > 1:
        return TokenClass.SAC

    return TokenClass.AMB


def _classify_batch(
    items: List[Tuple[str, int]],
    use_nltk: bool,
) -> Dict[int, int]:
    """Classify a batch (text, id) pairs.  Factored out for ProcessPoolExecutor."""
    global _NLTK_READY

    tagged_pairs: List[Tuple[str, int, Optional[str]]] = []

    if use_nltk:
        import nltk
        if not _NLTK_READY:
            try:
                nltk.data.find("taggers/averaged_perceptron_tagger_eng")
                _NLTK_READY = True
            except LookupError:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                _NLTK_READY = True

        words = []
        for tok_text, _ in items:
            clean = tok_text.replace("Ġ", " ").replace("▁", " ").strip()
            words.append(clean if clean else tok_text)
        tagged = nltk.pos_tag(words)
        for (tok_text, tok_id), (_, pos) in zip(items, tagged):
            tagged_pairs.append((tok_text, tok_id, pos))
    else:
        for tok_text, tok_id in items:
            tagged_pairs.append((tok_text, tok_id, None))

    return {tok_id: int(classify_token_text(txt, pos))
            for txt, tok_id, pos in tagged_pairs}


def classify_vocabulary(
    vocab: Dict[str, int],
    use_nltk: bool = True,
    n_workers: int = 1,
    batch_size: int = 512,
) -> Dict[int, int]:
    """Classify every token in a vocabulary.

    Parameters
    ----------
    vocab : dict[str, int]
        Mapping token-text → token-id.
    use_nltk : bool
        Use NLTK POS tagger for improved accuracy. Set False for fast mode.
    n_workers : int
        Number of parallel worker processes (uses ProcessPoolExecutor).
    batch_size : int
        How many tokens per NLTK batch.

    Returns
    -------
    dict[int, int]
        Mapping token-id → TokenClass value.
    """
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    classes: Dict[int, int] = {}

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_classify_batch, b, use_nltk) for b in batches]
            for fut in futures:
                classes.update(fut.result())
    else:
        for batch in batches:
            classes.update(_classify_batch(batch, use_nltk))

    return classes


# ════════════════════════════════════════════════════════════════
# SALT Tokenizer
# ════════════════════════════════════════════════════════════════

class SALTTokenizer:
    """ARNL-native tokenizer: Byte-Level BPE + numpy-backed token classification.

    Wraps the HuggingFace ``tokenizers`` fast-tokenizer and augments
    every encode() call with classification metadata needed by
    Arnold's subsystems.

    Directory layout after training::

        tokenizer_dir/
            tokenizer.json      — BPE model + merges (HF format)
            token_classes.npy   — uint8 numpy array of length vocab_size (fast μs load)
            token_classes.json  — {token_id: class_int, ...}  (human-readable)
            salt_meta.json      — vocab_size, special tokens, counts, etc.

    Memory footprint of class lookup:
        Old (Dict[int,int]): ~600 KB for 12 000-token vocab
        New (_class_lookup numpy uint8): 12 KB
    """

    def __init__(self, tokenizer_dir: str):
        """Load a pre-trained SALT tokenizer from *tokenizer_dir*."""
        tokenizer_dir = str(tokenizer_dir)
        tok_path  = os.path.join(tokenizer_dir, "tokenizer.json")
        npy_path  = os.path.join(tokenizer_dir, "token_classes.npy")
        cls_path  = os.path.join(tokenizer_dir, "token_classes.json")
        meta_path = os.path.join(tokenizer_dir, "salt_meta.json")

        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"SALT tokenizer not found at {tok_path}. "
                "Run  scripts/train_tokenizer.py  first."
            )

        # Load HF fast tokenizer
        from tokenizers import Tokenizer as HFTokenizer
        self._tok = HFTokenizer.from_file(tok_path)
        self._vocab_size: int = self._tok.get_vocab_size()

        # ── Load class lookup into a compact numpy uint8 array ──
        if os.path.exists(npy_path):
            # μs-fast binary load
            self._class_lookup: np.ndarray = np.load(npy_path)
        elif os.path.exists(cls_path):
            with open(cls_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            arr = np.full(self._vocab_size, int(TokenClass.AMB), dtype=np.uint8)
            for k, v in raw.items():
                idx = int(k)
                if 0 <= idx < self._vocab_size:
                    arr[idx] = int(v)
            self._class_lookup = arr
        else:
            print(f"  [SALT] WARNING: {cls_path} not found — all tokens classified as AMB.")
            self._class_lookup = np.full(self._vocab_size, int(TokenClass.AMB), dtype=np.uint8)

        # ── Pin special-token classes ───────────────────────────
        for i in range(len(SPECIAL_TOKENS)):
            if i < self._vocab_size:
                self._class_lookup[i] = (
                    int(TokenClass.SAC) if i in _ALL_ENTITY_IDS else int(TokenClass.SYN)
                )

        # ── Pre-compute entity replacement array ────────────────
        # _emlm_repl[token_id] = entity placeholder ID to use when that
        # token is EMLM-masked.  SAC tokens get an entity ID; others keep
        # their own ID (no masking).
        self._emlm_repl: np.ndarray = np.arange(self._vocab_size, dtype=np.uint16)
        vocab = self._tok.get_vocab()
        id_to_text: Dict[int, str] = {v: k for k, v in vocab.items()}
        for tid in range(len(SPECIAL_TOKENS), self._vocab_size):
            if self._class_lookup[tid] == int(TokenClass.SAC):
                surface = id_to_text.get(tid, "")
                _, eid = self._surface_to_entity(surface)
                self._emlm_repl[tid] = eid

        # ── Pre-compute skip set for decode() ───────────────────
        self._skip_ids: frozenset = (
            frozenset({0, 1, 2, 3}) | _ALL_ENTITY_IDS
        )

        # ── Load metadata ───────────────────────────────────────
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta: dict = json.load(f)
        else:
            self._meta = {}

    # ── Properties ──────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return 0   # [PAD]

    @property
    def bos_id(self) -> int:
        return 1   # [BOS]

    @property
    def eos_id(self) -> int:
        return 2   # [EOS]

    @property
    def unk_id(self) -> int:
        return 3   # [UNK]

    @property
    def entity_ids(self) -> frozenset:
        return _ALL_ENTITY_IDS

    @property
    def entity_noun_id(self) -> int:
        return ENTITY_NOUN_ID

    @property
    def entity_num_id(self) -> int:
        return ENTITY_NUM_ID

    @property
    def entity_concept_id(self) -> int:
        return ENTITY_CONCEPT_ID

    @property
    def entity_domain_id(self) -> int:
        return ENTITY_DOMAIN_ID

    @property
    def pad_token_id(self) -> int:
        return self.pad_id

    @property
    def eos_token_id(self) -> int:
        return self.eos_id

    # ── Class query ─────────────────────────────────────────────

    def class_of(self, token_id: int) -> TokenClass:
        """Return the :class:`TokenClass` for *token_id* (O(1), no hash)."""
        if 0 <= token_id < self._vocab_size:
            return TokenClass(int(self._class_lookup[token_id]))
        return TokenClass.AMB

    # Keep old private name for any internal callers
    def _get_class(self, token_id: int) -> TokenClass:
        return self.class_of(token_id)

    # ── Core encoding internals ─────────────────────────────────

    def _build_encoding(self, ids_list: List[int]) -> SALTEncoding:
        """Build a SALTEncoding from a list of token IDs.

        Fully vectorised using numpy; no Python loops over tokens.
        Entity spans respect Q1 correctness: adjacent spans of *different*
        entity types are emitted as separate spans.
        """
        if not ids_list:
            return SALTEncoding(
                input_ids=[],
                class_ids=[],
                anchor_mask=np.empty(0, dtype=np.uint8),
                syntactic_mask=np.empty(0, dtype=np.uint8),
                entity_spans=np.empty((0, 3), dtype=np.int16),
            )

        ids_arr = np.array(ids_list, dtype=np.int32)

        # Clamp out-of-range IDs for class lookup
        clamped = np.clip(ids_arr, 0, self._vocab_size - 1)
        class_arr = self._class_lookup[clamped]  # uint8, shape (N,)

        anchor_mask    = (class_arr == int(TokenClass.SAC)).astype(np.uint8)
        syntactic_mask = (class_arr == int(TokenClass.SYN)).astype(np.uint8)

        # ── Entity span detection (Q1 fix) ──
        # entity_ids_arr contains positions where token is an entity placeholder
        entity_pos = np.where(np.isin(ids_arr, np.array(list(_ALL_ENTITY_IDS), dtype=np.int32)))[0]

        spans: List[Tuple[int, int, int]] = []
        if len(entity_pos) > 0:
            i = 0
            while i < len(entity_pos):
                pos = int(entity_pos[i])
                cur_eid = int(ids_arr[pos])
                start = pos
                # Extend span only while same entity type AND consecutive
                j = i + 1
                end = pos + 1
                while j < len(entity_pos):
                    next_pos = int(entity_pos[j])
                    next_eid = int(ids_arr[next_pos])
                    if next_pos == end and next_eid == cur_eid:
                        end = next_pos + 1
                        j += 1
                    else:
                        break
                spans.append((start, end, cur_eid))
                i = j

        if spans:
            entity_spans = np.array(spans, dtype=np.int16)
        else:
            entity_spans = np.empty((0, 3), dtype=np.int16)

        return SALTEncoding(
            input_ids=ids_list,
            class_ids=class_arr.tolist(),
            anchor_mask=anchor_mask,
            syntactic_mask=syntactic_mask,
            entity_spans=entity_spans,
        )

    # ── Encode / Decode ─────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_encoding: bool = False,
    ) -> Union[List[int], SALTEncoding]:
        """Encode text into token IDs.

        Parameters
        ----------
        text : str
            Raw input text.
        add_special_tokens : bool
            Prepend [BOS] and append [EOS].
        return_encoding : bool
            If True, return a full :class:`SALTEncoding` with classification
            metadata.  Otherwise return a plain ``list[int]`` for drop-in
            compatibility with HuggingFace tokenizer interfaces.

        Returns
        -------
        list[int] | SALTEncoding
        """
        ids: List[int] = list(self._tok.encode(text).ids)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if not return_encoding:
            return ids
        return self._build_encoding(ids)

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        return_encoding: bool = False,
    ) -> Union[List[List[int]], List[SALTEncoding]]:
        """Encode a batch of texts, releasing the GIL into the HF Rust thread pool.

        Parameters
        ----------
        texts : list[str]
        add_special_tokens : bool
        return_encoding : bool
            If True, return list of SALTEncoding objects.

        Returns
        -------
        list[list[int]] | list[SALTEncoding]
        """
        # HF Rust encode_batch releases the GIL — ~10-20× faster than
        # calling self._tok.encode() N times in a Python loop.
        hf_batch = self._tok.encode_batch(texts)

        results = []
        for hf_enc in hf_batch:
            ids: List[int] = list(hf_enc.ids)
            if add_special_tokens:
                ids = [self.bos_id] + ids + [self.eos_id]
            if return_encoding:
                results.append(self._build_encoding(ids))
            else:
                results.append(ids)
        return results

    def decode(self, ids: Union[List[int], "torch.Tensor"], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        ids : list[int] | torch.Tensor
        skip_special_tokens : bool
            Strip [PAD], [BOS], [EOS], [UNK], and entity placeholder tokens.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        if skip_special_tokens:
            # self._skip_ids built once in __init__ — no allocation here
            ids = [i for i in ids if i not in self._skip_ids]
        return self._tok.decode(ids)

    # ── EMLM Masking ────────────────────────────────────────────

    def emlm_mask(
        self,
        text: str,
        add_special_tokens: bool = True,
        amb_mask_prob: float = 0.0,
    ) -> SALTEncoding:
        """Encode with EMLM masking: replace SAC (and optionally AMB) tokens.

        SAC tokens are replaced by entity placeholders.  AMB tokens are
        masked probabilistically when *amb_mask_prob* > 0.

        Entity type heuristic (per token):
          • Starts with a digit → [ENTITY_NUM]
          • Starts with uppercase letter → [ENTITY_NOUN]
          • Otherwise → [ENTITY_CONCEPT]

        Parameters
        ----------
        text : str
        add_special_tokens : bool
        amb_mask_prob : float
            Probability [0, 1] that an AMB token is also masked. (Q3 fix)

        Returns
        -------
        SALTEncoding with entity_spans marking every replaced region.
        """
        enc = self.encode(text, add_special_tokens=add_special_tokens, return_encoding=True)

        ids_arr   = np.array(enc.input_ids, dtype=np.int32)
        class_arr = np.array(enc.class_ids, dtype=np.uint8)

        # Vectorised SAC replacement via pre-computed _emlm_repl
        sac_mask = (class_arr == int(TokenClass.SAC))
        masked_arr = ids_arr.copy()
        masked_arr[sac_mask] = self._emlm_repl[ids_arr[sac_mask]].astype(np.int32)

        # Probabilistic AMB replacement (Q3)
        if amb_mask_prob > 0.0:
            amb_mask = (class_arr == int(TokenClass.AMB))
            rng_mask = np.random.random(len(ids_arr)) < amb_mask_prob
            replace_amb = amb_mask & rng_mask
            if replace_amb.any():
                masked_arr[replace_amb] = self._emlm_repl[ids_arr[replace_amb]].astype(np.int32)

        # Rebuild encoding from masked IDs (to get correct spans)
        masked_ids: List[int] = masked_arr.tolist()
        return self._build_encoding(masked_ids)

    # ── Utility ─────────────────────────────────────────────────

    def _id_to_text(self, token_id: int) -> str:
        return self._tok.id_to_token(token_id) or ""

    @staticmethod
    def _surface_to_entity(surface: str) -> Tuple[str, int]:
        """Map a token surface form to an entity (type_str, entity_id)."""
        clean = surface.replace("Ġ", "").replace("▁", "").strip()
        if not clean:
            return "CONCEPT", ENTITY_CONCEPT_ID
        if clean[0].isdigit():
            return "NUM", ENTITY_NUM_ID
        if clean[0].isupper():
            return "NOUN", ENTITY_NOUN_ID
        return "CONCEPT", ENTITY_CONCEPT_ID

    def get_vocab(self) -> Dict[str, int]:
        return self._tok.get_vocab()

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tok.id_to_token(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def bad_words_ids(self) -> List[List[int]]:
        """Entity token IDs formatted for HF generation ``bad_words_ids``."""
        return [[eid] for eid in sorted(_ALL_ENTITY_IDS)]

    def anchor_token_ids(self, ids: List[int]) -> List[int]:
        return [tid for tid in ids if self.class_of(tid) == TokenClass.SAC]

    def syntactic_token_ids(self, ids: List[int]) -> List[int]:
        return [tid for tid in ids if self.class_of(tid) == TokenClass.SYN]

    # ── Training ────────────────────────────────────────────────

    @staticmethod
    def train_from_corpus(
        corpus_files: List[str],
        output_dir: str,
        vocab_size: int = 12_000,
        min_frequency: int = 2,
        use_nltk: bool = True,
        n_workers: int = 1,
    ) -> "SALTTokenizer":
        """Train a new SALT tokenizer from scratch.

        Steps:
            1. Train a ByteLevel BPE tokenizer on *corpus_files*.
            2. Add special tokens with reserved IDs 0-9.
            3. Classify the vocabulary (NLTK optional, parallel optional).
            4. Save tokenizer.json, token_classes.npy, token_classes.json,
               salt_meta.json.

        Parameters
        ----------
        corpus_files : list[str]
        output_dir : str
        vocab_size : int
        min_frequency : int
        use_nltk : bool
            Use NLTK POS tagger for classification (default True).
        n_workers : int
            Parallel workers for classification (default 1).

        Returns
        -------
        SALTTokenizer
        """
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        os.makedirs(output_dir, exist_ok=True)

        print(f"[SALT] Training Byte-Level BPE (vocab_size={vocab_size}) …")

        tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size - len(SPECIAL_TOKENS),
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train(corpus_files, trainer)

        # Verify special-token IDs
        for expected_id, tok_str in enumerate(SPECIAL_TOKENS):
            actual_id = tokenizer.token_to_id(tok_str)
            if actual_id != expected_id:
                print(f"  [SALT] WARNING: {tok_str} got id={actual_id}, expected {expected_id}")

        tok_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tok_path)
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"[SALT] BPE trained — vocab_size={actual_vocab_size}, saved → {tok_path}")

        # Classify
        mode_str = f"NLTK+frozenset (workers={n_workers})" if use_nltk else "frozenset-only"
        print(f"[SALT] Classifying vocabulary [{mode_str}] …")
        vocab = tokenizer.get_vocab()
        token_classes_dict = classify_vocabulary(
            vocab, use_nltk=use_nltk, n_workers=n_workers
        )

        # Force special-token classes
        for i in range(len(SPECIAL_TOKENS)):
            token_classes_dict[i] = (
                int(TokenClass.SAC) if i in _ALL_ENTITY_IDS else int(TokenClass.SYN)
            )

        # Save as binary numpy array (μs load) + JSON (human-readable)
        arr = np.full(actual_vocab_size, int(TokenClass.AMB), dtype=np.uint8)
        for k, v in token_classes_dict.items():
            if 0 <= int(k) < actual_vocab_size:
                arr[int(k)] = int(v)

        npy_path = os.path.join(output_dir, "token_classes.npy")
        np.save(npy_path, arr)
        print(f"[SALT] Token classes (binary) saved → {npy_path}")

        cls_path = os.path.join(output_dir, "token_classes.json")
        with open(cls_path, "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in token_classes_dict.items()}, f)
        print(f"[SALT] Token classes (JSON)   saved → {cls_path}")

        counts = {
            "SYN": int(np.sum(arr == int(TokenClass.SYN))),
            "SAC": int(np.sum(arr == int(TokenClass.SAC))),
            "AMB": int(np.sum(arr == int(TokenClass.AMB))),
        }
        print(f"[SALT] Classification: SYN={counts['SYN']}, SAC={counts['SAC']}, AMB={counts['AMB']}")

        meta = {
            "vocab_size": actual_vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "n_special": len(SPECIAL_TOKENS),
            "use_nltk": use_nltk,
            "n_workers": n_workers,
            "classification_counts": counts,
        }
        meta_path = os.path.join(output_dir, "salt_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[SALT] Metadata saved → {meta_path}")

        return SALTTokenizer(output_dir)

    def __repr__(self) -> str:
        counts = {
            "SYN": int(np.sum(self._class_lookup == int(TokenClass.SYN))),
            "SAC": int(np.sum(self._class_lookup == int(TokenClass.SAC))),
            "AMB": int(np.sum(self._class_lookup == int(TokenClass.AMB))),
        }
        return (
            f"SALTTokenizer(vocab_size={self._vocab_size}, "
            f"SYN={counts['SYN']}, SAC={counts['SAC']}, AMB={counts['AMB']})"
        )
