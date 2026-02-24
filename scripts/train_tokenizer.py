#!/usr/bin/env python3
"""
scripts/train_tokenizer.py — Phase 0: Train the SALT Tokenizer
================================================================

Trains a Byte-Level BPE tokenizer on a text corpus, classifies the
resulting vocabulary into SYN / SAC / AMB classes using NLTK POS
tagging, and saves everything to a directory that Arnold can load.

This must be run **before** any model training (Phase 1-3).

Usage
─────
    # From a local text file
    python scripts/train_tokenizer.py \\
        --corpus data/corpus.txt \\
        --vocab-size 32000 \\
        --output-dir tokenizer/

    # From a HuggingFace dataset (downloads + extracts text)
    python scripts/train_tokenizer.py \\
        --hf-dataset wikitext \\
        --hf-config wikitext-2-raw-v1 \\
        --vocab-size 32000 \\
        --output-dir tokenizer/

    # Both sources combined
    python scripts/train_tokenizer.py \\
        --corpus data/extra.txt \\
        --hf-dataset wikitext \\
        --hf-config wikitext-2-raw-v1 \\
        --vocab-size 32000 \\
        --output-dir tokenizer/

Output
──────
    tokenizer/
        tokenizer.json       — Byte-Level BPE model (HuggingFace format)
        token_classes.json   — {token_id: class_int} for every token
        salt_meta.json       — vocab_size, special tokens, class counts
"""

import argparse
import os
import sys
import tempfile
import time

# Ensure parent dir is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_hf_texts(
    dataset_name: str,
    config_name: str | None = None,
    split: str = "train",
    text_field: str = "text",
    max_samples: int | None = None,
) -> list[str]:
    """Download a HuggingFace dataset and return a list of text strings."""
    try:
        import datasets
    except ImportError:
        print("[Phase 0] ERROR: `datasets` package not installed.  pip install datasets")
        sys.exit(1)

    print(f"[Phase 0] Loading HF dataset '{dataset_name}'"
          + (f" / '{config_name}'" if config_name else "")
          + f" (split={split}) …")

    ds = datasets.load_dataset(dataset_name, config_name, split=split)
    texts = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        t = row.get(text_field, "")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())

    print(f"[Phase 0] Extracted {len(texts):,} text documents from HF dataset.")
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 — Train the SALT tokenizer for ARNL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Corpus sources
    parser.add_argument(
        "--corpus", type=str, nargs="*", default=[],
        help="Path(s) to plain-text corpus file(s) for BPE training.",
    )
    parser.add_argument(
        "--hf-dataset", type=str, default=None,
        help="HuggingFace dataset repo ID (e.g. 'wikitext').",
    )
    parser.add_argument(
        "--hf-config", type=str, default=None,
        help="HuggingFace dataset config name (e.g. 'wikitext-2-raw-v1').",
    )
    parser.add_argument(
        "--hf-split", type=str, default="train",
        help="HF dataset split (default: 'train').",
    )
    parser.add_argument(
        "--hf-text-field", type=str, default="text",
        help="Column name containing text in the HF dataset (default: 'text').",
    )
    parser.add_argument(
        "--hf-max-samples", type=int, default=None,
        help="Max number of HF samples to use (default: all).",
    )

    # Tokenizer params
    parser.add_argument(
        "--vocab-size", type=int, default=32_000,
        help="Target vocabulary size (default: 32000).",
    )
    parser.add_argument(
        "--min-frequency", type=int, default=2,
        help="Min merge frequency during BPE training (default: 2).",
    )
    parser.add_argument(
        "--no-nltk", action="store_true", default=False,
        help="Skip NLTK POS tagging; use closed-class frozenset only (faster).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for vocabulary classification (default: 1).",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="tokenizer/",
        help="Directory to save trained SALT tokenizer (default: tokenizer/).",
    )

    # Verification
    parser.add_argument(
        "--verify-classes", action="store_true", default=False,
        help=(
            "After training, sample N tokens from each class and print them "
            "for human review. Useful for sanity-checking classification quality."
        ),
    )
    parser.add_argument(
        "--verify-samples", type=int, default=20,
        help="Number of tokens to sample per class during --verify-classes (default: 20).",
    )

    args = parser.parse_args()

    # ── Gather corpus files ─────────────────────────────────────
    corpus_files: list[str] = []
    temp_files: list[str] = []

    # Local files
    for path in args.corpus:
        if not os.path.exists(path):
            print(f"[Phase 0] WARNING: corpus file not found: {path}")
            continue
        corpus_files.append(os.path.abspath(path))
        print(f"[Phase 0] Corpus file: {path}")

    # HuggingFace dataset → dump to temp file
    if args.hf_dataset:
        texts = extract_hf_texts(
            args.hf_dataset,
            config_name=args.hf_config,
            split=args.hf_split,
            text_field=args.hf_text_field,
            max_samples=args.hf_max_samples,
        )
        if texts:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8",
            )
            for t in texts:
                tmp.write(t + "\n")
            tmp.close()
            corpus_files.append(tmp.name)
            temp_files.append(tmp.name)
            print(f"[Phase 0] HF texts written to temp file: {tmp.name}")
        else:
            print("[Phase 0] WARNING: no texts extracted from HF dataset.")

    if not corpus_files:
        print("[Phase 0] ERROR: no corpus provided. Use --corpus and/or --hf-dataset.")
        sys.exit(1)

    # ── Train ────────────────────────────────────────────────────
    from arnl.salt_tokenizer import SALTTokenizer

    t0 = time.time()

    print()
    print("=" * 60)
    print("  SALT Phase 0 — Tokenizer Training")
    print("=" * 60)
    print(f"  Corpus files:  {len(corpus_files)}")
    print(f"  Vocab size:    {args.vocab_size:,}")
    print(f"  Min frequency: {args.min_frequency}")
    print(f"  NLTK:          {'disabled' if args.no_nltk else 'enabled'}")
    print(f"  Workers:       {args.workers}")
    print(f"  Output dir:    {os.path.abspath(args.output_dir)}")
    print("=" * 60)
    print()

    tok = SALTTokenizer.train_from_corpus(
        corpus_files=corpus_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        use_nltk=not args.no_nltk,
        n_workers=args.workers,
    )

    elapsed = time.time() - t0

    # ── Verify ──────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Verification")
    print("─" * 60)

    test_sentences = [
        "The capital of France is Paris.",
        "Albert Einstein was born in 1879.",
        "Machine learning models require training data.",
        "The temperature is 72 degrees Fahrenheit.",
    ]

    for sent in test_sentences:
        enc = tok.encode(sent, return_encoding=True)
        anchors = enc.anchor_ids
        n_syn = int(enc.syntactic_mask.sum())
        n_sac = int(enc.anchor_mask.sum())
        n_amb = len(enc) - n_syn - n_sac
        tokens = [tok.id_to_token(i) or f"<{i}>" for i in enc.input_ids]
        print(f"\n  \"{sent}\"")
        print(f"    tokens={tokens}")
        print(f"    ids={enc.input_ids}")
        print(f"    SYN={n_syn}  SAC={n_sac}  AMB={n_amb}")
        print(f"    anchors={anchors}")

    # Test EMLM masking
    print("\n  EMLM mask test:")
    emlm_enc = tok.emlm_mask("The capital of France is Paris.")
    print(f"    masked_ids={emlm_enc.input_ids}")
    print(f"    entity_spans={emlm_enc.entity_spans_as_tuples()}")
    decoded = tok.decode(emlm_enc.input_ids, skip_special_tokens=False)
    print(f"    decoded=\"{decoded}\"")

    # ── --verify-classes sampling ────────────────────────────────
    if args.verify_classes:
        import random
        from arnl.salt_tokenizer import TokenClass
        print()
        print("-" * 60)
        print("  Class verification samples (--verify-classes)")
        print("-" * 60)
        vocab = tok.get_vocab()  # str → int
        id_to_text = {v: k for k, v in vocab.items()}
        n = args.verify_samples

        for cls in (TokenClass.SYN, TokenClass.SAC, TokenClass.AMB):
            # Collect token IDs with this class, excluding special tokens
            members = [
                tid for tid in range(len(tok._class_lookup))
                if tok._class_lookup[tid] == int(cls) and tid >= 10
            ]
            sample = random.sample(members, min(n, len(members)))
            surfaces = [id_to_text.get(tid, f"<{tid}>") for tid in sample]
            print(f"\n  {cls.name} ({len(members):,} tokens) — {len(sample)} samples:")
            for tid, surf in zip(sample, surfaces):
                print(f"    [{tid:>5}]  {surf}")

    print(f"\n  Phase 0 complete in {elapsed:.1f}s")
    print(f"  Tokenizer saved to: {os.path.abspath(args.output_dir)}")
    print(f"  Vocab size: {tok.vocab_size:,}")

    # ── Cleanup temp files ──────────────────────────────────────
    for tf in temp_files:
        try:
            os.unlink(tf)
        except OSError:
            pass


if __name__ == "__main__":
    main()
