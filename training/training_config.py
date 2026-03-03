"""
Central training defaults for `train.py`.

Each option has an inline comment explaining what it controls.
`train.py` automatically loads this file (via argparse set_defaults).
"""

DEFAULT_TRAINING_CONFIG = {
    "dataset": "HuggingFaceFW/fineweb-edu",  # Main HF dataset used for text training.
    "text_field": "text",  # Field name inside each dataset example that contains raw text.
    "split": "train",  # Dataset split to stream for optimization steps.
    "val_dataset": None,  # Optional separate validation dataset; None reuses `dataset`.
    "data_mode": "pretokenized",  # Choose input source: streaming | pretokenized | synthetic.
    "pretokenized_train_dir": "data/pretokenized/train",  # Directory for flat (non-curriculum) train shards.
    "pretokenized_val_dir": "data/pretokenized/val",  # Directory for flat (non-curriculum) val shards.
    "pretokenized_dir_pattern": "data/pretokenized/{split}_{seq_len}",  # Template for curriculum shard dirs.
    "pretokenized_shuffle": True,  # Shuffle shard order and sample rows in pretokenized mode.
    "no_pretokenized_shuffle": False,  # Disable pretokenized shuffling for deterministic iteration.

    "d_model": 640,   # Reduced from 768 → ~93M params (was 126.8M).
    "num_layers": 12,  # Number of hybrid blocks (depth of the network).
    "num_heads": 10,   # head_dim = 640/10 = 64 (standard, efficient on tensor cores).
    "num_kv_heads": 2,  # KV heads for GQA; 10q/2kv = 5× KV sharing.
    "qk_norm": True,  # L2-normalize Q/K with learned per-head temperature.
    "learned_vsa_positions": True,  # Learnable VSA position codebook (vs fixed RoPE-derived).
    "window_size": 256,  # Local attention context window in tokens.
    "group_size": 64,  # Tokens per VSA group update (global memory stride).
    "ffn_ratio": 4.0,  # FFN expansion multiplier (controls MLP capacity).
    "dropout": 0.08,  # Regularization to prevent overfitting on cycled data.
    "max_seq_len": 2048,  # Maximum sequence length the model can handle.
    "vocab_size": 0,  # 0 = auto-match tokenizer vocab size.

    "tokenizer": "cl100k_base",  # Fallback tiktoken encoding when custom tokenizer file is missing.
    "tokenizer_json": "tokenizers/vsa65k_mix/tokenizer.json",  # Default custom tokenizer artifact.
    "tokenizer_meta": "tokenizers/vsa65k_mix/tokenizer_meta.json",  # Metadata with special token IDs.
    "strict_tokenizer_match": False,  # If True, resume/chat fails when checkpoint tokenizer metadata differs.

    "batch_size": 2,  # Micro-batch size per optimizer micro-step (used in flat / stage-1 mode).
    "grad_accum": 16,  # Number of micro-steps before optimizer.step().
    "max_steps": 200_000,  # 40k per phase × 5 phases.
    "max_lr": 3e-4,  # Peak learning rate after warmup.
    "min_lr": 3e-5,  # Final cosine-decay learning rate floor.
    "warmup_steps": 3000,  # Linear warmup duration before cosine decay (stage 1 or flat).
    "weight_decay": 0.1,  # L2-style regularization strength on decay parameter groups.
    "vsa_lr_scale": 0.3,  # Multiplier for VSA parameter-group LR vs base LR.
    "vsa_grad_scale": 30.0,  # Gradient multiplier for VSA params (counteracts EMA attenuation).
    "gate_init_bias": -2.0,  # Initial vsa_gate bias; sigmoid(-2)≈0.12, VSA starts quiet.
    "grad_clip": 1.0,  # Global gradient norm clip threshold.
    "beta1": 0.9,  # AdamW beta1 (momentum of first moment estimate).
    "beta2": 0.95,  # AdamW beta2 (second-moment smoothing; 0.95 adapts faster than 0.999).

    # Thinking-mode control (no PonderNet — modes control think-block injection)
    # Mode tokens are kept in vocab but not used for adaptive computation.

    "eval_interval": 500,  # Run validation every N training steps.
    "eval_steps": 50,  # Number of validation batches per evaluation pass.
    "log_interval": 10,  # Print training metrics every N steps.
    "save_interval": 5000,  # Save periodic checkpoints every N steps.
    "checkpoint_dir": "checkpoints",  # Directory for best/step/final checkpoint files.

    # Checkpoint management
    "keep_best_k": 3,  # Keep the K checkpoints with lowest val loss.
    "keep_recent_k": 2,  # Keep the K most recent periodic checkpoints.

    # Curriculum training
    "enable_curriculum": True,  # Enable multi-stage curriculum. False = flat single-stage.
    "curriculum": [
        # Stage 0 (40K steps): short context, fast iteration.
        {
            "seq_len": 512,
            "pct": 0.20,  # 40,000 / 200,000
            "batch_size": 4,   # doubled vs old default; larger micro-batch → better GPU util
            "grad_accum": 8,   # halved to keep effective batch = 32
            "max_lr": 3e-4,
            "min_lr": 3e-5,
            "intra_warmup": 3000,  # Stage 0 gets the global warmup.
        },
        # Stage 1 (40K steps): medium context, VSA activation.
        {
            "seq_len": 1024,
            "pct": 0.20,  # 40,000 / 200,000
            "batch_size": 2,
            "grad_accum": 16,
            "max_lr": 2e-4,
            "min_lr": 2e-5,
            "intra_warmup": 500,
        },
        # Stage 2 (40K steps): long context, full hierarchical VSA pressure.
        {
            "seq_len": 2048,
            "pct": 0.20,  # 40,000 / 200,000
            "batch_size": 1,
            "grad_accum": 32,
            "max_lr": 1.5e-4,
            "min_lr": 1.5e-5,
            "intra_warmup": 500,
        },
        # Stage 3 (40K steps): think-token supervision.
        {
            "seq_len": 512,
            "pct": 0.20,  # 40,000 / 200,000
            "batch_size": 4,
            "grad_accum": 8,
            "max_lr": 5e-5,
            "min_lr": 5e-6,
            "intra_warmup": 200,
            "cot_mix": 1.0,
        },
        # Stage 4 (40K steps): alignment phase (instruction tuning + RLHF-style preference SFT).
        {
            "seq_len": 512,
            "pct": 0.20,  # 40,000 / 200,000
            "batch_size": 4,
            "grad_accum": 8,
            "max_lr": 3e-5,
            "min_lr": 3e-6,
            "intra_warmup": 200,
            "instruction_mix": 0.7,
            "preference_mix": 0.3,
        },
    ],

    # Diagnostics
    "log_gate_values": True,  # Log per-layer VSA gate means at eval time.
    "log_branch_grad_norms": True,  # Log attention/VSA/FFN gradient norms every log_interval.

    "compile": True,  # Request torch.compile (auto-skipped on Python 3.14+).
    "no_compile": False,  # Force-disable compile when set True.

    "resume": None,  # Checkpoint path to resume optimizer/model state.
    "cot_dataset": "data/cot/high_quality_cot_filtered.txt",  # Filtered high-quality CoT corpus for Stage 3.
    "instruction_dataset": "data/instruction/high_quality_instruction.jsonl",  # Instruction tuning corpus.
    "preference_dataset": "data/preference/high_quality_preference.jsonl",  # Preference pairs/chosen responses.

    "smoke": False,  # Quick tiny-run override mode for functional validation.
    "synthetic": False,  # Use synthetic token batches instead of HF dataset streaming.

    "reason_warmup_steps": 0,  # (deprecated) No longer used.
}
