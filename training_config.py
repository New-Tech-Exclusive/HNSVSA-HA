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
    "dropout": 0.05,  # Regularization; 0.05 balances underfitting and overfitting on cycled data.
    "max_seq_len": 2048,  # Maximum sequence length the model can handle.
    "vocab_size": 0,  # 0 = auto-match tokenizer vocab size.

    "tokenizer": "cl100k_base",  # Fallback tiktoken encoding when custom tokenizer file is missing.
    "tokenizer_json": "tokenizers/vsa65k_mix/tokenizer.json",  # Default custom tokenizer artifact.
    "tokenizer_meta": "tokenizers/vsa65k_mix/tokenizer_meta.json",  # Metadata with special token IDs.
    "strict_tokenizer_match": False,  # If True, resume/chat fails when checkpoint tokenizer metadata differs.

    "batch_size": 2,  # Micro-batch size per optimizer micro-step (used in flat / stage-1 mode).
    "grad_accum": 16,  # Number of micro-steps before optimizer.step().
    "max_steps": 163840,  # 32,768 per phase × 5 phases.
    "max_lr": 3e-4,  # Peak learning rate after warmup.
    "min_lr": 3e-5,  # Final cosine-decay learning rate floor.
    "warmup_steps": 3000,  # Linear warmup duration before cosine decay (stage 1 or flat).
    "weight_decay": 0.1,  # L2-style regularization strength on decay parameter groups.
    "vsa_lr_scale": 0.3,  # Multiplier for VSA parameter-group LR vs base LR.
    "vsa_grad_scale": 5.0,   # Gradient multiplier for VSA params. 5× keeps ∇vsa within ~2× of
                             # ∇attn; 30× was too aggressive and risked destabilizing the residual.
    "gate_init_bias": -2.0,  # Initial gate_proj bias; sigmoid(-2)≈0.12, VSA starts quiet.
    "vsa_gate_warmup": 2000, # Freeze gate_proj for this many steps so attention builds a stable
                             # basis before VSA starts contributing. 0 = no freeze.
    "vsa_decay_reg": 0.01,   # Penalize per-dim decay values approaching 1.0 (frozen-state collapse).
                             # Soft penalty: coefficient × mean(clamp(decay - 0.85, min=0)).
    "grad_clip": 0.5,  # Global gradient norm clip threshold.
    "beta1": 0.9,  # AdamW beta1 (momentum of first moment estimate).
    "beta2": 0.95,  # AdamW beta2 (0.95 is a good middle ground; 0.90 made second moments too noisy).

    # Thinking-mode control
    # Mode tokens (<|fast|>/<|reason|>/<|deep|>) control thinking depth:
    #   fast   = no <|think|> blocks (direct answer)
    #   reason = moderate <|think|> blocks (~25% of tokens)
    #   deep   = heavy <|think|> blocks (~50% of tokens)

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
        # Stage 0 (32,768 steps): short context, fast iteration.
        {
            "seq_len": 512,
            "pct": 0.20,  # 32,768 / 163,840
            "batch_size": 4,   # larger micro-batch → better GPU util
            "grad_accum": 8,   # eff batch = 32
            "max_lr": 3e-4,
            "min_lr": 3e-5,
            "intra_warmup": 3000,  # Stage 0 gets the global warmup.
        },
        # Stage 1 (32,768 steps): medium context, VSA activation.
        {
            "seq_len": 1024,
            "pct": 0.20,  # 32,768 / 163,840
            "batch_size": 2,
            "grad_accum": 16,
            "max_lr": 2e-4,
            "min_lr": 2e-5,
            "intra_warmup": 500,
        },
        # Stage 2 (32,768 steps): long context, full hierarchical VSA pressure.
        {
            "seq_len": 2048,
            "pct": 0.20,  # 32,768 / 163,840
            "batch_size": 1,
            "grad_accum": 32,
            "max_lr": 1.5e-4,
            "min_lr": 1.5e-5,
            "intra_warmup": 500,
        },
        # Stage 3 (32,768 steps): think-token supervision.
        # Teaches the model when/how to use <|think|>…<|/think|> spans.
        # 70% CoT + 30% base data preserves language-modeling grounding and
        # prevents memorization of a small CoT file.
        # Mode tokens are baked into CoT rows — skip sample_mode_tokens().
        {
            "seq_len": 512,
            "pct": 0.20,   # 32,768 / 163,840
            "batch_size": 4,
            "grad_accum": 8,
            "max_lr": 8e-5,   # was 5e-5; 3× jump with only 200 warmup was too sharp
            "min_lr": 8e-6,
            "intra_warmup": 750,  # was 200; avoids gradient-scale shock at stage entry
            "cot_mix": 0.70,  # was 1.0; 30% base data retains language-model grounding
            "skip_mode_tokens": True,  # CoT rows already have mode/think spans baked in
        },
        # Stage 4 (32,768 steps): alignment phase (instruction + RLHF-style preference SFT).
        # Instruction/preference data has its own formatting — don't prepend mode tokens.
        {
            "seq_len": 512,
            "pct": 0.20,  # 32,768 / 163,840
            "batch_size": 4,
            "grad_accum": 8,
            "max_lr": 3e-5,
            "min_lr": 3e-6,
            "intra_warmup": 500,  # was 200; gentler entry into alignment stage
            "instruction_mix": 0.7,
            "preference_mix": 0.3,
            "skip_mode_tokens": True,  # instruction/preference data doesn't expect leading mode token
        },
    ],

    # Use a filtered high-quality CoT file for Stage 3.
    "cot_dataset": "data/cot/high_quality_cot_filtered.txt",
    # Alignment datasets for Stage 4.
    "instruction_dataset": "data/instruction/high_quality_instruction.jsonl",
    "preference_dataset": "data/preference/high_quality_preference.jsonl",

    # Diagnostics
    "log_gate_values": True,  # Log per-layer VSA gate means at eval time.
    "log_branch_grad_norms": True,  # Log attention/VSA/FFN gradient norms every log_interval.

    "compile": True,  # Request torch.compile (auto-skipped on Python 3.14+).
    "no_compile": False,  # Force-disable compile when set True.

    # Regularization (opt-in — set > 0.0 to enable)
    "entropy_reg": 0.0,    # Entropy bonus: prevents vocabulary collapse (subtracts α·H(p) from loss).
                           # WARNING: log_softmax over full vocab on every micro-step is expensive.
                           # Typical useful range: 0.005–0.02. Default off.
    "embed_reg": 0.0,      # Embedding RMS norm regularization. Typical: 1e-4. Default off.
    "z_loss": 1e-4,        # Z-loss: penalizes log(Z)² to prevent logit explosion (PaLM/Gemma style).
    "gate_entropy_loss": 0.001,  # Gate entropy loss: prevents VSA gates from saturating at 0 or 1.

    # EMA model weights (opt-in — set > 0.0 to enable)
    "ema_decay": 0.9999,          # EMA decay rate; 0 = disable. 0.9999 ≈ 10k-step half-life.
                               # Adds ~103M params on GPU + update overhead every 10 steps.
    "ema_update_interval": 10, # Update shadow weights every N optimizer steps.

    # VSA stability (opt-in — interacts with torch.compile)
    "spectral_norm_vsa": False,  # Apply spectral norm to VSA Linear layers.

    "reason_warmup_steps": 0,  # (deprecated, kept for compat) No longer used.

    # Mode-conditioned reasoning
    "enable_mode_training": True,  # Prepend mode control tokens and inject think blocks per mode.
    "no_mode_training": False,  # Force-disable mode-conditioned training.
    "mode_mix": "0.50,0.30,0.20",  # Mode sampling probabilities: p_fast,p_reason,p_deep.

    "resume": None,  # Checkpoint path to resume optimizer/model state.

    "smoke": False,  # Quick tiny-run override mode for functional validation.
    "synthetic": False,  # Use synthetic token batches instead of HF dataset streaming.
}
