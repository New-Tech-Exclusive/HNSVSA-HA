#!/usr/bin/env python3
"""Quick DeepSpeed integration test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from nsvsa_ha import HybridNSVSA, HybridNSVSAConfig

try:
    import deepspeed
    print(f"deepspeed {deepspeed.__version__} imported OK")
except ImportError as e:
    print(f"SKIP: deepspeed not available ({e})")
    sys.exit(0)

cfg = HybridNSVSAConfig(
    d_model=128, vocab_size=200, num_layers=3, num_heads=4,
    max_seq_len=64, window_size=16, group_size=8,
)
model = HybridNSVSA(cfg).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
print("Model + optimizer created")

ds_config = {
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 0.0,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
    },
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "torch_autocast": {"enabled": True},
}

# Single-GPU: init torch.distributed before DeepSpeed
if not torch.distributed.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="nccl")

engine, opt, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config=ds_config,
)
print("DeepSpeed engine initialized (ZeRO-2 + CPU offload)")

ids = torch.randint(0, 200, (2, 32), device="cuda")
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = engine(ids, labels=ids)
    loss = out["loss"]
print(f"Forward OK, loss={loss.item():.4f}")

engine.backward(loss)
print("Backward OK")

engine.step()
print("Step OK")
print("ALL PASSED")
