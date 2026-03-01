# NSVSA-HA: Neuro-Symbolic VSA with Hierarchical Attention

A PyTorch implementation of a bipolar Vector Symbolic Architecture ({-1,1}^d) language model that replaces attention with hierarchical VSA operations.

## Architecture Overview

```
Token IDs → Neural Frontend → Bipolar Hypervectors
                                    ↓
                            Position Binding (⊗)
                                    ↓
                         Hierarchical Grouping
                          ┌─────────────────┐
                          │  Local Groups   │  G_j = sgn(Σ x_k ⊙ p_k)
                          │       ↓         │
                          │  Global State   │  S_t = sgn(Σ G_j ⊙ P_j)
                          └─────────────────┘
                                    ↓
                            Query Unbinding (⊘)
                                    ↓
                         Clean-Up Memory (Cosine Sim)
                                    ↓
                       Probability Distribution P(w|S)
```

## Key Features

- **Bipolar VSA Space**: All operations in {-1,1}^d for hardware efficiency
- **Straight-Through Estimator (STE)**: Enables gradient flow through sign function
- **Hierarchical Grouping**: Replaces attention with O(1) memory per group
- **Clean-Up Memory**: Neural retrieval from noisy VSA queries

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from nsvsa_ha import NSVSA, NSVSAConfig

# Create configuration
config = NSVSAConfig(
    d=10_000,           # Hypervector dimension
    vocab_size=50_257,  # Vocabulary size
    group_size=64,      # Tokens per group (K)
    max_seq_length=2048
)

# Initialize model
model = NSVSA(config)

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 128))
output = model(input_ids)
logits = output['logits']  # [batch, seq_len, vocab_size]

# Generation
generated = model.generate(input_ids[:, :10], max_new_tokens=50)
```

## System Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d | 10,000 | High dimensionality ensures quasi-orthogonality |
| V | 32K-50K | Standard vocabulary size |
| K | 64 | Prevents bundling capacity overflow (√d ≈ 100) |
| H | {-1,1}^d | Bipolar space for efficient operations |

## VSA Operations

### Binding (⊗) - Element-wise Multiplication
```python
b = x ⊙ p  # Hadamard product
```
- Reversible: unbinding is the same operation
- Preserves quasi-orthogonality

### Bundling (⊕) - Sum + Threshold
```python
M = sgn(Σ b_i)
```
- Creates superposition of vectors
- Capacity limit: ~√d items

### Unbinding (⊘) - Self-Inverse Property
```python
x_noisy = M ⊙ p  # p ⊙ p = 1, so this recovers x
```

## Module Structure

```
nsvsa_ha/
├── __init__.py          # Package exports
├── config.py            # Configuration classes
├── vsa_ops.py           # Core VSA operations with STE
├── frontend.py          # Neural Frontend (Codebook Embedder)
├── positional.py        # VSA Positional Encodings
├── grouping.py          # Hierarchical Grouping
├── backend.py           # Clean-Up Memory
├── model.py             # Main NSVSA model
├── training.py          # Training loop and losses
├── utils.py             # Utilities and debugging
└── example.py           # Usage examples
```

## Training

### Loss Functions

1. **Cross-Entropy Loss**: Standard next-token prediction
2. **Semantic Contrastive Loss (InfoNCE)**: Ensures similar tokens have similar embeddings pre-binarization

```python
from nsvsa_ha.training import NSVSATrainer, TrainingConfig

training_config = TrainingConfig(
    batch_size=32,
    num_epochs=100,
    warmup_steps=1000
)

trainer = NSVSATrainer(
    model=model,
    config=training_config,
    train_dataloader=train_loader
)

trainer.train()
```

### Gradient Flow

```
Loss → Cosine Similarity → Query Vector → Unbind/Bundle/Bind (linear + STE) → MLP → Embeddings
```

The STE bypasses the zero derivative of sgn() by using identity gradients.

## Known Vulnerabilities

### 1. Gradient Starvation
**Problem**: Poor STE implementation causes model to predict majority-class only.

**Solution**: MLP dimensions ramp smoothly (V → 1024 → 4096 → d) to preserve gradient fidelity.

### 2. Threshold Deadzones
**Problem**: When sum equals exactly 0, sgn() returns 0, breaking bipolar multiplication.

**Solution**: Force zeros to {-1, 1} via random assignment or positive forcing:
```python
def ste_sign(x, deadzone_strategy="random"):
    out = torch.sign(x)
    zeros = (out == 0)
    if deadzone_strategy == "random":
        out[zeros] = torch.randint(0, 2, zeros.shape) * 2 - 1
    else:
        out[zeros] = 1
    return out
```

## Theoretical Properties

### Quasi-Orthogonality
For d=10,000, random bipolar vectors have:
- Expected dot product: E[x·y] = 0
- Variance: Var[x·y] = d
- Normalized similarity: ~N(0, 1/d) ≈ N(0, 0.0001)

### Bundling Capacity
Maximum items before noise > signal: ~√d ≈ 100 for d=10,000

With K=64, we stay safely within limits.

## Example: VSA Operations

```python
from nsvsa_ha.vsa_ops import VSAOperations, ste_sign

vsa = VSAOperations()
d = 10_000

# Create vectors
x = ste_sign(torch.randn(d))
p = ste_sign(torch.randn(d))

# Bind
bound = vsa.bind(x, p)

# Unbind (recovers x)
recovered = vsa.unbind(bound, p)
similarity = vsa.similarity(recovered, x)  # ≈ 1.0

# Bundle
y = ste_sign(torch.randn(d))
bundled = vsa.bundle(torch.stack([x, y]))
# bundled is similar to both x and y
```

## Citation

This implementation is based on the architectural specification for Neuro-Symbolic VSA with Hierarchical Attention (NSVSA-HA).

## License

MIT License
