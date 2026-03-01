  
**ARNOLD**

**Hybrid NSVSA-HA Language Model**

*Technical Specification — Version 1.0*

136M Parameters  |  12 Layers  |  768 Model Dimension

Neuro-Symbolic VSA  |  Sliding-Window Attention  |  SwiGLU FFN

*Classification: Internal Technical Documentation*

February 2026

# **1\. Executive Summary & Design Philosophy**

Arnold is a custom 136-million parameter autoregressive language model built on the Hybrid NSVSA-HA (Neuro-Symbolic Vector Symbolic Architecture with Hybrid Attention) framework. It is designed to overcome the fundamental quadratic scaling dilemma of standard Transformer architectures without sacrificing either local syntactic fidelity or long-range semantic coherence.

**The architecture achieves this by cleanly separating two cognitive responsibilities:**

* **Local Grammar & Syntax:** Handled by a sliding-window Multi-Head Causal Attention mechanism restricted to the most recent W \= 256 tokens. Complexity scales sub-quadratically at O(W · L) rather than the O(L²) of full attention.

* **Long-Range Semantic Context:** Handled by a recurrent Vector Symbolic Architecture (VSA) memory backbone. Global context is compressed into a fixed-size, L2-normalized continuous state vector per layer, enabling theoretically infinite-context capability at a constant O(1) per-token inference cost with no KV-cache required.

Unlike alternative recurrent paradigms such as Mamba's Selective State Space model or Griffin's RG-LRU, Arnold's VSA backbone employs algebraically proven operators — specifically Hadamard product binding and Fréchet mean bundling — adapted into a fully continuous and differentiable format ("Soft VSA"). This preserves the theoretical rigor of classical Hyperdimensional Computing while remaining compatible with gradient-based training.

# **2\. Global Dimensions & Hyperparameters**

## **2.1. Core Structural Sizes**

The following table enumerates Arnold's primary architectural dimensions:

| Parameter | Value | Notes |
| :---- | :---- | :---- |
| **Total Parameters** | 136,031,232 | \~136M; weight-tied LM head saves \~37M |
| **Model Dimension (d\_model)** | 768 | Core hidden-state width |
| **Number of Layers (L)** | 12 | Identical HybridLayer blocks |
| **Attention Heads (H)** | 12 | Multi-head causal attention |
| **Head Dimension (d\_head)** | 64 | 768 ÷ 12 |
| **Vocabulary Size (V)** | 48,000 | BPE tokenizer; English \+ code |

## **2.2. Sub-Layer Hyperparameters**

| Hyperparameter | Value | Purpose |
| :---- | :---- | :---- |
| **Local Window Size (W)** | 256 tokens | Bounds attention lookback; O(W·L) complexity |
| **VSA Group Size (K)** | 64 tokens | Chunk size for local binding before EMA update |
| **FFN Expansion Ratio** | 4.0× | SwiGLU gated architecture (LLaMA-style) |
| **Intermediate FFN Dim** | 2,048 | ceil(768 × 4 × 2/3) rounded to nearest 64 |
| **RoPE Base Frequency** | 10,000.0 | Rotary position embedding base |

## **2.3. Optimization & Training Precision**

| Setting | Value / Configuration |
| :---- | :---- |
| **Numerical Precision** | torch.bfloat16 via Automatic Mixed Precision (AMP) |
| **Optimizer** | AdamW with fused CUDA kernel (where available) |
| **Beta Coefficients** | β1 \= 0.9, β2 \= 0.95 (faster adaptation than standard 0.999) |
| **Weight Decay** | 0.1 (applied to parameters with ≥2 dimensions only) |
| **Gradient Clipping** | Global norm clip at 1.0 |
| **Dropout Rate** | 0.08 (applied to Attention and FFN sub-layers) |

# **3\. Data Flow & Model Topology**

Given an input token sequence T of shape \[B × L\] where B is batch size and L is sequence length, the model processes data through three distinct stages.

## **3.1. Input Stage**

The raw integer token sequence enters the model through two operations:

* **Token Embedding:** Tokens are projected via embedding matrix E ∈ ℝ^(V × d\_model), yielding X₀ ∈ ℝ^(B × L × 768). Weights initialized via N(0, 0.02).

* **Positional Encoding Strategy:** No absolute learned positional embeddings are added at the input stage. All position awareness is handled internally within each layer via Rotary Positional Embeddings (RoPE), which provides relative position sensitivity without static position biases.

* **Initial Normalization:** X₀ is passed through LayerNorm before entering the layer stack.

## **3.2. Core Layer Stack (12 × HybridLayer)**

The normalized tensor propagates sequentially through 12 identical HybridLayer blocks under a strict pre-norm residual architecture. Each layer receives the output of the previous and produces an updated token representation alongside a VSA recurrent state:

**X\_i, S\_i \= HybridLayer\_i(X\_{i-1})**

S\_i is the recurrent global VSA state vector of dimension d\_model \= 768, maintained and updated incrementally across the full input sequence. This state is independent per layer.

## **3.3. Output Head (Weight-Tied)**

Following the 12th layer, the final representation is processed by:

* **Final Normalization:** X\_out \= LayerNorm(X\_12)

* **LM Head (Weight-Tied):** Logits \= X\_out × E^T, where E is the transposed token embedding matrix from the input stage. Weight tying eliminates a separate \~37M parameter projection matrix and acts as a strong implicit regularizer by coupling the output distribution space with the input embedding space.

# **4\. Deep Dive: The Hybrid Sub-Layers**

Within a single HybridLayer, the token sequence tensor undergoes three successive additive residual updates. These three sub-layers address complementary learning objectives and are applied in strict sequence.

## **4.1. Local Windowed Causal Attention**

This sub-layer establishes precise, local linguistic structure: noun-verb agreement, adjective binding, pronoun resolution within a phrase, and short-range dependency parsing. It is intentionally bounded to prevent it from being relied upon for cross-document context retrieval.

### ***Mechanism & Masking***

The causal windowed attention mask is defined as:

**Mask(p, q) \= 0.0  if  0 ≤ p−q \< 256  ;  −∞  otherwise**

This ensures that each token can attend only to tokens within the immediately preceding 256-token window. Tokens in the future (standard causal masking) and tokens more than 256 positions in the past are both masked to negative infinity pre-softmax.

### ***Computational Steps***

* **Norm:** x\_attn\_in \= LayerNorm(X\_{i-1})

* **Fused QKV Projection:** A single fused W\_qkv matrix generates Queries, Keys, and Values simultaneously for efficiency.

* **RoPE Application:** 1D Rotary Position Embeddings are applied to Q and K tensors to inject relative position awareness.

* **Windowed Masked Self-Attention:** Scaled dot-product attention with the additive window mask applied.

* **Output Projection:** AttnO \= Attention(Q, K, V) × W\_o

* **Residual Update:** X^(1) \= X\_{i-1} \+ Dropout(AttnO)

## **4.2. Soft Vector Symbolic Architecture (VSA) Memory Backbone**

This is the core novelty of the Arnold architecture. The VSA backbone replaces the role that full (global) self-attention plays in standard Transformers for managing long-term semantic context. Rather than re-attending over all past tokens, it maintains an algebraically structured running memory state that is updated recurrently.

### ***Step A: Local Grouping Phase (Binding & Bundling)***

Tokens are chunked into groups of K \= 64\. Within each group, each token is bound to its intra-group position and the group is then bundled into a single representative vector:

* **Unit Normalization:** x\_unit \= x / ||x||\_2  (push to unit hypersphere)

* **Binding (Hadamard):** B\_{j,k} \= x\_k ⊙ P\_{local\_k}  where P\_local is constructed via the RoPE kernel applied to a ones vector.

* **Bundling (Fréchet Mean):** G\_j \= normalize((1/K) × Σ B\_{j,k}, p=2, dim=-1)  — Continuous soft bundling, gradient-friendly (no binary sgn() required).

### ***Step B: Global Causal State (Exponential Moving Average)***

The recurrent core of the VSA backbone. A running state S\_j compresses all prior group summaries into a fixed-size vector via a learned, per-dimension EMA decay gate:

* **Decay Gate Vector (α):** α ∈ ℝ^d\_model is learned per dimension, initialized at \~0.9. This allows different semantic dimensions to have independently calibrated memory horizons.

* **Macro-Position Binding:** M\_j \= G\_j ⊙ P\_{macro\_j}  (bind the group summary to its chunk-level position index)

* **EMA Update:** S\_j \= α ⊙ S\_{j-1} \+ (1 − α) ⊙ M\_j

* **Renormalization:** S\_j ← normalize(S\_j, p=2, dim=-1)  (maintain L2 normalization invariant)

### ***Step C: Decoding / Querying the State***

Each token queries the accumulated global state to retrieve relevant long-range context:

* **Unbind & Query:** Q \= Linear\_Query\_Proj(S\_j ⊙ P\_{local\_k})  — unbinding the local position retrieves what the state "expects" at this slot.

* **Learned Scalar Gate (β):** β ∈ ℝ^d\_model initialized to −2.0 ⇒ σ(−2.0) ≈ 0.12. Critical: The VSA branch is nearly silent at initialization, allowing the Attention sub-layer to stabilize grammar learning before the VSA begins asserting long-range memory contributions.

* **Gated Output:** V\_out \= σ(β) ⊙ Linear\_out(Q)

* **Residual Update:** X^(2) \= X^(1) \+ V\_out

## **4.3. Feed-Forward Network (SwiGLU)**

The FFN provides dense, non-linear token-wise transformations, modeled after the LLaMA architecture's SwiGLU variant for improved expressivity over standard ReLU FFNs.

* **Norm:** x\_ffn\_in \= LayerNorm(X^(2))

* **Fused Gated Up-Projection:** A single linear expands to 2 × 2048 dimensions. The two halves serve as Gate and Up streams.

* **SwiGLU Activation:** H \= SiLU(Gate) ⊙ Up

* **Down-Projection:** FfnO \= H × W\_down  (projects 2048 → 768\)

* **Residual Update:** X\_final \= X^(2) \+ Dropout(FfnO)

# **5\. Tokenization Protocol**

| Property | Value |
| :---- | :---- |
| **Tokenizer Name** | vsa48k\_en\_code |
| **Algorithm** | Byte-Pair Encoding (BPE) with HuggingFace compatibility layer |
| **Vocabulary Size** | 48,000 uniquely coded tokens |
| **Training Distribution** | English web-text and programming code (dual-domain optimized) |
| **End-of-Sequence Token** | \<|endoftext|\>  (Token ID: 3\) — uniform document and sequence boundary marker |

The tokenizer is deliberately minimal in its control token set to avoid ambiguity. The single \<|endoftext|\> token functions as a universal delimiter for both document boundaries and sequence termination, simplifying data pipeline logic during training and inference alike.

# **6\. Efficient Inference Cache**

Standard autoregressive generation in Arnold achieves O(W · d) per-token complexity — a fixed cost bounded by the window size W rather than by the growing sequence length L as in full-attention Transformers. This is accomplished by maintaining an explicit two-part stateful cache during generation.

## **6.1. AttentionCache**

Stores Keys and Values for the sliding window attention mechanism only. The buffer is dynamically maintained and bounded:

| Shape | \[Batch, Heads, ≤ 256, HeadDim\]  —  i.e., \[B, 12, W, 64\] |
| :---: | :---- |

As generation proceeds and sequence length exceeds 256, the oldest key-value pairs are gracefully evicted from the buffer. The buffer never grows beyond the fixed window size, ensuring constant memory consumption per active generation stream regardless of total sequence length.

## **6.2. VSACache**

Stores the full mathematical state of the VSA recurrent system, allowing generation to resume seamlessly from any point:

* **global\_state:** \[Batch, 768\] — the current EMA state S\_j, representing the compressed summary of all tokens seen so far.

* **group\_accum:** \[Batch, 768\] — partial accumulator for the in-progress group of up to K=64 tokens. Stores the running sum of bound token vectors.

Each new token is bound to its local position and added to group\_accum in O(d) time. A full EMA state update (the more expensive Step B operation) triggers exactly once every 64 tokens when a complete group chunk has accumulated. This amortizes the bulk of VSA computation across tokens.

# **7\. Curriculum Training Schedule**

Arnold is trained using a 3-stage escalating sequence length curriculum. The core hypothesis is that exposing the model to full-length 2048-token sequences immediately causes gradient interference between syntactic learning (short-range) and semantic retrieval learning (long-range). By progressively expanding context length, each sub-system has the opportunity to stabilize before being stressed by longer dependencies.

A critical implementation detail: all gradient signals flowing through the Soft VSA parameters are multiplied by an explicit scalar factor of 30.0 via a custom backward hook. This counteracts mathematical gradient attenuation inherent in the EMA recurrence, ensuring VSA parameter learning rates remain symmetric with those of the Attention layers.

## **Training Stages Overview**

| Stage | Steps | Seq. Length | Learning Rate | Primary Objective |
| :---- | :---- | :---- | :---- | :---- |
| **Stage 0** | 0 → 60,000 | 512 tokens | 3×10⁻⁴ (3k warmup) | Embedding solidification; Window Attention establishes core syntactic comprehension |
| **Stage 1** | 60k → 80,000 | 1,024 tokens | 2×10⁻⁴ | Seq. length exceeds W=256; gradient pressure forces VSA gate to open logarithmically |
| **Stage 2** | 80k → 100,000 | 2,048 tokens | 1.5×10⁻⁴ | Deep hierarchical extraction; model learns to trust EMA state over massive context windows |

## **Stage-by-Stage Detail**

### ***Stage 0: Syntactic Foundation (Steps 0 – 60,000 | 60% of training)***

* **Sequence Length:** 512 tokens

* **Effective Batch Size:** 32 sequences

* **Learning Rate:** 3 × 10⁻⁴ with linear warmup over first 3,000 steps

* **Rationale:** At 512 tokens, the full sequence fits within two sliding windows. The VSA gate (β ≈ −2.0 ⇒ output ≈ 0.12) is nearly silent, allowing the attention sub-layer to stabilize embeddings and capture grammar without interference from the memory system.

### ***Stage 1: Context Pressure (Steps 60,000 – 80,000 | 20% of training)***

* **Sequence Length:** 1,024 tokens

* **Effective Batch Size:** 32 sequences

* **Learning Rate:** 2 × 10⁻⁴

* **Rationale:** Sequence length now exceeds four sliding windows. Tokens requiring information from positions \> 256 tokens ago cannot rely on attention; gradient pressure flows into the VSA parameters, opening the β gate and forcing the EMA state to encode retrievable semantic content.

### ***Stage 2: Long-Range Mastery (Steps 80,000 – 100,000 | 20% of training)***

* **Sequence Length:** 2,048 tokens

* **Effective Batch Size:** 32 sequences

* **Learning Rate:** 1.5 × 10⁻⁴

* **Rationale:** Full 2,048-token sequences span 32 group chunks of K=64 tokens each. The model must develop hierarchical contextual extraction strategies, learning to trust the accumulated EMA state for cross-passage semantic consistency.

# **8\. Complexity Analysis & Architectural Comparison**

The table below compares Arnold's asymptotic characteristics against standard Transformer variants:

| Architecture | Train Complexity | Inference / Token | Context Length |
| :---- | :---- | :---- | :---- |
| **Standard Transformer** | O(L² · d) | O(L · d) | Bounded by KV-cache memory |
| **Sliding Window Attn Only** | O(W · L · d) | O(W · d) | Limited to W tokens effective |
| **Arnold (Hybrid NSVSA-HA)** | O(W · L · d) | O(W · d) effective\* | Theoretically infinite via VSA |

*\* VSA inference cost is O(d) per token amortized, with a slightly higher O(K · d) operation once per K=64 tokens for EMA state update. Effective per-token cost is dominated by the attention window.*