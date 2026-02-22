  
**ARNL V1.0**

**Axiomatic-Recurrent Neural Logic**

*Full Technical Specification & Architecture Diagram*

A Neuro-Symbolic Replacement Architecture for Small Language Models

Parameter Range: 1B – 24B+  |  Target: High-Stakes Factual Domains and/or general

Purpose factual Artificial Intelligence 

*Classification: External Technical Review*

# **Abstract**

ARNL V1.0 is a heterogeneous dual-process architecture designed to replace the standard Transformer decoder for Small Language Models in the 1–24 billion parameter range. It separates syntactic fluency from factual reliability into two structurally distinct systems — a fact-blind recurrent backbone (System 1\) and a sparse, tiered Hyper-Adjacency Map (System 2\) — coordinated by a central Reasoning Head.

This version incorporates five critical structural mitigations identified through iterative peer review: (1) a Reasoning Head-controlled Semantic Saliency Filter resolving the bootstrap paradox, (2) a learned projection bridge W\_proj resolving the dimensionality mismatch, (3) a tiered dynamic consistency threshold τ replacing the static gate, (4) an Active Contradiction Detection pass with dead-zone protection, and (5) an Overflow Ceiling mechanism preventing unbounded salience accumulation.

The architecture's primary differentiators over standard Transformers are Granular Semantic Auditability, Hard-Lock behavioral constraints, and continual learning stability without catastrophic forgetting.

# **Table of Contents**

# **I. Global Architecture Overview**

ARNL V1.0 is a System 3 Symbiotic Bundle — a heterogeneous, inference-locked architecture in which two independently trained models and one controller operate as a unified generative system. Neither System 1 nor System 2 generates tokens alone; generation is the emergent product of their interaction through the Injection Bridge.

| Core Design Principle |
| :---- |
| *System 1 determines HOW something is said. System 2 determines WHAT is said. The Reasoning Head determines WHETHER and WHEN System 2 influences System 1\.* |

| Component | Role | Architecture Type | Parameter Budget |
| ----- | ----- | ----- | ----- |
| System 1 (LLSU) | Fluidity Core — syntactic generation | GLA / SSM Recurrent | \~80% of total |
| System 2 (Hyper-Adjacency Map) | Reliability Core — factual storage | Sparse Hash Map (non-neural) | Off-budget (database) |
| Reasoning Head | Synchronizer — control logic | Compact classifier \+ symbolic logic | \~15% of total |
| Injection Bridge (W\_proj) | Semantic-to-Syntactic translator | Learned linear projection | \~5% of total |
| Decay Engine | Memory lifecycle manager | Rule-based arithmetic | No parameters |

## **1.1  Parameter Budget Philosophy**

The 80/15/5 split is intentional and load-bearing. System 1 carries the entire weight of fluency and must be parameter-rich enough to produce coherent syntactic structures independently. The Reasoning Head's job is binary gating and projection — it does not need generative capacity, only discriminative precision. W\_proj is a single matrix and is cheap.

At the 7B total parameter scale, this yields approximately: System 1 at 5.6B, Reasoning Head at 1.05B, and W\_proj at 350M. This means System 1 is competitive with standalone 3B-class models in fluency while the overall system retains factual precision that no 7B Transformer can match in bounded domains.

# **II. System 1 — The Syntactic Linear-Latent Selective Unit (LLSU)**

## **2.1  Role and Objective**

System 1 is the Fluidity Core. Its sole objective is to model the transition probability of grammatical structures — the 'shape' of language — without internalizing any factual content. It is responsible for word order, grammatical agreement, prosody, register consistency, and local coherence.

System 1 is designed to be fact-blind by training, not by architectural restriction. It can receive factual content via injection, but it cannot independently recall or confabulate specific entities, dates, or propositions.

## **2.2  Architecture: Gated Linear Attention (GLA) / SSM**

System 1 uses a Selective State-Space Model with Gated Linear Attention rather than standard scaled dot-product attention. This choice is deliberate for three reasons:

* Linear complexity in sequence length — no quadratic memory cost at long contexts.

* Recurrent inference — the model processes tokens as a state machine, compatible with ARNL's autoregressive loop.

* No attention sink artifacts — GLA does not develop the 'early token dominance' pathology common in Transformers.

### **State Transition Formula**

**h\_t \= A\_t ⊙ h\_{t-1}  \+  B\_t ⊙ x\_t**

**Where:**  h\_t \= hidden state at step t  |  A\_t \= selective forget gate (data-dependent)  |  B\_t \= input gate  |  x\_t \= current input token embedding

The selectivity of A\_t and B\_t — their dependence on the input x\_t rather than being fixed — is what distinguishes this from a vanilla RNN and gives System 1 Transformer-comparable context sensitivity at linear cost.

## **2.3  Training: Entity-Masked Language Modeling (EMLM)**

System 1 is pretrained exclusively on EMLM. During training, all factual entities in the corpus are identified and replaced with generic semantic placeholder tokens before the model ever sees them.

| Masked Entity Type | Replacement Token | Preserved Information |
| ----- | ----- | ----- |
| Proper Nouns (people, places) | \[ENTITY\_NOUN\] | Syntactic role, grammatical gender |
| Numerical Data (dates, quantities) | \[ENTITY\_NUM\] | Numerical context, units |
| Named Concepts (products, events) | \[ENTITY\_CONCEPT\] | Conceptual slot in sentence |
| Domain-Specific Terms | \[ENTITY\_DOMAIN\] | Register and field marker |

The effect is that System 1's weights optimize entirely for the transitions between structural elements — prepositions, conjunctions, verb phrases, clause boundaries — rather than memorizing which entities go where. A sentence like 'The \[ENTITY\_NOUN\] of \[ENTITY\_NOUN\] is \[ENTITY\_NOUN\]' trains perfect syntactic pattern learning with zero factual encoding.

*EMLM Verification Test: After training, probe System 1 by asking it to complete 'The capital of France is \_\_\_' without injection. A correctly trained LLSU will output a high-entropy distribution over all nouns — not 'Paris.' If 'Paris' appears with high probability, the EMLM masking was incomplete.*

## **2.4  The Injection Port**

Every LLSU block contains a Latent Summation Gate positioned between the recurrent layer output and the Output Feed-Forward Network (FFN). This gate is the landing surface for W\_proj injections.

The gate is additive — it does not replace h\_t, it biases it. This means the syntactic structure computed by the recurrent layer remains intact; the factual injection merely shifts the distribution toward the target content without overwriting the grammatical frame.

| Why Additive Injection Preserves Fluency |
| :---- |
| Logit-level override (old approach): System 2 forces a token by adding directly to the output softmax. System 1's grammatical context is ignored. Result: factually correct but syntactically jarring. Residual stream injection (ARNL V1.0): System 2 biases h\_t before the FFN. System 1 processes the biased state through its own FFN, producing a token that is both factually guided and syntactically coherent. |

# **III. System 2 — The Sparse Hyper-Adjacency Map**

## **3.1  Role and Structure**

System 2 is the Reliability Core. It is not a neural network — it is a sparse, content-addressable hash map that stores discrete factual pathways called Hyperedges. It has no trainable parameters and does not participate in gradient computation. Its influence on generation is entirely mediated through the Injection Bridge.

## **3.2  The Hyperedge Structure**

**H \= { Hash(C\_salient) → v\_target,  S\_base,  S\_overflow }**

| Field | Type | Range | Description |
| ----- | ----- | ----- | ----- |
| Hash(C\_salient) | Integer key | Hash space | Content-addressable key derived from semantic anchor tokens identified by the Reasoning Head |
| v\_target | Vector pointer | R^d (frozen) | Pointer to the target token's embedding in the frozen hyperspherical space used by the Reasoning Head |
| S\_base | Integer counter | \[1, 1000\] | Axiomatic strength. Below 50 \= Incubation; 50-1000 \= Axiomatic. Governs decay tier. |
| S\_overflow | Integer counter | \[0, S\_max\] | Contextual salience score. Active only when S\_base \= 1000\. Bounded by soft ceiling S\_max. |

## **3.3  Saliency-Weighted Hashing**

To avoid combinatorial explosion, the Reasoning Head — not System 1 — identifies the k most semantically dense tokens in the context window C. These tokens (nouns, specific verbs, named entities) become the Semantic Anchors C\_salient. Only these anchors are permuted and hashed as hyperedge keys.

This resolves the bootstrap paradox from earlier versions: asking a fact-blind backbone to identify factually salient tokens is incoherent. The Reasoning Head operates in the frozen semantic space where entities are fully represented, making it the correct component for this task.

### **Semantic Anchor Filter Logic**

* Project all tokens in C into the frozen embedding space R^d.

* Compute the norm of each token vector. High-norm vectors carry more semantic density.

* Filter out tokens within a cosine distance threshold of known syntactic centroids (prepositions, conjunctions, determiners, auxiliary verbs).

* Select the top k remaining tokens by norm as C\_salient.

*Fallback Rule (V5.0 Addition): If fewer than k anchors are identified (e.g., highly abstract or pronoun-heavy text), the system does not attempt a partial hash. Instead, System 2 retrieval is skipped for that token step, and System 1 generates unassisted. This is logged as a 'semantic vacuum' event for diagnostic purposes.*

# **IV. The Reasoning Head — Control Logic & Synchronizer**

The Reasoning Head is a non-generative controller. It does not produce tokens. It orchestrates the interaction between System 1 and System 2 by performing four sequential operations at each token step: Saliency Filtering, Consistency Gating, Conflict Scanning, and Injection Coordination.

## **4.1  Phase 1 — Semantic Saliency Filter**

Described in Section III.3. The Reasoning Head projects C into R^d and identifies C\_salient as the semantic anchors for hyperedge key construction.

## **4.2  Phase 2 — Tiered Consistency Gate (τ)**

When a novel token sequence is proposed as a new Hyperedge, the Reasoning Head validates it against existing Axioms using a tiered threshold. The gate is no longer a fixed scalar — it is dynamic and context-sensitive.

| Threshold Tier | τ Value | Applied To | Rationale |
| ----- | ----- | ----- | ----- |
| τ\_axiom | 0.90 – 0.95 | Physical constants, definitions, mathematical facts, historical events | High semantic density required. False positives here are costly. |
| τ\_domain | 0.75 – 0.85 | Scientific claims, technical specifications, legal facts | Moderately strict. Domain knowledge varies in precision. |
| τ\_user | 0.50 – 0.70 | User names, preferences, stylistic habits, personal context | Permissive. Subjective data should accumulate quickly. |

### **Gate Function**

**Gate(H\_new) \= σ( Sim(v\_new, v\_axioms) − τ\_tiered )**

If Gate \> 0.5, H\_new is instantiated in the Incubation tier at S\_base \= 1\. If Gate ≤ 0.5, the pathway is discarded. The sigmoid ensures a soft boundary rather than a hard step function, preventing extreme sensitivity to threshold choice.

*Paraphrase Pass: If H\_new fails the gate but lands within a cosine distance radius of 0.15 from an existing Tier 2 Axiom without contradicting it, it is instantiated at S\_base \= 25 (mid-Incubation) rather than 1\. This allows stylistic variation of known facts to accumulate faster.*

## **4.3  Phase 3 — Active Conflict & Inversion Detection**

Before any hyperedge retrieval, the Reasoning Head performs a Conflict Scan against existing Axioms (S\_base ≥ 50). This is the mechanism that prevents contradictory beliefs from coexisting at high strength.

### **Conflict Score**

**Conflict(v\_new, v\_old) \= cos\_sim(v\_new, v\_old)**

### **Inversion Decay Rule (V5.0 — with Dead Zone)**

**If Conflict \< −δ:   S\_base\_old \= S\_base\_old × (1 \+ Conflict)**

**If −δ ≤ Conflict ≤ 0:  No penalty applied  \[Dead Zone\]**

**δ \= 0.25  (calibrated threshold — adjustable per embedding space)**

| Why the Dead Zone Is Critical |
| :---- |
| In high-dimensional embedding spaces, two semantically unrelated concepts will often score mildly negative cosine similarity (typically −0.05 to −0.20) due to geometric noise, not genuine opposition. Without the dead zone, the inversion decay would silently erode valid Axioms whenever any new unrelated fact is processed. The dead zone of ±0.25 ensures only genuine semantic inversions (Conflict \< −0.25) trigger the penalty. |

### **Logic Duel Suppression (V5.0 Addition)**

While an H\_old is under active inversion penalty and its S\_base is falling, its injection weight is proportionally suppressed. A hyperedge at 30% of its original strength injects at 30% of its normal α weight — not 30% of the maximum possible weight.

**α\_effective \= α\_normal × (S\_base\_current / S\_base\_original)**

This prevents a half-eroded false belief from still producing meaningful output during the duel period, which was a silent consistency failure in V4.0.

## **4.4  Phase 4 — Injection Coordination**

After retrieval and conflict scanning, the Reasoning Head computes the Sigmoid-Ramped Influence Scaler α and dispatches the injection to the W\_proj bridge.

**α(S\_base) \= σ( (S\_base − 50\) / 10 )**

This ramp ensures that Incubation-tier hyperedges (S\_base \< 50\) have near-zero injection influence, Axiomatic-tier entries (S\_base \= 50–1000) ramp smoothly from weak to strong influence, and Overflow additions provide logarithmically scaled supplemental weight.

**Dynamic α Modulation:** α is further modulated by the syntactic role of the current generation position. When System 1 is generating a function word (preposition, conjunction, determiner), α is suppressed by a factor of 0.2 regardless of S\_base. Factual injection at structurally critical grammatical positions destroys fluency. Function word generation is System 1's exclusive domain.

# **V. The Injection Bridge — W\_proj**

## **5.1  The Dimensionality Problem**

System 2 stores v\_target vectors in the Reasoning Head's frozen hyperspherical embedding space R^(d\_sys2). System 1 operates in its own internal residual stream space R^(d\_sys1). These spaces have different dimensionalities, different geometric structures, and were initialized from different training objectives. Direct addition of vectors from these spaces is mathematically undefined — it would be equivalent to adding meters to kilograms.

## **5.2  The W\_proj Solution**

**W\_proj ∈ R^(d\_sys1 × d\_sys2)**

### **The Full Injection Formula**

**h\_t' \= h\_t  \+  α · W\_proj · v\_target**

**Where:**  h\_t \= System 1 hidden state before FFN  |  α \= Sigmoid-Ramped Influence Scaler  |  W\_proj · v\_target \= v\_target projected into System 1's geometric space  |  h\_t' \= modified hidden state passed to System 1 FFN

## **5.3  Training Procedure (V5.0 — Resolved Joint Training Problem)**

W\_proj cannot be trained jointly with System 1 from scratch without causing System 1 to drift toward factual recall to compensate for early projection errors. The correct training sequence is:

| Phase | What Trains | What is Frozen | Objective |
| ----- | ----- | ----- | ----- |
| Phase 1: EMLM Pretraining | System 1 (LLSU) fully | Nothing (fresh init) | Next-token prediction on entity-masked corpus. System 1 converges on syntactic structure. |
| Phase 2: Projection Alignment | W\_proj \+ lightweight LoRA adapters on System 1 | System 1 core weights | Minimize reconstruction loss: W\_proj · v\_sys2 should map to the region of System 1's residual stream that produces the target token. |
| Phase 3: End-to-End Fine-Tuning | W\_proj \+ LoRA adapters only | System 1 core, System 2, Reasoning Head | Task-specific alignment. Ensures injection feels natural in System 1's generative flow. |

*The LoRA adapters in Phase 2 are critical. They allow System 1 to make minor geometric adjustments to be receptive to injections without corrupting the EMLM-trained syntactic structure. Adapter rank r \= 16 is the recommended starting point. These adapters add negligible parameters (\~0.1% of System 1's weight count).*

# **VI. Tiered Memory & Decay Engine**

## **6.1  Memory Tier Overview**

The Decay Engine implements a 'survival of the fittest' lifecycle for Hyperedges. Facts that are consistently confirmed accumulate strength; facts that are contradicted or ignored erode. The tier structure enforces different decay regimes appropriate to the epistemic status of each knowledge class.

| Tier | S\_base Range | Epistemic Status | Decay Rule | Deletion Trigger |
| ----- | ----- | ----- | ----- | ----- |
| Incubation | 1 – 49 | Unverified hypothesis | Zero decay. Protected from noise. | Never (auto-promotes or lingers) |
| Axiomatic | 50 – 1000 | Verified factual pathway | 1-in-5: −1 point per 5 missed contexts | S\_base → 0 (sustained contradiction) |
| Overflow | 1000 \+ n | Contextual salience / preference | 5-20 rule: \+20 on hit, −20 on miss | S\_overflow → 0 (not S\_base deletion) |

## **6.2  Tier 1 — Incubation (S\_base: 1–49)**

* Status: Unverified hypothesis. The Reasoning Head accepted the pathway but it has not been independently confirmed.

* Influence: Near-zero. The sigmoid ramp α(S\_base) produces α ≈ 0.007 at S\_base \= 1, rising to α ≈ 0.12 at S\_base \= 49\. Incubation entries whisper rather than speak.

* Decay: Zero. Rare or niche facts would be destroyed by noise decay before they could accumulate confirming evidence. Protection is essential for long-tail vocabulary and specialized domains.

* Promotion: Each confirmed generation (hit) adds \+1 to S\_base. At S\_base \= 50, the entry automatically promotes to Axiomatic.

## **6.3  Tier 2 — Axiomatic (S\_base: 50–1000)**

* Status: Established factual pathway. Confidence has been confirmed across multiple independent contexts.

* Influence: High. α ranges from \~0.62 (S\_base \= 50\) to \~0.99 (S\_base \= 1000).

* Decay Rule (1-in-5): The decay engine tracks a miss counter per hyperedge. A 'miss' is defined as: the context C\_salient was present in the input, but the target token v\_target was NOT the generated output.

**If miss\_count mod 5 \== 0:  S\_base \-= 1**

* Deletion: If S\_base reaches 0 through sustained contradiction, the hyperedge is permanently purged from the hash map and logged as a 'refuted pathway.'

## **6.4  Tier 3 — Overflow (S\_overflow: 0 → S\_max)**

* Status: Contextual salience or user preference. S\_base has reached the axiomatic ceiling of 1000; further confidence is tracked in the separate overflow counter.

* Influence: Logarithmically scaled addition to the base weight:

**W\_total \= W\_base \+ β · ln(1 \+ S\_overflow)**

* Decay Rule (5-20): Aggressive bidirectional adjustment.

**Hit:  S\_overflow \+= 20**

**Miss: S\_overflow \= max(0, S\_overflow − 20\)**

| V5.0 Fix: Overflow Ceiling S\_max |
| ----- |
| Problem: S\_overflow was previously unbounded (\[0, ∞)). Frequent access over long sessions could accumulate thousands of overflow points, causing that pathway to permanently dominate generation through frequency alone — defeating the axiomatic tier hierarchy. Solution: S\_overflow is now bounded by a soft ceiling S\_max. Recommended default: S\_max \= 2000 (approximately 100 consecutive hits from zero). The logarithmic dampening means the practical influence difference between S\_overflow \= 1000 and S\_overflow \= 2000 is small — the ceiling prevents runaway accumulation without creating a hard cliff. **S\_overflow \= min(S\_overflow \+ 20, S\_max)** |

# **VII. Full Architecture Diagram — The Generative Loop**

The following diagram traces the complete data flow for a single token generation step in ARNL V1.0, from input ingestion through to the final generated token and post-generation reinforcement.

| ARNL V1.0 — Token Generation Data Flow |
| :---: |

| 1 | INPUT INGESTION User context C \= {t₁, t₂, ... tₖ} is received. Tokenized and passed simultaneously to System 1 and the Reasoning Head. System 1 begins processing syntax; the Reasoning Head begins semantic analysis. |
| :---: | :---- |

▼

| 2 | SEMANTIC SALIENCY FILTER  \[Reasoning Head\] The Reasoning Head projects all tokens in C into the frozen hyperspherical embedding space R^d. It computes token vector norms and filters out syntactic/functional tokens (prepositions, conjunctions, determiners) by cosine proximity to known syntactic centroids. The top k remaining high-norm tokens become C\_salient — the Semantic Anchors. FALLBACK: If fewer than k anchors found, System 2 retrieval is skipped this step. |
| :---: | :---- |

▼

| 3 | HYPEREDGE RETRIEVAL  \[System 2\] The Reasoning Head hashes C\_salient using the content-addressable hash function. O(1) lookup in the Hyper-Adjacency Map. Returns the best-matching Hyperedge H\_best \= { v\_target, S\_base, S\_overflow } if one exists. If no matching hyperedge exists, the injection phase is skipped and System 1 generates unassisted. |
| :---: | :---- |

▼

| 4 | CONFLICT SCAN  \[Reasoning Head\] Before injecting, the Reasoning Head computes Conflict \= cos\_sim(v\_target, v\_existing\_axioms). DEAD ZONE: If −0.25 ≤ Conflict ≤ 0 — no action (geometric noise, not genuine contradiction). INVERSION: If Conflict \< −0.25 — trigger Inversion Decay on the contradicted Axiom: S\_base\_old × (1 \+ Conflict). DUEL SUPPRESSION: If H\_old is under active decay, its injection weight is proportionally reduced. |
| :---: | :---- |

▼

| 5 | SYSTEM 1 FORWARD PASS  \[LLSU\] System 1 processes context C through its GLA/SSM layers, computing the recurrent hidden state h\_t. This state encodes the syntactic expectation — what grammatical shape the next token should have. System 1 is 'hungry' for a specific part of speech but has no opinion on which specific word fills that slot. |
| :---: | :---- |

▼

| 6 | SIGMOID RAMP COMPUTATION  \[Reasoning Head\] α(S\_base) \= σ((S\_base − 50\) / 10). This yields near-zero α for Incubation entries and strong α for Axiomatic entries. FUNCTION WORD SUPPRESSION: If System 1's current position predicts a function word, α is multiplied by 0.2. Factual injection does not override grammatical structure words. |
| :---: | :---- |

▼

| 7 | LATENT INJECTION  \[W\_proj Bridge\] The Injection Bridge computes: h\_t' \= h\_t \+ α · W\_proj · v\_target. W\_proj translates v\_target from the Reasoning Head's semantic space R^(d\_sys2) into System 1's residual stream space R^(d\_sys1). The result h\_t' is a hidden state that carries both syntactic expectation (from h\_t) and semantic intent (from the injected vector). h\_t' is passed to System 1's output FFN. |
| :---: | :---- |

▼

| 8 | TOKEN GENERATION  \[System 1 FFN \+ Softmax\] System 1's output FFN processes h\_t' and produces final logits. Softmax sampling occurs normally. The injected bias has shifted the probability distribution toward v\_target without hard-overriding it — System 1's syntactic understanding 'smooths' the transition. The model generates t\_next. |
| :---: | :---- |

▼

| 9 | POST-GENERATION REINFORCEMENT  \[Decay Engine\] HIT (t\_next matches v\_target): S\_overflow \+= 20 (capped at S\_max). Miss counter reset. MISS (context was present but t\_next ≠ v\_target): Miss counter incremented. If miss\_count mod 5 \== 0: S\_base −= 1\. If S\_overflow \> 0: S\_overflow −= 20\. DELETION CHECK: If S\_base reaches 0, hyperedge is purged and logged. |
| :---: | :---- |

▼

| 10 | AUTOREGRESSIVE APPEND t\_next is appended to C. The loop returns to Step 1\. This continues until the \<END\> token is generated or maximum sequence length is reached. The Hyperedge Map state persists across the entire generation sequence, maintaining factual consistency throughout long outputs. |
| :---: | :---- |

# **VIII. Component Interaction Reference**

| Component | Inputs | Outputs | Mathematical Interface | Trainable? |
| ----- | ----- | ----- | ----- | ----- |
| System 1 (LLSU) | x\_t, h\_{t-1}, h\_t' (injected) | h\_t, logits, t\_next | h\_t \= A\_t ⊙ h\_{t-1} \+ B\_t ⊙ x\_t | Yes (EMLM \+ LoRA adapters) |
| Reasoning Head — Saliency | Context C, frozen embeddings | C\_salient (anchor tokens) | High-pass norm filter on R^d | No (rule-based \+ frozen space) |
| Reasoning Head — Gate | v\_new, existing Axioms | Gate score, Tier assignment | σ(Sim(v\_new, Axioms) − τ\_tiered) | No (threshold-based) |
| Reasoning Head — Conflict | v\_new, v\_old pairs | Inversion penalties, duel flags | S\_old × (1 \+ Conflict) if Conflict \< −δ | No (arithmetic) |
| System 2 (Hyper-Adjacency Map) | Hash(C\_salient) | H\_best \= {v\_target, S\_base, S\_overflow} | O(1) hash lookup | No (counter-based) |
| W\_proj Bridge | v\_target ∈ R^(d\_sys2), α | Bias vector ∈ R^(d\_sys1) | α · W\_proj · v\_target | Yes (alignment phase only) |
| Decay Engine | Hit/miss signals, S values | Updated S\_base, S\_overflow, deletions | Arithmetic decay rules | No (deterministic) |

# **IX. Worked Example — End-to-End Token Generation**

Input: "The capital of France is \_\_\_"

Assumed Hyperedge Map state: H \= { Hash({capital, France}) → v\_Paris, S\_base \= 1000, S\_overflow \= 150 }

| Step | Action | State / Result |
| ----- | ----- | ----- |
| 1 | Input ingested | C \= { "The", "capital", "of", "France", "is" } |
| 2 | Saliency filter | C\_salient \= { "capital", "France" }. "The", "of", "is" filtered as syntactic centroids. |
| 3 | Hyperedge retrieval | Hash({capital, France}) → H\_best found. v\_target \= v\_Paris. S\_base \= 1000, S\_overflow \= 150\. |
| 4 | Conflict scan | No existing Axioms conflict with v\_Paris. Conflict scores all \> −0.25. Dead zone. No penalties. |
| 5 | System 1 forward pass | h\_t computed. System 1 predicts a high-entropy noun distribution. 'Paris' may appear but not dominantly — EMLM trained System 1 does not strongly recall it. |
| 6 | α computation | α(1000) \= σ((1000 − 50)/10) \= σ(95) ≈ 0.999. Strong injection. Position is a content noun slot — no function word suppression. |
| 7 | Latent injection | h\_t' \= h\_t \+ 0.999 · W\_proj · v\_Paris. System 1's hidden state is now strongly biased toward the semantic content of 'Paris' expressed in its own geometric space. |
| 8 | Token generation | System 1 FFN processes h\_t'. Output distribution peaks sharply on 'Paris'. Token sampled: 'Paris'. |
| 9 | Reinforcement (HIT) | t\_next \= 'Paris' matches v\_target. S\_overflow \= min(150 \+ 20, 2000\) \= 170\. Miss counter reset. |
| 10 | Append \+ loop | "Paris" appended to C. Next step begins with C \= { "The", "capital", "of", "France", "is", "Paris" }. |

# **X. Granular Semantic Auditability & Hard Lock**

## **10.1  Total Belief Transparency**

Unlike a Transformer where factual recall is distributed across billions of float16 weights with no interpretable structure, every factual assertion ARNL makes can be traced to a specific Hyperedge entry in System 2\. The hyperedge record exposes:

* The exact context hash that triggered retrieval.

* The target token and its current axiomatic strength S\_base.

* The full decay history — how many hits and misses have occurred.

* The tier status and when the entry was created and last accessed.

An administrator can print the entire belief state of an ARNL model as a human-readable table. This is impossible with any standard Transformer architecture.

## **10.2  Manual Intervention**

| Operation | Mechanism | Effect |
| ----- | ----- | ----- |
| Hard Delete | Remove H by key from hash map | Model permanently loses that factual pathway. System 1 fills the gap probabilistically. |
| Hard Lock | Set S\_base \= 1000, disable all decay flags | Model cannot be argued out of this fact. Inversion decay is suppressed. Used for ground-truth constraints. |
| Hard Override | Set S\_base \= 0, trigger deletion on next step | Equivalent to controlled refutation. Faster than waiting for decay erosion. |
| Tier Freeze | Disable all decay for a specific tier | Used during testing or domain-specific deployment where decay should not apply. |

*Security Note: The persistent Hyperedge Map accumulates user-specific data in the τ\_user tier across sessions. In multi-user or production deployments, maps must be scoped per user and subject to data retention policies. Administrators should implement periodic τ\_user tier audits and provide user-facing map inspection and deletion interfaces. However, only local models will be produced officially.*

# **XI. Competitive Positioning vs. Standard Transformers**

ARNL V1.0 is not a better Transformer — it is a different paradigm optimized for a different point in the capability-reliability tradeoff space. The following table characterizes the performance profile honestly.

| Capability | Standard Transformer SLM | ARNL V1.0 | Winner |
| ----- | ----- | ----- | ----- |
| Open-domain generative fluency | High — benefits from full factual+syntactic co-training | Competitive — EMLM \+ injection achieves comparable fluency in-domain | Transformer (slight) |
| Factual precision (bounded domain) | Moderate — probabilistic recall degrades under distribution shift | High — axiomatic gating with discrete counters resists degradation | ARNL |
| Continual learning (no forgetting) | Poor — fine-tuning overwrites prior weights | Excellent — Hyperedge Map is append-and-decay, not gradient-overwrite | ARNL |
| Auditability | None — weights are opaque float arrays | Complete — every belief is a readable, modifiable record | ARNL |
| Hard behavioral constraints | None — cannot guarantee a specific output | Yes — Hard Lock creates deterministic output for specific inputs | ARNL |
| Inference latency | Single forward pass, well-optimized | Additional Reasoning Head \+ hash lookup overhead (\~15–20% latency increase) | Transformer |
| Open-domain creative generation | High | Moderate — fact-blindness limits unconstrained creative range | Transformer |
| Personalization without fine-tuning | None — requires expensive retraining | Yes — τ\_user tier adapts in real-time | ARNL |

| Recommended Deployment Profile |
| :---- |
| ARNL V1.0 is the superior architecture for: medical decision support, legal citation and document analysis, scientific question answering, regulated enterprise assistants, and any application where factual errors carry real-world consequences and auditability is required. Standard Transformers remain preferable for: open-ended creative writing, broad conversational assistants with no domain constraint, and applications where maximum generative range is the primary objective. |

# **XII. Known Limitations & Open Research Questions**

| Limitation | Current Status | Mitigation / Future Work |
| ----- | ----- | ----- |
| Multi-hop reasoning | ARNL retrieves individual facts; it does not chain them symbolically across multiple inference steps. | Future: Introduce a Reasoning Chain register that accumulates sequential hyperedge retrievals and checks their joint consistency before injection. |
| Cold start knowledge | New deployments have empty or sparse Hyperedge Maps. Until τ\_user entries accumulate, System 1 carries full generative load. | Mitigation: Pre-populate maps with domain-specific Axioms at S\_base \= 1000 using curated knowledge bases before deployment. |
| Embedding space drift | W\_proj is trained against a frozen semantic space. If the frozen embedding model is updated, W\_proj becomes misaligned. | Mitigation: Lazy re-projection — periodic W\_proj re-alignment triggered when embedding space version changes, without full System 1 retraining. |
| τ calibration sensitivity | The tiered τ values (0.90, 0.80, 0.65) require empirical calibration per embedding space. Wrong τ causes either over-rejection or over-acceptance. | Future: Adaptive τ that calibrates automatically using a small validation set during deployment initialization. |
| Semantic vacuum events | When fewer than k anchors are identified, System 2 is skipped. Frequency of this event in target domain is unknown. | Mitigation: Log all semantic vacuum events during testing. If frequency exceeds 15% of tokens, reduce k or adjust the syntactic centroid filter radius. |

# **Appendix A — Mathematical Symbol Reference**

| Symbol | Definition |
| ----- | ----- |
| C | Full context window — all tokens in current input |
| C\_salient | Semantic Anchor subset — top k factually dense tokens identified by Reasoning Head |
| h\_t | System 1 hidden state at token step t (pre-injection) |
| h\_t' | System 1 hidden state at token step t (post-injection) |
| A\_t | Selective forget gate in GLA/SSM (data-dependent) |
| B\_t | Input gate in GLA/SSM (data-dependent) |
| v\_target | Target token embedding vector in frozen hyperspherical space R^(d\_sys2) |
| S\_base | Axiomatic strength counter \[1, 1000\] |
| S\_overflow | Contextual salience counter \[0, S\_max\] |
| S\_max | Overflow ceiling (default: 2000\) |
| W\_proj | Learned projection matrix R^(d\_sys1 × d\_sys2) — the Injection Bridge |
| α | Sigmoid-Ramped Influence Scaler \= σ((S\_base − 50\) / 10\) |
| τ\_tiered | Dynamic consistency threshold (0.90 axiom / 0.80 domain / 0.65 user) |
| δ | Inversion dead zone threshold (default: 0.25) |
| β | Logarithmic overflow scaling coefficient |

# **Appendix B — V1.0 Change Log vs. V0.9**

| Change | Problem Solved | Location in Spec |
| ----- | ----- | ----- |
| Dead zone δ \= 0.25 added to inversion decay | Geometric noise in high-dim spaces was silently eroding valid Axioms | Section IV.3 |
| Overflow ceiling S\_max \= 2000 | Unbounded S\_overflow caused frequency-based dominance over axiomatic hierarchy | Section VI.4 |
| W\_proj three-phase training procedure | Joint training caused System 1 drift toward factual recall, undermining EMLM | Section V.3 |
| Semantic vacuum fallback rule | Undefined behavior when fewer than k anchors found in abstract text | Section III.3 |
| Logic Duel suppression (α\_effective scaling) | Half-eroded false beliefs still injected at normal weight during duel period | Section IV.3 |
| Data retention security note | User-scoped τ\_user data requires privacy controls in production | Section X.2 |
| Paraphrase pass at gate (S\_base \= 25 init) | Stylistic variants of known facts were blocked, limiting expressive range | Section IV.2 |

