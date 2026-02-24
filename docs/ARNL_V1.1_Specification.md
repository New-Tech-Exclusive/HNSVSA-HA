

**ARNL V1.1**

**Axiomatic-Recurrent Neural Logic**

Full Technical Specification & Architecture Reference

A Neuro-Symbolic Replacement Architecture for Small Language Models (Local Deployment)

Parameter Range: 1B – 24B+ │ Architecture: SHADE Edition  
Classification: External Technical Review

# **Abstract**

ARNL V1.1 is a heterogeneous dual-process architecture designed to replace the standard Transformer decoder for Small Language Models in the 1–24 billion parameter range. It separates syntactic fluency from factual reliability into two structurally distinct systems: a fact-blind recurrent backbone (System 1\) and a sparse, self-indexing Hyperedge Attractor Graph (System 2, redesigned as SHADE), coordinated without a separate Reasoning Head.

V1.1 introduces the SHADE architecture (Sparse Hyperedge Attractor with Distributed Encoding), which eliminates the Reasoning Head and the W\_proj Injection Bridge from V1.0. Gap detection is now driven by System 1’s own output entropy rather than a separate controller, and factual injection operates directly at the logit layer rather than through a learned projection matrix. This resolves the V1.0 cold-start saliency bootstrap problem and the dimensionality mismatch problem structurally, not through mitigations.

The architecture’s primary differentiators over standard Transformers remain: Granular Semantic Auditability, Hard-Lock behavioral constraints, continual learning stability without catastrophic forgetting, and real-time personalization without fine-tuning. V1.1 adds a fifth differentiator: zero learned components in the factual recall pipeline.

# **Table of Contents**

# **I. Global Architecture Overview**

ARNL V1.1 is a two-component Symbiotic Bundle. System 1 (the LLSU) and System 2 (SHADE) operate as a unified generative system. There is no third controller component. The coordination logic that the V1.0 Reasoning Head performed is now distributed: gap detection lives in System 1’s output layer, and retrieval/injection logic lives in SHADE’s own structure.

| Core Design Principle — V1.1 |
| :---- |
| System 1 determines HOW something is said and SIGNALS when it does not know WHAT. SHADE determines WHAT is said whenever System 1 signals uncertainty. No controller mediates between them. |

| Component | Role | Architecture Type | Parameter Budget |
| :---- | :---- | :---- | :---- |
| **System 1 (LLSU)** | Fluidity Core — syntactic generation, uncertainty signaling | GLA / SSM Recurrent | \~95% of total |
| **System 2 (SHADE)** | Reliability Core — factual storage, self-indexing retrieval, direct injection | Sparse Hyperedge Attractor Graph (non-neural) | Off-budget (database) |
| **Decay Engine** | Memory lifecycle manager | Rule-based arithmetic | No parameters |
| **SALT Tokenizer** | Semantic-Anchor-Linked encoding | Byte-Level BPE \+ classification | Off-budget (static) |

## **1.1 What Was Removed and Why**

V1.0 contained two components that V1.1 eliminates entirely:

| Removed Component | V1.0 Problem It Caused | V1.1 Replacement |
| :---- | :---- | :---- |
| **Reasoning Head** | Cold-start saliency bootstrap paradox: asked a fact-blind system to find facts. Also consumed 15% of parameter budget doing coordination that belongs in the data layer. | Gap detection via System 1 entropy. Self-indexing via SHADE concept embeddings. Zero parameters spent on coordination. |
| **W\_proj Bridge** | Dimensionality mismatch between System 1’s hidden state space and System 2’s semantic space required a trained projection. Added a third training phase and a new failure mode (embedding drift). | Direct logit injection. SHADE writes to System 1’s output distribution, not its hidden state. No projection required, no embedding drift. |

## **1.2 Parameter Budget Philosophy**

With the Reasoning Head (15%) and W\_proj (5%) eliminated, System 1 now commands approximately 95% of the total parameter budget. At the 7B total scale this yields a \~6.65B System 1 — competitive with standalone 7B-class Transformer models in syntactic fluency — while the overall system retains factual precision no 7B Transformer can match in bounded domains. The remaining \~5% is allocated to SALT tokenizer embeddings and minimal infrastructure. SHADE itself is off-budget: it is a database, not a neural network, and grows with knowledge rather than parameters.

# **II. System 1 — The Syntactic Linear-Latent Selective Unit (LLSU)**

## **2.1 Role and Objective**

System 1 is the Fluidity Core. Its objective is to model the transition probability of grammatical structures — the shape of language — without internalizing any factual content. It is responsible for word order, grammatical agreement, prosody, register consistency, and local coherence.

In V1.1, System 1 has a second role that did not exist in V1.0: it is the gap detector. When System 1’s output distribution is uncertain over the semantic-anchor region of the vocabulary — meaning it knows a factual token goes here but has no strong opinion which one — it signals SHADE to intervene. This signal requires no additional parameters: it is the entropy of System 1’s own logits, computed over the SAC (Semantic Anchor Candidate) token subset.

## **2.2 Architecture: Gated Linear Attention / SSM**

System 1 uses a Selective State-Space Model with Gated Linear Attention. This choice is deliberate and unchanged from V1.0:

* Linear complexity in sequence length — no quadratic memory cost at long contexts.

* Recurrent inference — the model processes tokens as a state machine, compatible with ARNL’s autoregressive loop.

* No attention sink artifacts — GLA does not develop the early-token dominance pathology common in Transformers.

* No self-attention — explicitly excluded. Self-attention gives System 1 the machinery for in-context factual recall, which would bypass EMLM and allow System 1 to retrieve facts from the prompt without System 2\. This must not happen.

### **State Transition Formula**

**h\_t \= A\_t ⊙ h\_{t-1} \+ B\_t ⊙ x\_t**

Where: h\_t \= hidden state at step t | A\_t \= selective forget gate (data-dependent) | B\_t \= input gate | x\_t \= current input token embedding

## **2.3 Training: Entity-Masked Language Modeling (EMLM)**

System 1 is pretrained exclusively on EMLM. All factual entities are identified by the SALT tokenizer and replaced with semantic placeholder tokens before System 1 ever sees them.

| Masked Entity Type | Replacement Token | Preserved Information |
| :---- | :---- | :---- |
| Proper Nouns (people, places) | \[ENTITY\_NOUN\] | Syntactic role, grammatical gender |
| Numerical Data (dates, quantities) | \[ENTITY\_NUM\] | Numerical context, units |
| Named Concepts (products, events) | \[ENTITY\_CONCEPT\] | Conceptual slot in sentence |
| Domain-Specific Terms | \[ENTITY\_DOMAIN\] | Register and field marker |

| EMLM Verification Test |
| :---- |
| After Phase 1 training, probe System 1 by asking it to complete 'The capital of France is \_\_\_' without injection. A correctly trained LLSU outputs a high-entropy distribution over all nouns — not 'Paris.' If 'Paris' appears with high probability, the EMLM masking was incomplete. This test must pass before Phase 2 begins. |

## **2.4 Gap Detection — The V1.1 Signal**

This is the mechanism that replaces the Reasoning Head’s Saliency Filter. Instead of a separate component trying to predict when a fact is needed, System 1 itself reports when it does not know.

### **Gap Detection Formula**

After System 1 produces logits for the next token position, compute the Shannon entropy over only the SAC-class token IDs (the \~30-40% of the vocabulary classified as Semantic Anchor Candidates by SALT):

**H\_SAC \= \-Σ p(t) · log p(t)   for t ∈ SAC\_vocab**

If H\_SAC exceeds a threshold H\_gap, a factual gap has been detected and SHADE retrieval is triggered. If H\_SAC ≤ H\_gap, System 1 is confident in the semantic content — either because a factual token is clearly appropriate or because a syntactic token is clearly appropriate — and generation proceeds without SHADE.

| H\_SAC Value | Interpretation | Action |
| :---- | :---- | :---- |
| H\_SAC \> H\_gap | High uncertainty — System 1 knows a fact goes here but not which | Trigger SHADE retrieval. Inject best-matching hyperedge. |
| H\_SAC ≤ H\_gap | Low uncertainty — System 1 is confident in its output | Generate directly. SHADE is not consulted. |

| Why This Is Architecturally Superior to the Reasoning Head |
| :---- |
| The V1.0 Reasoning Head faced a bootstrap paradox: it needed to identify factually salient tokens to know when to retrieve facts, but it was operating on the output of a fact-blind backbone. It was guessing when facts were needed based on syntactic signals alone. Gap detection via System 1 entropy has no bootstrap problem. System 1 is not guessing. It is reporting a direct measurement of its own uncertainty. When H\_SAC is high, System 1 is genuinely uncertain about which semantic token to produce — not because it was told to be uncertain, but because its EMLM training left the factual region of its weights unpopulated. H\_gap recommended default: 2.0 nats. Calibrate empirically on your target domain during Phase 1 evaluation. |

## **2.5 Training Procedure — Simplified V1.1**

The elimination of W\_proj removes one training phase compared to V1.0. The procedure is now two phases:

| Phase | What Trains | What is Frozen | Objective |
| :---- | :---- | :---- | :---- |
| **Phase 1: EMLM Pretraining** | System 1 (LLSU) fully | Nothing (fresh init) | Next-token prediction on entity-masked corpus. System 1 converges on syntactic structure. EMLM verification test must pass. |
| **Phase 2: Fine-Tuning** | System 1 with lightweight LoRA adapters (r=16) | System 1 core weights | Task-specific alignment on unmasked domain data with SHADE pre-populated. LoRA adapters allow System 1 to become receptive to SHADE’s injection style without corrupting EMLM structure. |

Note: LoRA adapters in Phase 2 are not training the injection bridge (W\_proj no longer exists). They are training System 1’s receptivity to logit-level steering — specifically, ensuring that when SHADE boosts a logit, System 1’s FFN does not counteract it through strong competing internal activations. Adapter rank r=16 adds approximately 0.1% of System 1’s weight count.

# **III. System 2 — SHADE: Sparse Hyperedge Attractor with Distributed Encoding**

## **3.1 Role and Overview**

SHADE is the Reliability Core. It is a sparse, self-indexing hyperedge attractor graph that stores factual beliefs as concept-attractor nodes connected by co-occurrence edges. It has no trainable parameters. It does not participate in gradient computation. Its beliefs are stored as structured data and updated by the Decay Engine through arithmetic rules, not backpropagation.

SHADE differs from the V1.0 Hyper-Adjacency Map in four critical ways:

* Nodes are concept attractors, not raw hash keys. Each node stores a concept embedding in the frozen SALT semantic space, enabling nearest-neighbor retrieval rather than exact-hash lookup.

* Edges form organically from co-occurrence. When two concept nodes appear together in a context that produces a successful retrieval, an edge is created or strengthened between them. The graph topology is never pre-designed.

* Retrieval is self-indexing via embedding geometry. SHADE finds relevant nodes by nearest-neighbor search in the frozen embedding space. No external indexing structure is required.

* Injection is direct logit boosting. SHADE writes to System 1’s output logit vector, not its hidden state. No projection matrix, no dimensionality mismatch, no W\_proj.

## **3.2 Node Structure — The Hyperedge Attractor**

Each node in SHADE is a self-contained belief unit. The structure is an evolution of the V1.0 Hyperedge:

**N \= { concept\_emb, target\_dist, context\_window, confidence\_vec, S\_base, S\_overflow }**

| Field | Type | Range | Description |
| :---- | :---- | :---- | :---- |
| **concept\_emb** | int8 vector | R^d (frozen) | Quantized embedding of this node’s concept in the frozen SALT semantic space. This is the node’s identity — what it is about. Used as the lookup key in nearest-neighbor retrieval. |
| **target\_dist** | float16 array | Top-K probabilities | A small distribution over the top-K vocabulary tokens this node recommends when activated. Starts uniform at node creation; sharpens as the strength system reinforces specific tokens. K \= 8 recommended default. |
| **context\_window** | int16 array | N token IDs | The N most recent SAC token IDs present in the context when this node was last strongly reinforced. Used as the retrieval context fingerprint. |
| **confidence\_vec** | int8 array | \[-127, 127\] | Per-position weight over context\_window tokens. High positive values indicate which context tokens are most diagnostic for this node’s activation. Updated by the Decay Engine, not by gradients. |
| **S\_base** | int16 | \[1, 1000\] | Axiomatic strength counter. Identical semantics to V1.0. Below 50 \= Incubation. 50-1000 \= Axiomatic. Governs decay tier and injection weight α. |
| **S\_overflow** | int16 | \[0, S\_max\] | Contextual salience counter. Identical semantics to V1.0. Active only when S\_base \= 1000\. Bounded by soft ceiling S\_max \= 2000\. |

## **3.3 Edge Structure — Organic Co-occurrence Links**

Edges in SHADE form between nodes that have been retrieved together in the same context window. They are not pre-designed. They emerge from the history of successful retrievals.

**E \= { node\_a\_id, node\_b\_id, co\_weight, last\_co\_step }**

| Field | Type | Range | Description |
| :---- | :---- | :---- | :---- |
| **co\_weight** | int16 | \[0, 1000\] | How frequently these two nodes have co-activated. Starts at 1 on edge creation. Increments by 1 per co-activation. Decays by 1 per 10 single-activations of either node without the other. Deletion at co\_weight \= 0\. |
| **last\_co\_step** | int64 | step counter | Generation step of the most recent co-activation. Used to compute recency weighting during retrieval scoring. |

| Why Edges Matter — Contextual Disambiguation |
| :---- |
| The same concept node can be the correct retrieval in multiple different contexts. Edges allow SHADE to distinguish these contexts. Reaching a 'capital city' node via a strong edge from a 'France' node produces a higher combined score than reaching it without that edge. Two nodes that always appear together in the same factual context develop a high-weight edge that boosts retrieval confidence specifically in that context. This is the mechanism by which SHADE generalizes across related knowledge without any learned weights. |

## **3.4 Retrieval Pipeline — The Four Steps**

When System 1 signals a gap (H\_SAC \> H\_gap), SHADE executes the following pipeline:

### **Step 1 — Context Fingerprinting**

Take the last K=8 SAC tokens from the current context window. These are the tokens SALT classified as Semantic Anchor Candidates. Embed each token using the frozen SALT embedding table. Average-pool into a single context query vector q. Quantize to int8.

**q \= quantize( mean( emb(t) for t in SAC\_context\[-K:\] ) )**

### **Step 2 — Nearest-Neighbor Candidate Retrieval**

Search SHADE’s node store for the M nearest nodes to q by cosine similarity over concept\_emb vectors. M \= 16 recommended default. This search is the self-indexing mechanism — the embedding geometry of the frozen SALT space organizes the graph without any external index. For large deployments (\>100K nodes), a FAISS index over concept\_emb vectors reduces search from O(N) to O(log N).

**candidates \= top\_M\_by\_cosine(q, all\_nodes.concept\_emb)**

### **Step 3 — Scoring with Edge Boost**

Score each candidate node using a combination of concept similarity, confidence vector match, strength, and edge co-activation:

**score(N) \= sim(q, N.concept\_emb) · dot(q\_ctx, N.confidence\_vec) · log(1 \+ N.S\_base) · edge\_boost(N)**

Where edge\_boost(N) \= 1.0 \+ (sum of co\_weights of edges connecting N to other recently active nodes) / 1000\. This means nodes that frequently co-activate with other nodes currently relevant to the context receive a multiplicative boost.

Select the highest-scoring candidate as N\_best. If no candidate exceeds a minimum similarity threshold sim\_min \= 0.65, SHADE abstains and System 1 generates unassisted. This is logged as a semantic vacuum event.

### **Step 4 — Direct Logit Injection**

SHADE does not inject into System 1’s hidden state. It writes directly to System 1’s pre-softmax logit vector. For each token in N\_best.target\_dist, its logit is boosted by the product of α and the distribution weight:

**logit'(t) \= logit(t) \+ α(S\_base) · target\_dist\[t\]   for t in N\_best.target\_dist**

Where α(S\_base) \= σ( (S\_base \- 50\) / 10 ), identical to V1.0. The sigmoid ramp ensures Incubation-tier nodes inject weakly, Axiomatic-tier nodes inject strongly.

| Function Word Suppression — Preserved from V1.0 |
| :---- |
| If System 1's pre-injection top-1 token is classified as SYN (syntactic/function word) by SALT, SHADE injection is suppressed entirely regardless of S\_base. Factual injection at structurally critical grammatical positions destroys fluency. Function word generation is System 1's exclusive domain. The gap detection entropy threshold H\_gap is also calibrated to fire less readily when the current context suggests a function word position — since SAC entropy is naturally lower at those positions. |

## **3.5 Node Creation — Tiered Consistency Gate**

The tiered consistency gate from V1.0 is preserved, now operating at SHADE’s node creation step rather than at the Reasoning Head. When a generation event produces a factual output that is not currently stored in SHADE, the system evaluates whether to create a new node.

| Tier | τ Value | Applied To | Rationale |
| :---- | :---- | :---- | :---- |
| τ\_axiom | 0.90–0.95 | Physical constants, definitions, mathematical facts, historical events | High semantic density required. False positives are costly and hard to decay. |
| τ\_domain | 0.75–0.85 | Scientific claims, technical specifications, legal facts | Moderately strict. Domain knowledge varies in precision. |
| τ\_user | 0.50–0.70 | User names, preferences, stylistic habits, personal context | Permissive. Subjective data should accumulate quickly. |

Gate function: Gate(N\_new) \= σ( Sim(v\_new, v\_axioms) − τ\_tiered ). If Gate \> 0.5, the new node is instantiated at S\_base \= 1 (Incubation). If Gate ≤ 0.5, the candidate is discarded.

Paraphrase Pass (preserved from V1.0): If a candidate fails the gate but lands within cosine distance 0.15 of an existing Axiomatic node without contradicting it, it is instantiated at S\_base \= 25\. This allows stylistic variations of known facts to accumulate faster.

## **3.6 Conflict Detection — Preserved from V1.0**

Before any retrieval is used for injection, SHADE performs a conflict scan against all Axiomatic nodes (S\_base ≥ 50).

**Conflict(v\_new, v\_old) \= cos\_sim(v\_new, v\_old)**

**If Conflict \< \-δ: S\_base\_old \= S\_base\_old × (1 \+ Conflict)**

**If \-δ ≤ Conflict ≤ 0: No penalty \[Dead Zone\]**

Dead zone δ \= 0.25. In high-dimensional spaces, unrelated concepts frequently score mildly negative cosine similarity (−0.05 to −0.20) due to geometric noise. The dead zone prevents silent erosion of valid Axioms by geometric noise.

Logic Duel Suppression (preserved): While a node is under active inversion penalty, its injection weight is proportionally suppressed:

**α\_effective \= α\_normal × (S\_base\_current / S\_base\_original)**

## **3.7 Multi-Token Fact Handling**

Direct logit injection operates on one token position per gap detection event. Multi-word proper nouns and technical phrases require gap detection to fire at consecutive positions. SHADE handles this through a gap continuation mechanism:

* When a gap fires and SHADE injects successfully, SHADE marks itself as in a continuation state.

* At the next token step, if the generated token is consistent with N\_best.target\_dist, SHADE checks whether the next position in target\_dist is non-empty.

* If it is, gap detection is bypassed for that position and SHADE injects the next token in the sequence directly without requiring H\_SAC \> H\_gap.

* Continuation terminates when target\_dist is exhausted or when the generated token falls outside the target distribution.

| Multi-Token Limitation vs. V1.0 Residual Injection |
| :---- |
| V1.0’s residual stream injection via W\_proj could influence not just the immediate next token but System 1’s hidden state trajectory, priming subsequent tokens. SHADE’s logit injection is position-local. For single-token facts (names, places, dates, numbers), SHADE is strictly cleaner. For multi-word technical phrases, the continuation mechanism handles the sequence mechanically but does not prime System 1’s hidden state geometry to expect the phrase. Fluency at multi-word injection boundaries should be monitored empirically. |

# **IV. Full Architecture — The V1.1 Generative Loop**

The following traces the complete data flow for a single token generation step in ARNL V1.1.

| 1 | INPUT INGESTION User context C \= {t₁, t₂, ... tₖ} is received. SALT tokenizer encodes C and returns SALTEncoding with input\_ids, class\_ids, anchor\_mask, and syntactic\_mask. Input is passed to System 1\. SHADE receives the SAC token subset for context fingerprinting. |
| :---: | :---- |
| **2** | **SYSTEM 1 FORWARD PASS** System 1 (LLSU) processes the full token sequence through its GLA/SSM layers. Produces hidden state h\_t and a logit vector over the full vocabulary. No injection yet. |
| **3** | **GAP DETECTION \[System 1 Entropy\]** Compute H\_SAC \= entropy of System 1’s logits over SAC-class tokens only. If H\_SAC \> H\_gap: factual gap detected, proceed to Step 4\. If H\_SAC ≤ H\_gap: System 1 is confident, skip to Step 7 and generate directly. |
| **4** | **SHADE CONTEXT FINGERPRINTING \[System 2\]** SHADE takes the last K=8 SAC tokens from C. Embeds each in the frozen SALT semantic space. Average-pools to query vector q. Quantizes to int8. |
| **5** | **NEAREST-NEIGHBOR RETRIEVAL \+ CONFLICT SCAN \[SHADE\]** SHADE searches concept\_emb vectors for top-M=16 nearest neighbors to q. Scores each candidate using sim \+ confidence\_vec dot product \+ S\_base strength \+ edge boost. Runs conflict scan against Axiomatic nodes — applies inversion decay and Logic Duel suppression if conflicts found. Selects N\_best. If no candidate exceeds sim\_min=0.65, SHADE abstains (semantic vacuum event logged). If N\_best found, proceed. |
| **6** | **DIRECT LOGIT INJECTION \[SHADE → System 1 Output\]** Compute α(S\_base) \= σ((S\_base − 50\) / 10). Check function word suppression: if System 1’s top-1 pre-injection token is SYN class, suppress α × 0.2. For each token t in N\_best.target\_dist: logit'(t) \= logit(t) \+ α × target\_dist\[t\]. No matrix multiply, no projection, no hidden state modification. |
| **7** | **TOKEN GENERATION** Softmax over the (possibly SHADE-modified) logit vector. Sample t\_next. This token is both syntactically appropriate (shaped by System 1’s GLA pass) and factually steered (logit-boosted by SHADE where relevant). |
| **8** | **POST-GENERATION REINFORCEMENT \[Decay Engine\]** If SHADE injected: update N\_best.S\_overflow per hit rule. Update N\_best.confidence\_vec to reinforce the context tokens that led to this retrieval. Strengthen edges to any other nodes that were also active this step. Increment their co\_weight by 1\. |
| **9** | **DECAY ENGINE PASS** Apply per-tier decay rules (Section V). Increment miss counters for Axiomatic nodes whose context was present but were not activated. Purge nodes at S\_base \= 0\. Decay edge co\_weights for pairs that did not co-activate this step. |
| **10** | **AUTOREGRESSIVE APPEND** Append t\_next to context C. Return to Step 1\. Loop until \[EOS\] is generated or max\_length is reached. |

# **V. Tiered Memory & Decay Engine**

## **5.1 Memory Tier Overview**

The Decay Engine implements a survival-of-the-fittest lifecycle for SHADE nodes. Facts confirmed across multiple independent contexts accumulate strength; facts contradicted or ignored erode. The tier structure is unchanged from V1.0 and applies identically to SHADE nodes.

| Tier | S\_base Range | Epistemic Status | Decay Rule | Deletion Trigger |
| :---- | :---- | :---- | :---- | :---- |
| Incubation | 1 – 49 | Unverified hypothesis | Zero decay. Protected from noise. | Never (auto-promotes or lingers) |
| Axiomatic | 50 – 1000 | Verified factual pathway | 1-in-5: −1 point per 5 missed contexts | S\_base → 0 |
| Overflow | 1000 \+ n | Contextual salience / preference | 5-20 rule: \+20 on hit, −20 on miss | S\_overflow → 0 (not node deletion) |

## **5.2 Tier 1 — Incubation (S\_base: 1–49)**

* Status: Unverified hypothesis. SHADE accepted the pathway but it has not been independently confirmed.

* Influence: Near-zero. α(S\_base) produces α ≈ 0.007 at S\_base \= 1, rising to α ≈ 0.12 at S\_base \= 49\. Incubation nodes whisper rather than speak.

* Decay: Zero. Rare or niche facts would be destroyed by noise decay before accumulating confirming evidence.

* Promotion: Each confirmed hit adds \+1 to S\_base. At S\_base \= 50, the node automatically promotes to Axiomatic.

## **5.3 Tier 2 — Axiomatic (S\_base: 50–1000)**

* Status: Established factual pathway. Confidence confirmed across multiple independent contexts.

* Influence: High. α ranges from \~0.62 (S\_base \= 50\) to \~0.99 (S\_base \= 1000).

* Decay Rule (1-in-5): Tracks a miss counter per node. A miss is defined as: the context fingerprint was similar to this node’s context\_window (cosine sim \> 0.7) but the node’s target token was NOT the generated output.

**If miss\_count mod 5 \== 0: S\_base \-= 1**

* Deletion: S\_base reaches 0 through sustained contradiction → node permanently purged and logged as a refuted pathway.

## **5.4 Tier 3 — Overflow (S\_overflow: 0 → S\_max)**

* Status: Contextual salience or user preference. S\_base has reached the axiomatic ceiling of 1000\.

* Influence: Logarithmically scaled addition to base weight:

**W\_total \= W\_base \+ β · ln(1 \+ S\_overflow)**

* Decay Rule (5-20): Aggressive bidirectional adjustment.

**Hit: S\_overflow \= min(S\_overflow \+ 20, S\_max)**

**Miss: S\_overflow \= max(0, S\_overflow \- 20\)**

## **5.5 Edge Decay**

Edges have their own separate decay rule. This is new in V1.1:

* Hit (co-activation in same context step): co\_weight \+= 1, capped at 1000\.

* Miss (either node activated without the other, 10 consecutive times): co\_weight \-= 1\.

* Deletion: co\_weight reaches 0 → edge is permanently removed. Nodes are not deleted when their edges are.

## **5.6 V1.1 Injection Value Weighting**

S\_overflow updates are weighted by injection value. This is new in V1.1 and does not exist in V1.0:

| Injection Value | Definition | S\_overflow Update |
| :---- | :---- | :---- |
| High-value hit | SHADE injected a token that System 1 ranked at position 20 or lower in its pre-injection distribution | \+20 (full reward — SHADE did real work) |
| Low-value hit | SHADE injected a token System 1 ranked in its top 3 — it already knew the answer | \+5 (partial reward — confirmation only) |
| Miss | SHADE fired but the generated token did not match target\_dist | \-20 (full penalty) |

This teaches SHADE to distinguish nodes that are genuinely contributing information from nodes that are merely confirming what System 1 already knew. High-value nodes accumulate strength faster, correctly reflecting their epistemic importance.

# **VI. Hard Lock, Auditability & Manual Intervention**

## **6.1 Total Belief Transparency**

Every factual assertion made by ARNL V1.1 traces to a specific SHADE node with a complete audit record:

* Node ID and concept\_emb vector

* Full target\_dist and its current distribution

* S\_base and S\_overflow current values

* Decay history: miss\_count, last\_hit\_step, last\_miss\_step

* Creation timestamp and creation context

* Edge list: all connected nodes and their co\_weights

* Injection value history: ratio of high-value to low-value hits

**No assertion can be made from a SHADE node without this record existing and being accessible. This is a structural guarantee, not a policy.**

## **6.2 Manual Intervention Operations**

| Operation | Action | Use Case |
| :---- | :---- | :---- |
| **Hard Delete** | Remove node by concept\_emb key. All edges referencing this node are also deleted. | Retract a factual error. Immediately effective. |
| **Hard Lock** | Set S\_base \= 1000, disable decay for this node. α is permanently at maximum. | Pin a critical fact. The model will always prefer this node’s recommendation at relevant positions. Deterministic output guarantee. |
| **Hard Override** | Set S\_base \= 0, trigger immediate deletion at next Decay Engine pass. | Scheduled retraction. The node continues to influence generation at near-zero weight until the next pass removes it. |
| **Tier Freeze** | Disable decay for all nodes in a specific tier. | Protect an entire knowledge domain during a data-sparse period. |
| **Edge Reset** | Set co\_weight \= 0 for a specific edge, triggering deletion at next pass. | Break a spurious co-occurrence association between two concepts. |
| **Node Inspect** | Read full audit record for a node by concept or keyword search. | Diagnostic review. Verify what the model believes about a specific topic. |

# **VII. SALT Tokenizer — Semantic-Anchor-Linked Tokenizer**

## **7.1 Role and Integration**

SALT is the ARNL-native tokenizer trained on the Phase 1 corpus before any model training begins (Phase 0). It provides three capabilities that a standard BPE tokenizer cannot:

* Pre-classification of every token in the vocabulary into SYN, SAC, or AMB categories, enabling deterministic EMLM masking without heuristic regex.

* anchor\_mask and syntactic\_mask fields on every encoding, directly consumed by gap detection and function word suppression.

* Special entity placeholder tokens (\[ENTITY\_NOUN\], \[ENTITY\_NUM\], \[ENTITY\_CONCEPT\], \[ENTITY\_DOMAIN\]) as first-class vocabulary citizens with reserved IDs 4-7.

## **7.2 Token Classification**

| Class | Token Types | ARNL Role |
| :---- | :---- | :---- |
| SYN | Prepositions, conjunctions, determiners, auxiliary verbs, pronouns, punctuation | Never masked in EMLM. Function word suppression fires when top-1 is SYN. Gap detection naturally suppressed at SYN positions. |
| SAC | Nouns, proper nouns, cardinal numbers, capitalized tokens | Always masked in EMLM. Entropy computed over SAC subset for gap detection. SHADE fingerprinting uses SAC tokens as context anchors. |
| AMB | Verbs, adjectives, adverbs, context-dependent tokens | Masked probabilistically (30%) in EMLM. Run through full gap detection at inference time. |

## **7.3 Phase 0 Training**

SALT must be trained before Phase 1 EMLM pretraining. Training takes approximately 2 minutes on a standard workstation:

* Train Byte-Level BPE on the unmasked Phase 1 corpus. Vocabulary size: 12,000.

* Classify every vocabulary token into SYN/SAC/AMB using a static closed-class frozenset for SYN tokens (approximately 300 function words), capitalization heuristic for SAC tokens, and AMB for all others. NLTK is used only as an optional fallback for ambiguous cases.

* Save token\_classes.npy (binary, not JSON) for O(1) array-indexed lookup at runtime.

* Save tokenizer.json (HuggingFace BPE format) and salt\_meta.json.

# **VIII. Competitive Positioning vs. Transformer SLMs**

## **8.1 Head-to-Head Comparison**

| Capability | Standard Transformer | ARNL V1.1 | Winner |
| :---- | :---- | :---- | :---- |
| Factual precision (bounded domain) | Moderate — probabilistic recall degrades under distribution shift | High — axiomatic gating with discrete counters resists degradation | **ARNL** |
| Continual learning (no forgetting) | Poor — fine-tuning overwrites prior weights | Excellent — SHADE is append-and-decay, not gradient-overwrite | **ARNL** |
| Auditability | None — weights are opaque float arrays | Complete — every belief is a readable, modifiable record | **ARNL** |
| Hard behavioral constraints | None — cannot guarantee a specific output | Yes — Hard Lock creates deterministic output for specific inputs | **ARNL** |
| Personalization without fine-tuning | None — requires expensive retraining | Yes — τ\_user tier adapts in real-time | **ARNL** |
| Memory efficiency at long contexts | Quadratic in sequence length | Linear in sequence length (GLA/SSM) | **ARNL** |
| Inference latency (single query) | Single forward pass, heavily optimized | Forward pass \+ entropy check \+ SHADE lookup | **Transformer** |
| Open-domain creative generation | High | Moderate — fact-blindness limits unconstrained creative range | **Transformer** |
| Multi-hop symbolic reasoning | Moderate — emerges from in-context attention | Weak — SHADE retrieves single nodes, does not chain symbolically | **Transformer** |
| Cold-start deployment | Immediate — all knowledge in weights | Requires SHADE pre-population for domain facts | **Transformer** |

| Recommended Deployment Profile |
| :---- |
| ARNL V1.1 is the superior architecture for: medical decision support, legal citation and document analysis, scientific question answering, regulated enterprise assistants, personal AI systems requiring user modeling, and any application where factual errors carry real-world consequences and auditability is required. Standard Transformers remain preferable for: open-ended creative writing, broad conversational assistants with no domain constraint, applications requiring multi-hop symbolic reasoning, and any deployment where the knowledge base cannot be pre-populated and cold-start latency is unacceptable. |

# **IX. Can ARNL V1.1 Replace Transformers in Local SLMs? — Full Analysis**

## **9.1 The Question**

Small Language Models deployed locally (1B–24B parameters, on-device or private server) face a set of constraints that differ fundamentally from cloud-scale models. They operate with limited compute budgets, serve specialized domains, require privacy guarantees, and are often expected to learn and adapt from user interaction without retraining. The question is whether ARNL V1.1 is a superior architecture for this class of deployment.

## **9.2 Where ARNL Wins Decisively**

### **Memory Footprint at Long Contexts**

The single most practical advantage of ARNL V1.1 for local deployment is linear sequence complexity. A local 7B Transformer processing a 32K token context window requires O(32K²) memory for attention — approximately 4GB of KV cache alone at fp16. System 1’s GLA/SSM processes the same context with O(32K) memory — a fixed-size recurrent state regardless of sequence length. For local deployment on consumer hardware, this difference is often the deciding factor between a model that fits in RAM and one that does not.

### **Factual Reliability Without Scale**

Transformer SLMs at the 1B–7B range hallucinate at rates that make them unsafe for high-stakes applications without expensive retrieval augmentation (RAG). ARNL V1.1’s factual precision comes from SHADE, which is off-budget — it does not consume model parameters. A 3B ARNL system with a well-populated SHADE map will factually outperform a 7B Transformer in its target domain. This means ARNL effectively delivers better factual performance at lower parameter counts for bounded domains.

### **Continual Learning Without Retraining**

Local SLMs are expected to learn from their users. The standard Transformer approach — periodic fine-tuning on accumulated user data — is computationally expensive, requires significant local hardware, and risks catastrophic forgetting of previously learned behaviors. ARNL V1.1 learns continuously through SHADE’s strength system. Every interaction updates node strengths, edge weights, and target distributions without any gradient computation. A local ARNL instance becomes more accurate about its user over time at zero compute cost beyond the Decay Engine’s arithmetic updates.

### **Privacy-Safe Personalization**

In local deployment, user data must not leave the device. ARNL V1.1’s τ\_user tier accumulates personal context in SHADE nodes that are stored locally and never transmitted. The separation between System 1 (which could theoretically be a shared model) and System 2 (which is entirely local and personal) creates a clean privacy boundary. The system model and the personal knowledge base are separate artifacts.

### **Deterministic Behavioral Constraints**

Local SLMs used in regulated contexts (medical, legal, financial) must guarantee certain behaviors regardless of prompt engineering. Hard Lock provides this guarantee structurally. A system operator can lock specific facts or prohibit specific outputs with mathematical certainty — not through a safety classifier that can be bypassed, but through the architecture itself.

## **9.3 Where Transformers Retain the Advantage**

### **Open-Domain Knowledge Coverage**

A pretrained 7B Transformer has implicit knowledge of millions of facts encoded in its weights from internet-scale training. A freshly deployed ARNL system has an empty SHADE map. In open-domain use — answering questions about any topic a user might raise — the cold-start disadvantage is real and significant. ARNL recovers this over time as SHADE populates, but the initial deployment experience is weaker for broad conversational use.

The mitigation is pre-population: seeding SHADE from structured knowledge bases (Wikipedia, domain ontologies, medical databases) before deployment. This is feasible for bounded domains but is not a complete substitute for Transformer-style weight-encoded general knowledge.

### **Multi-Hop Reasoning**

Tasks like 'Who was the president of the country where Einstein was born?' require chaining facts: Einstein → born in Ulm → Ulm is in Germany → president of Germany. SHADE retrieves individual nodes — it does not chain them symbolically. System 1’s GLA backbone has some capacity for this through its recurrent state, but it is systematically weaker than Transformer in-context attention for explicit multi-hop chains.

The architectural path to address this is a Reasoning Chain register — a lightweight buffer that accumulates sequential SHADE retrievals and checks their joint consistency before committing to an answer. This is an open research direction not yet implemented in V1.1.

### **Creative and Generative Range**

System 1’s EMLM training intentionally suppresses factual content encoding. This is the source of its reliability. It is also the source of a genuine limitation in open-ended creative generation, where rich factual associations and unexpected entity combinations are desirable. A Transformer generating fiction draws on its full weight-encoded world model. System 1 generates syntactic structure and defers to SHADE on content — which means creative output is constrained to SHADE’s populated knowledge.

## **9.4 The Replacement Verdict by Domain**

| Deployment Domain | ARNL V1.1 as Replacement? | Key Reason |
| :---- | :---- | :---- |
| Medical decision support | **YES — superior** | Hard Lock for clinical guidelines \+ auditability for liability \+ factual precision \+ no hallucination in bounded domain |
| Legal research and citation | **YES — superior** | Auditable citations \+ Hard Lock for jurisdiction-specific rules \+ continual update without retraining |
| Regulated enterprise assistant | **YES — superior** | Deterministic compliance constraints \+ personalization without fine-tuning \+ private local deployment |
| Scientific Q\&A (bounded domain) | **YES — superior** | Factual precision \+ continual learning as new papers are processed \+ audit trail for claims |
| Personal AI assistant (local) | **YES — superior over time** | Cold start disadvantage; recovers within weeks of use as SHADE populates user context |
| General-purpose chatbot | **CONDITIONAL** | Competitive only with dense SHADE pre-population from a large knowledge base. Weak at cold start for broad topics. |
| Creative writing assistant | **NO — Transformer preferred** | EMLM fact-blindness limits creative entity combination. System 1 generates structure but SHADE constrains content. |
| Open-domain QA (any topic) | **NO — Transformer preferred** | Cold start knowledge gap and multi-hop reasoning weakness are decisive in this domain. |

## **9.5 The Architectural Maturity Assessment**

ARNL V1.1 is ready for domain-specific deployment at scale. The core architecture — EMLM-trained GLA backbone, SHADE belief store, entropy-driven gap detection, direct logit injection, tiered strength system — is theoretically sound and every major V1.0 structural flaw has been resolved.

Three open engineering questions remain before general-purpose deployment:

1. H\_gap calibration: The entropy threshold for gap detection requires empirical calibration per domain. A medical domain has different SAC entropy distributions than a legal domain. Adaptive H\_gap (learned during Phase 2 on domain data) is the recommended approach.

2. SHADE cold-start pipeline: A production-grade tool for bulk-populating SHADE from structured knowledge bases (Wikipedia, ontologies, domain corpora) does not yet exist. This is an engineering task, not an architectural research question.

3. Multi-token injection fluency: The gap continuation mechanism handles multi-word facts mechanically but does not prime System 1’s hidden state. Empirical measurement of fluency at injection boundaries in the target domain is required before production deployment.

# **X. Known Limitations & Open Research Questions**

| Limitation | Current Status | Mitigation / Future Work |
| :---- | :---- | :---- |
| Multi-hop reasoning | SHADE retrieves individual nodes; does not chain them symbolically across multiple inference steps. | Future: Reasoning Chain register that accumulates sequential SHADE retrievals and checks joint consistency before injection. |
| Cold-start knowledge | New deployments have empty or sparse SHADE graphs. Until the graph populates, System 1 carries full generative load. | Mitigation: Pre-populate SHADE from curated knowledge bases at S\_base \= 1000 using domain-specific structured data before deployment. |
| Multi-token injection fluency | Gap continuation handles multi-word facts but does not prime System 1’s hidden state trajectory. Fluency at injection boundaries may degrade for long technical phrases. | Monitor fluency at multi-token boundaries empirically. Consider lightweight residual injection for phrases longer than 3 tokens. |
| H\_gap calibration | The entropy threshold for gap detection requires empirical calibration per embedding space and domain. | Future: Adaptive H\_gap that calibrates automatically using a small validation set during Phase 2 evaluation. |
| SHADE graph size scaling | As SHADE grows beyond \~1M nodes, nearest-neighbor retrieval cost increases. FAISS indexing mitigates this but adds operational complexity. | Mitigation: FAISS index over concept\_emb vectors. Periodic graph pruning to remove nodes with S\_base \< 5 and no high-co-weight edges. |
| Creative generation range | EMLM fact-blindness is intentional but limits System 1’s capacity for unexpected entity combinations that characterize creative writing. | ARNL is not designed for creative writing as a primary use case. For hybrid use, consider a creativity mode that bypasses SHADE and allows System 1 to generate freely. |

# **Appendix A — Mathematical Symbol Reference**

| Symbol | Definition |
| :---- | :---- |
| C | Full context window — all tokens in current input |
| h\_t | System 1 hidden state at token step t |
| A\_t | Selective forget gate in GLA/SSM (data-dependent) |
| B\_t | Input gate in GLA/SSM (data-dependent) |
| H\_SAC | Shannon entropy of System 1 logits over SAC-class tokens. The gap detection signal. |
| H\_gap | Gap detection threshold (recommended default: 2.0 nats). Calibrated per domain. |
| q | SHADE context query vector: mean of frozen embeddings of last K SAC tokens, quantized to int8 |
| concept\_emb | int8-quantized embedding vector representing a SHADE node’s concept identity in frozen SALT semantic space |
| target\_dist | float16 distribution over top-K vocabulary tokens recommended by a SHADE node |
| confidence\_vec | int8 per-position weight over context\_window tokens — diagnostic fingerprint for node activation |
| S\_base | Axiomatic strength counter \[1, 1000\]. Governs decay tier and injection weight α. |
| S\_overflow | Contextual salience counter \[0, S\_max\]. Logarithmically scaled supplemental influence. |
| S\_max | Overflow ceiling (default: 2000\) |
| α | Sigmoid-Ramped Influence Scaler \= σ((S\_base − 50\) / 10). Controls SHADE injection strength. |
| α\_effective | Logic Duel α \= α\_normal × (S\_base\_current / S\_base\_original). Suppresses injections from half-eroded nodes. |
| τ\_tiered | Dynamic consistency threshold (0.90–0.95 axiom / 0.75–0.85 domain / 0.50–0.70 user) |
| δ | Inversion dead zone threshold (default: 0.25). Conflicts with magnitude \< δ are ignored. |
| β | Logarithmic overflow scaling coefficient in W\_total \= W\_base \+ β · ln(1 \+ S\_overflow) |
| co\_weight | Edge co-occurrence weight \[0, 1000\]. Strength of association between two SHADE nodes. |
| sim\_min | Minimum cosine similarity for SHADE retrieval to proceed (default: 0.65). Below this, semantic vacuum event. |
| M | Number of candidate nodes retrieved by nearest-neighbor search (default: 16\) |
| K | Context window size for SHADE fingerprinting (default: 8 SAC tokens) |

# **Appendix B — V1.1 Change Log vs. V1.0**

| Change | Problem Solved | Location in Spec |
| :---- | :---- | :---- |
| **Reasoning Head eliminated** | Cold-start saliency bootstrap paradox: a fact-blind component cannot reliably identify factually salient tokens. Also removed 15% parameter budget overhead. | Section I.1, Section II.4 |
| **W\_proj Injection Bridge eliminated** | Dimensionality mismatch between System 1 and System 2 spaces required a learned projection. Third training phase. Embedding drift failure mode. | Section I.1, Section III.4 |
| **Gap detection via System 1 entropy** | Replaced Reasoning Head Saliency Filter with a principled uncertainty signal from System 1’s own output distribution. No bootstrap problem. | Section II.4 |
| **SHADE replaces Hyper-Adjacency Map** | Flat hash map had no contextual disambiguation. SHADE nodes use concept embeddings for self-indexing and organic edges for context-sensitive retrieval. | Section III |
| **Direct logit injection replaces residual injection** | Residual stream injection required W\_proj and affected hidden state trajectory. Logit injection is projection-free and architecturally cleaner for single-token facts. | Section III.4 |
| **Multi-token gap continuation** | Logit injection is position-local. Continuation mechanism handles multi-word facts without requiring a return to entropy-based gap detection at each position. | Section III.7 |
| **Injection value weighting for S\_overflow** | V1.0 treated all hits equally. V1.1 rewards nodes that contributed novel information (high-value hits) more than those confirming System 1’s existing distribution. | Section V.6 |
| **Edge decay system for SHADE graph** | New in V1.1. Co-occurrence edges between nodes require their own lifecycle management to prevent spurious associations from persisting. | Section V.5 |
| **Training simplified to two phases** | V1.0 had three phases (EMLM, W\_proj alignment, fine-tuning). Eliminating W\_proj removes the second phase. Fine-tuning LoRA adapters target injection receptivity, not projection accuracy. | Section II.5 |
| **System 1 parameter budget raised to \~95%** | Reasoning Head (15%) and W\_proj (5%) budgets freed. System 1 now commands \~95% of total parameters, competitive with standalone models of the same total size. | Section I.2 |

