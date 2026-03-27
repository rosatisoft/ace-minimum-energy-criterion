# ACE Architecture

## Overview

The Axiomatic Criterion Engine (ACE) introduces a semantic decision layer between language model generation and final output delivery.

Instead of accepting the first generated response, ACE evaluates candidate responses against a semantic reference structure derived from the input context.

The result is a **criterion-based response control layer**.

---

# System Architecture

The ACE pipeline can be summarized as:

User Prompt
↓
LLM Generation (multiple candidates)
↓
Embedding Projection
↓
Reference Subspace Construction
↓
Origin Cost Evaluation
↓
Decision Layer


Possible outcomes:

- answer
- clarify
- abstain

---

# Step 1: Context Construction

ACE builds a contextual reference using three sources:

Prompt
Axioms
Contextual Knowledge


These elements define the intended semantic region.

Each element is embedded into vector space.

---

# Step 2: Semantic Subspace Construction

The embedded vectors form a matrix:

[prompt_vector
axiom_vectors
knowledge_vectors]

ACE computes the principal semantic structure of this matrix.

Using SVD:

S = span(prompt, axioms, knowledge)


This produces a **reference semantic subspace**.

---

# Step 3: Candidate Generation

The language model generates multiple candidate responses.

Example:

candidate_1
candidate_2
candidate_3


Each candidate is embedded into the same vector space.

---

# Step 4: Origin Cost Evaluation

For each candidate embedding \(V_z\), ACE computes:

\[
O(z) = || V_z - \Pi_S(V_z) ||^2
\]

Where:

- \(S\) is the semantic subspace
- \(\Pi_S\) is projection onto that subspace

This measures **semantic drift from the contextual origin**.

---

# Step 5: Candidate Ranking

Candidates are ranked by origin cost:

lowest O(z) → best alignment


However, ranking alone is not sufficient.

ACE therefore adds a decision layer.

---

# Step 6: Decision Layer

ACE determines whether a response should be produced.

Three outcomes exist:

### answer

A candidate aligns with the semantic context.

### clarify

Candidates are too similar or the semantic space is underdefined.

The system should request clarification.

### abstain

All candidates fall outside acceptable alignment.

The system should not produce an answer.

---

# Decision Logic

A simplified logic:

if best_score <= low_threshold:
return answer

elif best_score >= high_threshold:
return abstain

elif candidates_are_close:
return clarify


---

# Design Principles

ACE follows several architectural principles:

### Model agnostic

ACE does not modify the internal architecture of the LLM.

### Deterministic evaluation

Candidate evaluation is deterministic once embeddings are produced.

### External control layer

ACE operates as middleware between generation and response delivery.

### Geometric criterion

Semantic alignment is measured through vector geometry.

---

# Integration Points

ACE can be integrated in multiple ways.

### Post-generation filtering

LLM → candidates → ACE → final output


### Multi-agent orchestration

ACE evaluates outputs from multiple agents.

### Safety layer

ACE can detect semantic drift before responses reach users.

---

# Performance Characteristics

ACE evaluation involves:

- embedding computation
- SVD of a small matrix
- projection calculations

For typical contexts, the computational cost is small relative to generation.

---

# Future Architecture Directions

Possible extensions include:

- origin-aware decoding
- integration with retrieval systems
- hierarchical semantic subspaces
- adaptive threshold learning

---

# Summary

ACE transforms the language model pipeline:

probabilistic generation
→ semantic evaluation
→ deterministic decision


This introduces a structural control mechanism for semantic stability.
