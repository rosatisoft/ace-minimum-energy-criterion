# ACE Pipeline Diagram

The following diagram summarizes the ACE decision pipeline.

User Prompt
│
▼
Language Model
(probabilistic generation)
│
▼
Candidate Responses
│
▼
Embedding Projection
│
▼
Reference Subspace Construction
(prompt + axioms + knowledge)
│
▼
Origin Cost Evaluation
O(z)
│
▼
Decision Layer
│
├── answer
├── clarify
└── abstain


---

# Conceptual Shift

Traditional LLM pipeline:


Prompt
↓
Probabilistic Sampling
↓
Output


ACE pipeline:


Prompt
↓
Probabilistic Generation
↓
Semantic Criterion
↓
Deterministic Decision


---

# Key Insight

Generation remains probabilistic.

Selection becomes **criterion-based**.

This allows systems to enforce semantic alignment without modifying the internal model.
