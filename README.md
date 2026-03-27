# ACE: Axiomatic Criterion Engine

### A criterion-based semantic decision layer for LLM outputs

ACE is a model-agnostic framework for constraining semantic drift in language models.

Instead of relying on prompt-engineered control, ACE introduces a **computable semantic criterion** that evaluates whether a response is sufficiently aligned with a defined reference context.

This repository contains the mathematical basis, experimental notebooks, and the first usable middleware layer of ACE.

- **Paper / DOI**: https://doi.org/10.5281/zenodo.19162999
- **Related framework repo**: https://github.com/rosatisoft/axiomatic-criterion-engine

---

## Core Idea

Language generation is not only probabilistic — it can also be evaluated as a constrained process in semantic space.

ACE introduces an origin-based criterion:

\[
O(z) = \| V_z - \Pi_S(V_z) \|^2
\]

Where:

- \(V_z\) = embedding of candidate response
- \(S\) = reference subspace built from prompt, axioms, and contextual knowledge
- \(\Pi_S\) = projection onto the reference subspace

### Intuition

- **low O(z)** → aligned, grounded response
- **high O(z)** → semantic drift, instability, or escape from context

---

## What ACE Does

ACE does not try to make the model “behave” through prompts alone.

It introduces a **criterion layer** that decides whether the system should:

- **answer**
- **ask for clarification**
- **abstain**

This makes ACE not just a scorer, but a **semantic decision layer**.

---

## Why It Matters

ACE helps:

- reduce semantic drift
- improve coherence
- support controlled abstention
- reduce unnecessary token usage
- preserve criterion under ambiguity

Important distinction:

ACE optimizes for **semantic alignment**, not direct factual verification.

That means:

- it can constrain invalid semantic trajectories
- it can detect when a response should not be produced
- it should still be combined with external verification layers in high-stakes settings

---

## Decision Behavior

Given a prompt, axioms, contextual knowledge, and candidate responses, ACE can return:

- **answer** → a candidate aligns sufficiently with the semantic reference subspace
- **clarify** → the context is underdefined or top candidates are too close
- **abstain** → no candidate sufficiently aligns with the defined context

This is the key shift:

> ACE does not only select the best answer.  
> It determines whether an answer is valid.

---

## Practical Framing

Traditional LLM control:

- tell the model what to do

ACE:

- define a criterion
- evaluate semantic admissibility
- decide whether a response should exist at all

Generation remains probabilistic.  
Evaluation and decision become deterministic.

---

## ACE Decision Pipeline

```mermaid
flowchart TD
    A[User Prompt] --> B[LLM Generation]
    B --> C[Candidate Responses]

    A --> D[Prompt]
    AX[Axioms] --> E[Reference Context]
    K[Contextual Knowledge] --> E
    D --> E

    E --> F[Embedding Projection]
    C --> G[Candidate Embeddings]

    F --> H[Reference Subspace]
    G --> I[Origin Cost Evaluation]

    H --> I
    I --> J{Decision Layer}

    J -->|low drift| K1[Answer]
    J -->|underdetermined| K2[Clarify]
    J -->|high drift| K3[Abstain]

## ACE Architecture

```mermaid
flowchart LR
    subgraph Context
        P[Prompt]
        AX[Axioms]
        K[Knowledge]
    end

    subgraph Generation
        LLM[Language Model]
        CR[Candidate Responses]
    end

    subgraph ACE
        EMB[Embeddings]
        SUB[Semantic Subspace]
        COST[Origin Cost]
        DEC{Decision}
    end

    subgraph Output
        ANS[Answer]
        CLR[Clarify]
        ABS[Abstain]
    end

    P --> LLM
    LLM --> CR

    P --> EMB
    AX --> EMB
    K --> EMB

    EMB --> SUB
    CR --> COST
    SUB --> COST
    COST --> DEC

    DEC --> ANS
    DEC --> CLR
    DEC --> ABS

```markdown
ACE introduces a semantic decision layer between probabilistic generation and final output delivery.

Instead of accepting the first response, ACE evaluates candidate outputs against a reference semantic subspace built from prompt, axioms, and contextual knowledge.

This enables three possible outcomes:

- **Answer** when alignment is sufficient
- **Clarify** when the semantic space is underdefined
- **Abstain** when no candidate satisfies the criterion

---

## Quick Example

```python
from ace.scorer import ACEScorer

scorer = ACEScorer()

decision = scorer.decide(
    prompt="Explain why semantic grounding matters for reliable LLM outputs.",
    axioms=[
        "A response should remain aligned with its source context.",
        "Semantic drift increases hallucination risk.",
    ],
    knowledge=[
        "Grounded responses preserve coherence.",
        "Ungrounded responses tend toward speculation.",
    ],
    candidates={
        "precise": "Semantic grounding matters because it reduces drift.",
        "expansive": "Meaning unfolds through broad interpretive possibility.",
        "speculative": "Semantic grounding improves GPU memory allocation.",
    },
)

print(decision)

Example output:

{
  "action": "answer",
  "best_candidate": "precise",
  "best_score": 0.23,
  "scores": {
    "precise": 0.23,
    "expansive": 0.41,
    "speculative": 0.45
  },
  "reason": "A candidate aligns sufficiently with the reference subspace."
}

Repository Structure
ace/
├── __init__.py
├── scorer.py
├── core/
│   ├── __init__.py
│   ├── origin_cost.py
│   ├── scorer.py
│   └── subspace.py
└── types.py
docs/
├── THEORETICAL_FRAMEWORK.md

examples/
├── basic_origin_cost.py
└── basic_usage.py

notebooks/
└── ace_semantic_convergence.ipynb
Demo Notebook

See:

/notebooks/ace_semantic_convergence.ipynb

Open in Colab:
https://colab.research.google.com/github/rosatisoft/ace-minimum-energy-criterion/blob/main/notebooks/ace_semantic_convergence.ipynb

Experimental Direction

Current ACE results are associated with:

lower semantic drift
more stable response selection
shorter, more efficient outputs
a practical path toward semantic middleware for LLM systems

The next implementation layer is the ACE Deep Pipeline:

LLM → candidate responses → ACE scoring / decision → answer | clarify | abstain

Scope

ACE is currently:

a semantic criterion engine
a middleware-oriented decision layer
a geometric approach to response admissibility

ACE is not yet:

a full truth engine
a replacement for external verification
an internal modification of transformer generation dynamics
Paper

See:

/docs/ACE-Minimum-Energy-Criterion.pdf

Suggested citation:

Rosati Beristain, Ernesto.
ACE: A Minimum-Energy Criterion for Entropy-Controlled Language Generation — Modeling Semantic Stability as an Attractor in Language Models.
Zenodo, 2026.
DOI: 10.5281/zenodo.19162999

License

This repository is licensed under the terms specified in LICENSE.

Collaboration

Research and implementation discussions are welcome, especially around:

semantic control layers
drift reduction
response abstention
criterion-based middleware
edge / secure / constrained AI systems
