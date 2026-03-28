# ACE Minimum Energy Criterion

### A Minimum-Energy Framework for Semantic Stability in LLMs

ACE is a model-agnostic framework that stabilizes language generation by modeling it as a **minimum-energy optimization problem under semantic constraints**.

This repository implements the origin cost component of the ACE framework.

It introduces a deterministic scoring layer that evaluates candidate responses based on their **semantic alignment with a defined origin**.

📄 DOI (paper): https://doi.org/10.5281/zenodo.19162999  
📦 Full framework: https://github.com/rosatisoft/axiomatic-criterion-engine  

---

## 🧠 Core Idea

Language generation is not only probabilistic — it is a **constrained dynamical system**.

ACE defines an extended energy function:

E'(z | x, A, K) = λ_p (-log P(z|x)) + λ_h H_c(x) + λ_a D(z,A) + λ_k C(z,K) + λ_o O(z)

Where:

- H_c(x): contextual disorder  
- D(z,A): conceptual deviation  
- C(z,K): knowledge inconsistency  
- O(z): **origin cost (semantic grounding penalty)**  

---

## 📌 Origin Cost (O(z))

The core contribution of ACE is the formalization of **semantic anchoring**:

O(z) = ||V_z - Π_S(V_z)||²

Where:
- V_z = embedding of candidate response  
- S = subspace defined by prompt, axioms, and knowledge  
- Π_S = projection onto S  

👉 Intuition:
- low O(z) → aligned, grounded response  
- high O(z) → drift, hallucination, semantic escape  

---
## Repository Structure

ace-minimum-energy-criterion
│
├─ notebooks/                         ← real research and experimental work
│  ├─ ace_semantic_convergence.ipynb
│  ├─ ace_semantic_convergence_V1.0.ipynb
│  ├─ ace_semantic_test.ipynb
│  └─ ace_marble_rolling_animation.ipynb
│
├─ ace/                               ← main Python package
│  ├─ __init__.py                     ← package entry point
│  ├─ scorer.py                       ← decision layer (embedding-based pipeline)
│  ├─ types.py                        ← shared dataclasses and structures
│  │
│  └─ core/                           ← mathematical core of ACE
│     ├─ __init__.py
│     ├─ core_scorer.py               ← minimal scoring based on origin cost
│     ├─ subspace.py                  ← construction of the semantic subspace
│     └─ origin_cost.py               ← computation of the origin cost
│
├─ examples/                          ← minimal usage examples
│  └─ basic_origin_cost.py
│
├─ tests/                             ← unit tests
│  └─ test_origin_cost.py
│
├─ docs/                              ← documentation and theoretical material
│  ├─ ACE-Minimum-Energy-Criterion.pdf
│  ├─ ACE_ARCHITECTURE.md
│  ├─ ACE_DIAGRAM.md
│  ├─ ACE_PIPELINE_DIAGRAM.md
│  └─ THEORETICAL_FRAMEWORK.md
│
├─ ACE-20.json                        ← experimental dataset
│
├─ README.md                          ← main project documentation
├─ README_V1.0.md                     ← previous version of the README
│
├─ pyproject.toml                     ← Python package configuration
├─ requirements.txt                   ← additional dependencies
│
└─ LICENSE

## ⚙️ Quick Example

```python
import numpy as np
from ace.core.scorer import ACEScorer

scorer = ACEScorer()

score = scorer.score_candidate(
    prompt_embedding=np.array([...]),
    axiom_embeddings=[np.array([...])],
    knowledge_embeddings=[np.array([...])],
    candidate_embedding=np.array([...])
)

print(score.origin_cost)
