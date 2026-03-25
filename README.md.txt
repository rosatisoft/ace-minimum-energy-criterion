# ACE: Axiomatic Criterion Engine

### A Minimum-Energy Framework for Semantic Stability in LLMs

ACE is a model-agnostic framework that stabilizes language generation by modeling it as a **minimum-energy optimization problem under semantic constraints**.

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