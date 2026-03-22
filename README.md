# ACE: Axiomatic Criterion Engine

### A Minimum-Energy Framework for Semantic Stability in LLMs

ACE is a model-agnostic framework that stabilizes language generation by modeling it as a **minimum-energy optimization problem under semantic constraints**.

---

## 🧠 Core Idea

Language generation is not only probabilistic — it is a **constrained dynamical process**.

ACE defines an energy function:

E'(z | x, A, K) = λ_p (-log P(z|x)) + λ_h H_c(x) + λ_a D(z,A) + λ_k C(z,K) + λ_o O(z)

Where:

- H_c(x): contextual disorder  
- D(z,A): conceptual deviation  
- C(z,K): knowledge inconsistency  
- O(z): **origin cost (structural grounding)**  

---

## 🔥 Key Insight

> Semantic stability requires structural grounding.

Without a well-defined origin, vector representations become semantically underdetermined, leading to hallucinations and inefficiency.

---

## 📊 Empirical Observation

Across multiple LLMs:

- 25–35% token reduction  
- improved coherence  
- reduced hallucination  

---

## 🧪 ACE-20 Benchmark

ACE-20 evaluates models across:

- Factual  
- Ambiguous  
- Manipulative  
- Metaphorical  
- Conflictive  

---

## ⚙️ Use Cases

- LLM guardrails  
- hallucination mitigation  
- semantic validation  
- cost optimization  

---

## 📄 Paper

See `/docs/ACE-paper.pdf`

---

## 📜 License

© 2026 Ernesto Rosati Beristain  
Licensed under CC-BY 4.0

---

## 🤝 Collaboration

Developed with AI research collaboration (ChatGPT – “Séptimo”)
