# Theoretical Framework of the Axiomatic Criterion Engine (ACE)

## Overview

The Axiomatic Criterion Engine (ACE) introduces a geometric criterion for evaluating semantic stability in language model outputs.

Rather than treating generation purely as a probabilistic token-selection process, ACE models candidate responses as vectors evolving in a semantic space constrained by contextual structure.

The key hypothesis is that **hallucination and semantic drift emerge when responses escape the reference semantic subspace defined by the input context.**

ACE therefore evaluates responses through their **distance from that subspace**.

---

# 1. Language Generation as a Semantic System

Large Language Models generate responses by maximizing conditional token probabilities:

\[
P(z|x)
\]

Where:

- \(x\) = input context
- \(z\) = generated response

While this probabilistic formulation enables flexible language generation, it does not guarantee:

- semantic grounding
- contextual consistency
- conceptual stability

As a result, models can produce responses that are **probabilistically plausible but semantically unstable**.

ACE reframes the problem.

Instead of optimizing only probability, ACE evaluates **structural semantic alignment**.

---

# 2. Semantic Representation

Let:

\[
V_z \in \mathbb{R}^d
\]

be the embedding vector of a candidate response.

Semantic context is represented by a set of vectors derived from:

- the prompt
- guiding axioms
- contextual knowledge

These vectors form a **reference semantic structure**.

---

# 3. The Reference Subspace

ACE constructs a semantic reference subspace \(S\) from the vectors:

\[
S = span(V_{prompt}, V_{axioms}, V_{knowledge})
\]

Intuitively, this subspace represents the **region of semantic space consistent with the intended context**.

Responses aligned with this subspace remain semantically grounded.

Responses outside the subspace indicate semantic drift.

---

# 4. Origin-Based Criterion

The ACE criterion evaluates candidate responses using the residual distance from the reference subspace:

\[
O(z) = \| V_z - \Pi_S(V_z) \|^2
\]

Where:

- \(V_z\) = candidate response embedding
- \(\Pi_S(V_z)\) = projection of \(V_z\) onto subspace \(S\)

Interpretation:

- **Low \(O(z)\)** → response lies within the semantic context
- **High \(O(z)\)** → response diverges from contextual grounding

This quantity is referred to as the **origin cost**.

---

# 5. Semantic Stability

ACE interprets semantic alignment as a stability condition.

Responses close to the reference subspace form **stable semantic attractors**.

Responses far from the subspace represent **unstable trajectories** that often correspond to:

- hallucinations
- speculative reasoning
- semantic drift

Thus:

\[
Semantic\ Stability \propto \frac{1}{O(z)}
\]

---

# 6. Generation vs Selection

ACE does not modify the internal generation process of a language model.

Instead, it introduces a **selection criterion** applied after candidate responses are generated.

The process becomes:

Prompt
↓
LLM generates candidate responses
↓
ACE evaluates semantic alignment
↓
Decision layer


Possible outcomes:

- **answer** → a candidate sufficiently aligns with the semantic subspace
- **clarify** → the semantic region is underdefined
- **abstain** → no candidate lies within acceptable alignment

---

# 7. From Probabilistic Generation to Constrained Convergence

Traditional LLM pipeline:

Prompt → Probabilistic sampling → Output


ACE pipeline:


Prompt
↓
Probabilistic generation
↓
Semantic criterion evaluation
↓
Deterministic decision


Generation remains probabilistic.

Selection becomes **criterion-based**.

---

# 8. Relation to Entropy

ACE complements entropy-based uncertainty approaches.

Entropy measures uncertainty in probability distributions.

ACE measures **structural alignment in semantic space**.

Both perspectives address different aspects of the same problem:

| Concept | Measures |
|-------|-------|
| Shannon entropy | probabilistic uncertainty |
| semantic entropy | meaning-level uncertainty |
| ACE origin cost | structural alignment with context |

---

# 9. Practical Implication

ACE suggests that many hallucinations arise not from probability errors but from **lack of structural grounding**.

If responses are constrained to remain near a contextual semantic subspace:

- semantic drift decreases
- responses converge faster
- token usage is reduced

This leads to a practical architecture:

LLM → candidate responses → ACE evaluation → decision


---

# 10. Research Direction

Future work includes:

- formalizing contextual disorder \(H_c(x)\)
- identifying origin structures within model representations
- integrating origin-aware constraints into decoding strategies
- large-scale empirical evaluation across models

---

# Conclusion

ACE proposes a shift in perspective:

Language model reliability may depend not only on better probability estimation, but on enforcing **structural semantic grounding**.

The origin-based criterion provides a simple geometric mechanism to evaluate that grounding.

This transforms response selection from purely probabilistic sampling into **constrained semantic convergence**.

---

**Related work**

Rosati Beristain, Ernesto  
*ACE: A Minimum-Energy Criterion for Entropy-Controlled Language Generation*  
Zenodo, 2026

DOI:  
https://doi.org/10.5281/zenodo.19162999
