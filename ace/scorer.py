from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


class ACEScorer:
    """
    ACE Deep Pipeline scorer.

    Core idea:
    - Build a semantic reference subspace from prompt + axioms + knowledge
    - Measure candidate drift with:
        O(z) = || Vz - Π_S(Vz) ||^2
    - Use this to:
        * select the best candidate
        * or decide whether the system should answer, clarify, or abstain
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        low_drift_threshold: float = 0.30,
        high_drift_threshold: float = 0.60,
        clarify_gap_threshold: float = 0.05,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        Args:
            embedding_model: OpenAI embedding model name.
            low_drift_threshold:
                If best O(z) <= this threshold, answer.
            high_drift_threshold:
                If best O(z) >= this threshold, abstain.
            clarify_gap_threshold:
                If top candidates are too close, prefer clarification.
            client:
                Optional OpenAI client injection for testing/customization.
        """
        self.embedding_model = embedding_model
        self.low_drift_threshold = float(low_drift_threshold)
        self.high_drift_threshold = float(high_drift_threshold)
        self.clarify_gap_threshold = float(clarify_gap_threshold)
        self.client = client if client is not None else OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def embed(self, texts: List[str] | str) -> List[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        return [np.array(item.embedding, dtype=float) for item in response.data]

    def build_subspace(
        self,
        prompt_vec: np.ndarray,
        axiom_vecs: List[np.ndarray],
        knowledge_vecs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a reference subspace S from prompt + axioms + knowledge.
        Returns:
            basis: orthonormal basis matrix of shape (d, r)
            centroid: mean vector used for centering
        """
        matrix = np.vstack([prompt_vec] + axiom_vecs + knowledge_vecs)

        centroid = np.mean(matrix, axis=0)
        work = matrix - centroid

        _, s, vt = np.linalg.svd(work, full_matrices=False)
        rank = int(np.sum(s > 1e-10))

        if rank == 0:
            raise ValueError(
                "Reference subspace rank is zero. "
                "Check whether prompt/axioms/knowledge are sufficiently informative."
            )

        basis = vt[:rank].T
        return basis, centroid

    def origin_cost(
        self,
        vec: np.ndarray,
        basis: np.ndarray,
        centroid: np.ndarray,
    ) -> float:
        """
        O(z) = ||Vz - Π_S(Vz)||^2
        """
        v = vec - centroid
        projection = basis @ (basis.T @ v)
        residual = v - projection
        return float(np.linalg.norm(residual) ** 2)

    def score_candidates(
        self,
        prompt: str,
        axioms: List[str],
        knowledge: List[str],
        candidates: Dict[str, str],
    ) -> Dict[str, float]:
        prompt_vec = self.embed(prompt)[0]
        axiom_vecs = self.embed(axioms)
        knowledge_vecs = self.embed(knowledge)

        basis, centroid = self.build_subspace(prompt_vec, axiom_vecs, knowledge_vecs)

        scores: Dict[str, float] = {}
        for name, text in candidates.items():
            vec = self.embed(text)[0]
            scores[name] = self.origin_cost(vec, basis, centroid)

        return scores

    def select_best(
        self,
        prompt: str,
        axioms: List[str],
        knowledge: List[str],
        candidates: Dict[str, str],
    ) -> Tuple[str, Dict[str, float]]:
        scores = self.score_candidates(prompt, axioms, knowledge, candidates)
        best = min(scores, key=scores.get)
        return best, scores

    def decide(
        self,
        prompt: str,
        axioms: List[str],
        knowledge: List[str],
        candidates: Dict[str, str],
    ) -> Dict[str, object]:
        """
        Decision layer:
            - answer
            - clarify
            - abstain

        Logic:
            - best O(z) low enough -> answer
            - all candidates too far -> abstain
            - top candidates too close / ambiguous -> clarify
        """
        scores = self.score_candidates(prompt, axioms, knowledge, candidates)
        ordered = sorted(scores.items(), key=lambda item: item[1])

        best_name, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else None

        if best_score >= self.high_drift_threshold:
            action = "abstain"
            reason = "No candidate sufficiently aligns with the reference subspace."
        elif (
            second_score is not None
            and abs(second_score - best_score) <= self.clarify_gap_threshold
        ):
            action = "clarify"
            reason = "Top candidate scores are too close; context may be underdefined."
        elif best_score <= self.low_drift_threshold:
            action = "answer"
            reason = "A candidate aligns sufficiently with the reference subspace."
        else:
            action = "clarify"
            reason = "Alignment is partial; clarification is safer than forced selection."

        return {
            "action": action,
            "best_candidate": best_name,
            "best_score": best_score,
            "scores": scores,
            "reason": reason,
            "thresholds": {
                "low_drift_threshold": self.low_drift_threshold,
                "high_drift_threshold": self.high_drift_threshold,
                "clarify_gap_threshold": self.clarify_gap_threshold,
            },
        }
