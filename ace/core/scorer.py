from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ace.core.origin_cost import compute_origin_cost
from ace.core.subspace import build_reference_subspace
from ace.types import CandidateScore, ContextBundle, SubspaceResult, Vector


class ACEScorer:
    """
    Minimal ACE scorer focused on Origin Cost O(z).

    Future versions can extend this scorer with:
    - D(z, A): axiom distance
    - C(z, K): knowledge consistency
    - H_c(x): contextual entropy
    - -log P(z): probabilistic penalty
    """

    def __init__(
        self,
        *,
        center_subspace: bool = True,
        default_rank: Optional[int] = None,
        svd_tol: float = 1e-10,
        lambda_o: float = 1.0,
    ) -> None:
        self.center_subspace = center_subspace
        self.default_rank = default_rank
        self.svd_tol = svd_tol
        self.lambda_o = float(lambda_o)

    def build_subspace(
        self,
        prompt_embedding: Vector,
        axiom_embeddings: List[Vector],
        knowledge_embeddings: List[Vector],
        *,
        rank: Optional[int] = None,
    ) -> SubspaceResult:
        return build_reference_subspace(
            prompt_embedding=prompt_embedding,
            axiom_embeddings=axiom_embeddings,
            knowledge_embeddings=knowledge_embeddings,
            center=self.center_subspace,
            rank=rank if rank is not None else self.default_rank,
            svd_tol=self.svd_tol,
        )

    def score_candidate(
        self,
        *,
        prompt_embedding: Vector,
        axiom_embeddings: List[Vector],
        knowledge_embeddings: List[Vector],
        candidate_embedding: Vector,
        candidate_label: Optional[str] = None,
        rank: Optional[int] = None,
    ) -> CandidateScore:
        subspace = self.build_subspace(
            prompt_embedding=prompt_embedding,
            axiom_embeddings=axiom_embeddings,
            knowledge_embeddings=knowledge_embeddings,
            rank=rank,
        )

        origin = compute_origin_cost(
            candidate_embedding,
            subspace_result=subspace,
            center_with_centroid=True,
        )

        total_energy = self.lambda_o * origin.origin_cost

        return CandidateScore(
            origin_cost=origin.origin_cost,
            residual_norm=origin.residual_norm,
            projected_norm=origin.projected_norm,
            total_energy=total_energy,
            candidate_label=candidate_label,
            details={
                "lambda_o": self.lambda_o,
                "subspace_rank": subspace.rank,
                "subspace_metadata": subspace.metadata,
                "singular_values": subspace.singular_values.tolist(),
            },
        )

    def score_candidates(
        self,
        *,
        prompt_embedding: Vector,
        axiom_embeddings: List[Vector],
        knowledge_embeddings: List[Vector],
        candidate_embeddings: List[Vector],
        candidate_labels: Optional[List[str]] = None,
        rank: Optional[int] = None,
    ) -> List[CandidateScore]:
        subspace = self.build_subspace(
            prompt_embedding=prompt_embedding,
            axiom_embeddings=axiom_embeddings,
            knowledge_embeddings=knowledge_embeddings,
            rank=rank,
        )

        scores: List[CandidateScore] = []
        for idx, candidate_embedding in enumerate(candidate_embeddings):
            label = None
            if candidate_labels and idx < len(candidate_labels):
                label = candidate_labels[idx]

            origin = compute_origin_cost(
                candidate_embedding,
                subspace_result=subspace,
                center_with_centroid=True,
            )

            total_energy = self.lambda_o * origin.origin_cost

            scores.append(
                CandidateScore(
                    origin_cost=origin.origin_cost,
                    residual_norm=origin.residual_norm,
                    projected_norm=origin.projected_norm,
                    total_energy=total_energy,
                    candidate_label=label,
                    details={
                        "lambda_o": self.lambda_o,
                        "subspace_rank": subspace.rank,
                        "candidate_index": idx,
                    },
                )
            )

        return scores

    def select_best_candidate(
        self,
        *,
        prompt_embedding: Vector,
        axiom_embeddings: List[Vector],
        knowledge_embeddings: List[Vector],
        candidate_embeddings: List[Vector],
        candidate_labels: Optional[List[str]] = None,
        rank: Optional[int] = None,
    ) -> CandidateScore:
        scores = self.score_candidates(
            prompt_embedding=prompt_embedding,
            axiom_embeddings=axiom_embeddings,
            knowledge_embeddings=knowledge_embeddings,
            candidate_embeddings=candidate_embeddings,
            candidate_labels=candidate_labels,
            rank=rank,
        )

        if not scores:
            raise ValueError("At least one candidate embedding is required.")

        return min(scores, key=lambda item: float("inf") if item.total_energy is None else item.total_energy)

    def score_from_context_bundle(
        self,
        context: ContextBundle,
        candidate_embedding: Vector,
        *,
        candidate_label: Optional[str] = None,
        rank: Optional[int] = None,
    ) -> CandidateScore:
        return self.score_candidate(
            prompt_embedding=context.prompt_embedding,
            axiom_embeddings=context.axiom_embeddings,
            knowledge_embeddings=context.knowledge_embeddings,
            candidate_embedding=candidate_embedding,
            candidate_label=candidate_label,
            rank=rank,
        )
