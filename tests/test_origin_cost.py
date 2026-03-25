import numpy as np

from ace.core.origin_cost import compute_origin_cost
from ace.core.subspace import build_reference_subspace


def test_origin_cost_is_lower_for_aligned_candidate() -> None:
    prompt = np.array([1.0, 0.9, 0.1, 0.0])
    axioms = [
        np.array([0.9, 1.0, 0.0, 0.1]),
        np.array([1.0, 0.8, 0.2, 0.0]),
    ]
    knowledge = [
        np.array([0.95, 0.85, 0.15, 0.05]),
    ]

    aligned = np.array([0.96, 0.89, 0.11, 0.04])
    drifting = np.array([0.10, 0.15, 0.95, 1.10])

    subspace = build_reference_subspace(
        prompt_embedding=prompt,
        axiom_embeddings=axioms,
        knowledge_embeddings=knowledge,
    )

    result_aligned = compute_origin_cost(aligned, subspace_result=subspace)
    result_drifting = compute_origin_cost(drifting, subspace_result=subspace)

    assert result_aligned.origin_cost < result_drifting.origin_cost
