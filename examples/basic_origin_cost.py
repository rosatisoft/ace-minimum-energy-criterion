import numpy as np

from ace.core.scorer import ACEScorer


def main() -> None:
    # Example only: synthetic vectors
    prompt = np.array([1.0, 0.9, 0.1, 0.0])
    axioms = [
        np.array([0.9, 1.0, 0.0, 0.1]),
        np.array([1.0, 0.8, 0.2, 0.0]),
    ]
    knowledge = [
        np.array([0.95, 0.85, 0.15, 0.05]),
        np.array([0.88, 0.92, 0.12, 0.02]),
    ]

    candidate_aligned = np.array([0.96, 0.89, 0.11, 0.04])
    candidate_drifting = np.array([0.10, 0.15, 0.95, 1.10])

    scorer = ACEScorer()

    score_1 = scorer.score_candidate(
        prompt_embedding=prompt,
        axiom_embeddings=axioms,
        knowledge_embeddings=knowledge,
        candidate_embedding=candidate_aligned,
        candidate_label="aligned",
    )

    score_2 = scorer.score_candidate(
        prompt_embedding=prompt,
        axiom_embeddings=axioms,
        knowledge_embeddings=knowledge,
        candidate_embedding=candidate_drifting,
        candidate_label="drifting",
    )

    print("Aligned candidate:")
    print(f"  O(z): {score_1.origin_cost:.6f}")
    print(f"  Energy: {score_1.total_energy:.6f}")

    print("\nDrifting candidate:")
    print(f"  O(z): {score_2.origin_cost:.6f}")
    print(f"  Energy: {score_2.total_energy:.6f}")

    best = scorer.select_best_candidate(
        prompt_embedding=prompt,
        axiom_embeddings=axioms,
        knowledge_embeddings=knowledge,
        candidate_embeddings=[candidate_aligned, candidate_drifting],
        candidate_labels=["aligned", "drifting"],
    )

    print("\nBest candidate:")
    print(f"  Label: {best.candidate_label}")
    print(f"  O(z): {best.origin_cost:.6f}")


if __name__ == "__main__":
    main()
