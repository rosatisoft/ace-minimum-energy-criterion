from ace.scorer import ACEScorer

prompt = "Explain why semantic grounding matters for reliable LLM outputs."

axioms = [
    "A response should remain aligned with its source context.",
    "Semantic drift increases hallucination risk.",
]

knowledge = [
    "Grounded responses preserve coherence.",
    "Ungrounded responses tend toward speculation.",
]

candidates = {
    "precise": (
        "Semantic grounding matters because it keeps the response tied to the original "
        "context and reduces drift, ambiguity, and hallucination risk."
    ),
    "expansive": (
        "Semantic grounding matters because language unfolds through many possible "
        "interpretive layers, and meaning depends on broad conceptual resonance."
    ),
    "speculative": (
        "Semantic grounding mainly matters because it improves hardware-level memory "
        "allocation inside transformer GPUs."
    ),
}

scorer = ACEScorer(
    low_drift_threshold=0.30,
    high_drift_threshold=0.60,
    clarify_gap_threshold=0.05,
)

best, scores = scorer.select_best(
    prompt=prompt,
    axioms=axioms,
    knowledge=knowledge,
    candidates=candidates,
)

print("=== SELECT BEST ===")
print("Best:", best)
print("Scores:", scores)

decision = scorer.decide(
    prompt=prompt,
    axioms=axioms,
    knowledge=knowledge,
    candidates=candidates,
)

print("\n=== DECISION ===")
for key, value in decision.items():
    print(f"{key}: {value}")
