from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ace.types import SubspaceResult, Vector, Matrix


def _ensure_1d_vector(vec: Vector, name: str) -> Vector:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={arr.shape}")
    return arr


def _validate_same_dimension(vectors: List[Vector]) -> None:
    if not vectors:
        raise ValueError("At least one vector is required to build the subspace.")

    dim = vectors[0].shape[0]
    for i, vec in enumerate(vectors):
        if vec.shape[0] != dim:
            raise ValueError(
                f"All vectors must have the same dimension. "
                f"Vector 0 has dim={dim}, vector {i} has dim={vec.shape[0]}"
            )


def _stack_vectors(vectors: Iterable[Vector]) -> Matrix:
    processed = [_ensure_1d_vector(v, "embedding") for v in vectors]
    _validate_same_dimension(processed)
    return np.vstack(processed)


def build_reference_subspace(
    prompt_embedding: Vector,
    axiom_embeddings: List[Vector],
    knowledge_embeddings: List[Vector],
    *,
    center: bool = True,
    rank: Optional[int] = None,
    svd_tol: float = 1e-10,
) -> SubspaceResult:
    """
    Build an orthonormal basis for the reference subspace S using SVD.

    The subspace is constructed from:
    - prompt embedding
    - axiom embeddings
    - knowledge embeddings

    If center=True, vectors are centered by the centroid before SVD.
    """
    prompt_embedding = _ensure_1d_vector(prompt_embedding, "prompt_embedding")
    axiom_embeddings = [_ensure_1d_vector(v, "axiom_embedding") for v in axiom_embeddings]
    knowledge_embeddings = [_ensure_1d_vector(v, "knowledge_embedding") for v in knowledge_embeddings]

    all_vectors = [prompt_embedding] + axiom_embeddings + knowledge_embeddings
    matrix = _stack_vectors(all_vectors)

    centroid = np.mean(matrix, axis=0) if center else None
    work_matrix = matrix - centroid if center else matrix.copy()

    # Rows are samples, columns are dimensions.
    # Right singular vectors (Vt) span the feature-space directions.
    _, s, vt = np.linalg.svd(work_matrix, full_matrices=False)

    inferred_rank = int(np.sum(s > svd_tol))
    final_rank = inferred_rank if rank is None else min(rank, inferred_rank)

    if final_rank <= 0:
        raise ValueError(
            "The computed subspace rank is zero. "
            "This usually means the reference vectors are degenerate."
        )

    basis = vt[:final_rank].T  # columns = orthonormal basis vectors

    return SubspaceResult(
        basis=basis,
        singular_values=s,
        rank=final_rank,
        centroid=centroid,
        metadata={
            "centered": center,
            "input_count": len(all_vectors),
            "dimension": matrix.shape[1],
            "inferred_rank": inferred_rank,
            "used_rank": final_rank,
            "svd_tol": svd_tol,
        },
    )
