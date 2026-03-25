from __future__ import annotations

from typing import Optional

import numpy as np

from ace.types import OriginCostResult, Vector, Matrix, SubspaceResult


def _ensure_1d_vector(vec: Vector, name: str) -> Vector:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={arr.shape}")
    return arr


def _ensure_2d_matrix(mat: Matrix, name: str) -> Matrix:
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape={arr.shape}")
    return arr


def project_vector(
    vector: Vector,
    basis: Matrix,
) -> Vector:
    """
    Project a vector onto a subspace defined by an orthonormal basis.

    basis shape: (d, r)
    vector shape: (d,)
    """
    vector = _ensure_1d_vector(vector, "vector")
    basis = _ensure_2d_matrix(basis, "basis")

    if basis.shape[0] != vector.shape[0]:
        raise ValueError(
            f"Dimension mismatch: vector dim={vector.shape[0]}, "
            f"basis dim={basis.shape[0]}"
        )

    # Since basis columns are orthonormal:
    # projection = B(B^T v)
    return basis @ (basis.T @ vector)


def compute_origin_cost(
    candidate_embedding: Vector,
    *,
    basis: Optional[Matrix] = None,
    subspace_result: Optional[SubspaceResult] = None,
    center_with_centroid: bool = True,
) -> OriginCostResult:
    """
    Compute O(z) = ||Vz - Proj_S(Vz)||^2

    If subspace_result is provided and it includes a centroid, the candidate
    is centered before projection and the projected/residual vectors are
    computed in centered coordinates.
    """
    candidate_embedding = _ensure_1d_vector(candidate_embedding, "candidate_embedding")

    if subspace_result is None and basis is None:
        raise ValueError("Provide either 'basis' or 'subspace_result'.")

    if subspace_result is not None:
        basis = subspace_result.basis
        centroid = subspace_result.centroid
    else:
        centroid = None

    basis = _ensure_2d_matrix(basis, "basis")

    if basis.shape[0] != candidate_embedding.shape[0]:
        raise ValueError(
            f"Dimension mismatch: candidate dim={candidate_embedding.shape[0]}, "
            f"basis dim={basis.shape[0]}"
        )

    working_vector = candidate_embedding.copy()

    if center_with_centroid and centroid is not None:
        working_vector = working_vector - centroid

    projected_vector = project_vector(working_vector, basis)
    residual_vector = working_vector - projected_vector

    residual_norm = float(np.linalg.norm(residual_vector))
    projected_norm = float(np.linalg.norm(projected_vector))
    origin_cost = float(residual_norm ** 2)

    return OriginCostResult(
        origin_cost=origin_cost,
        residual_norm=residual_norm,
        projected_norm=projected_norm,
        residual_vector=residual_vector,
        projected_vector=projected_vector,
    )
