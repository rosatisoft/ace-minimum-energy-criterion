from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


@dataclass
class OriginCostResult:
    origin_cost: float
    residual_norm: float
    projected_norm: float
    residual_vector: Vector
    projected_vector: Vector


@dataclass
class SubspaceResult:
    basis: Matrix
    singular_values: Vector
    rank: int
    centroid: Optional[Vector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateScore:
    origin_cost: float
    residual_norm: float
    projected_norm: float
    total_energy: Optional[float] = None
    candidate_label: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBundle:
    prompt_embedding: Vector
    axiom_embeddings: List[Vector]
    knowledge_embeddings: List[Vector]
    metadata: Dict[str, Any] = field(default_factory=dict)
