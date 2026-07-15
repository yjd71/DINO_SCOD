"""KMeans-free boundary prototype utility sampling primitives."""

from .boundary_score import (
    BOUNDARY_SCORE_VERSION,
    SCORE_FORMULA_VERSION,
    BoundaryUtility,
    ScorePrototypeResult,
    compute_boundary_utility,
    score_and_prototype,
    sobel_magnitude,
)
from .cache import (
    PROTOTYPE_PAYLOAD_TYPE,
    SCORE_PAYLOAD_TYPE,
    build_prototype_payload,
    build_prototype_cache,
    build_score_cache,
    build_score_payload,
    validate_prototype_cache,
    validate_prototype_payload,
    validate_score_cache,
    validate_score_payload,
)
from .greedy_acquisition import AcquisitionResult, greedy_acquire
from .prototype import PROTOTYPE_VERSION, build_boundary_prototype

__all__ = [
    "AcquisitionResult",
    "BOUNDARY_SCORE_VERSION",
    "BoundaryUtility",
    "PROTOTYPE_PAYLOAD_TYPE",
    "PROTOTYPE_VERSION",
    "SCORE_FORMULA_VERSION",
    "SCORE_PAYLOAD_TYPE",
    "ScorePrototypeResult",
    "build_boundary_prototype",
    "build_prototype_cache",
    "build_prototype_payload",
    "build_score_cache",
    "build_score_payload",
    "compute_boundary_utility",
    "greedy_acquire",
    "score_and_prototype",
    "sobel_magnitude",
    "validate_prototype_payload",
    "validate_prototype_cache",
    "validate_score_cache",
    "validate_score_payload",
]
