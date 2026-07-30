"""Labeled-only PC-HBM-Lite memory components."""

from .pc_memory import CompatibilityResult, PCMemory
from .pc_region_builder import (
    REGION_BG_NEAR,
    REGION_FG_BOUNDARY,
    REGION_IGNORE,
    build_boundary_pair_regions,
)
from .sampling_policy import (
    DEFAULT_REGION_SAMPLING,
    MAX_QUOTA,
    MIN_QUOTA,
    REGION_NAMES,
    SAMPLING_RATIO,
    RegionSamplingRule,
    rules_from_config,
    sample_region_indices,
)

__all__ = [name for name in globals() if not name.startswith("_")]
