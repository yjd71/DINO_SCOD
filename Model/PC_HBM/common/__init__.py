"""Shared tensor helpers for DINO PC-HBM."""

from .utils import (
    EPS,
    REGION_NAMES,
    REGION_TO_ID,
    add_tokens_to_map,
    boundary_features_from_logits,
    entropy_from_probs,
    finite_or_zero,
    gather_local_patches,
    gather_tokens,
    geometry_map_from_logits,
    gradient_strength,
    js_divergence,
    local_window_gather,
    make_normalized_grid,
    masked_softmax,
    merge_parent_results,
    morph_boundary,
    normalize,
    normalize_prob,
    safe_topk,
    scale_flat_indices,
    scatter_tokens,
    token_indices_from_score,
)

__all__ = [name for name in globals() if not name.startswith("_")]

