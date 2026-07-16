from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


SCORE_FORMULA_VERSION = "pc_bacs_v2_replicate_hypot_eps_denominator_only"


@dataclass(frozen=True)
class PCBACSConfig:
    """Frozen, reproducible configuration for offline PC-BACS selection."""

    input_size: int = 392
    output_size: int = 98
    feature_dim: int = 768
    decoder_arch: str = "legacy_transformer"

    n_clusters: int = 40
    target_counts: Tuple[int, ...] = (41, 202, 404)
    random_seed: int = 2025
    selector_seed_count: int = 40

    feature_batch_size: int = 16
    score_batch_size: int = 16
    num_workers: int = 8
    use_amp: bool = True

    dedup_threshold: float = 0.98
    eps: float = 1e-6
    score_formula_version: str = SCORE_FORMULA_VERSION

    def validate(self, sample_count: int | None = None) -> None:
        if self.decoder_arch != "legacy_transformer":
            raise ValueError("PC-BACS artifacts are locked to decoder_arch='legacy_transformer'.")
        if self.input_size != 392:
            raise ValueError("PC-BACS must use the current DINO_SCOD input size 392.")
        if self.output_size != 98:
            raise ValueError("PC-BACS score must be computed at decoder output size 98.")
        if self.feature_dim != 768:
            raise ValueError("PC-BACS expects 768-dimensional DINOv2-ViT-B/14 features.")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if self.selector_seed_count <= 0:
            raise ValueError("selector_seed_count must be positive.")
        if not self.target_counts:
            raise ValueError("target_counts must not be empty.")
        if any(isinstance(count, bool) or not isinstance(count, int) for count in self.target_counts):
            raise TypeError("target_counts must contain integers.")
        if any(count <= 0 for count in self.target_counts):
            raise ValueError("target counts must be positive.")
        if tuple(sorted(set(self.target_counts))) != self.target_counts:
            raise ValueError("target_counts must be strictly increasing.")
        if self.selector_seed_count > self.target_counts[0]:
            raise ValueError("selector_seed_count exceeds the smallest target count.")
        if self.feature_batch_size <= 0 or self.score_batch_size <= 0:
            raise ValueError("feature and score batch sizes must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive.")
        if self.dedup_threshold >= 0.0 and not 0.0 < self.dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold must be in (0, 1] or negative to disable.")
        if self.score_formula_version != SCORE_FORMULA_VERSION:
            raise ValueError(
                "score_formula_version must identify the replicate-padding/hypot formula."
            )

        if sample_count is not None:
            if isinstance(sample_count, bool) or not isinstance(sample_count, int):
                raise TypeError("sample_count must be an integer or None.")
            if sample_count <= 0:
                raise ValueError("sample_count must be positive.")
            if self.n_clusters > sample_count:
                raise ValueError("n_clusters exceeds sample count.")
            if self.selector_seed_count > sample_count:
                raise ValueError("selector_seed_count exceeds sample count.")
            if self.target_counts[-1] > sample_count:
                raise ValueError("largest target exceeds sample count.")


__all__ = ["PCBACSConfig", "SCORE_FORMULA_VERSION"]
