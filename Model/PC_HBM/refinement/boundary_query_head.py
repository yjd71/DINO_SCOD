"""Parameter-free boundary query selection for PC-HBM-Lite."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryQuerySelector(nn.Module):
    """Select bounded P3 queries from morphology and probability uncertainty."""

    def __init__(
        self,
        top_ratio: float = 0.10,
        min_tokens: int = 16,
        max_tokens: int = 64,
        boundary_weight: float = 0.5,
        uncertainty_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.top_ratio = float(top_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.boundary_weight = float(boundary_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        if not math.isfinite(self.top_ratio) or not 0.0 <= self.top_ratio <= 1.0:
            raise ValueError("top_ratio must be finite and in [0, 1]")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be >= min_tokens")
        if (
            not math.isfinite(self.boundary_weight)
            or not math.isfinite(self.uncertainty_weight)
            or self.boundary_weight < 0.0
            or self.uncertainty_weight < 0.0
            or max(self.boundary_weight, self.uncertainty_weight) <= 0.0
        ):
            raise ValueError(
                "query weights must be finite, non-negative, and not both zero"
            )
        weight_scale = max(self.boundary_weight, self.uncertainty_weight)
        scaled_boundary = self.boundary_weight / weight_scale
        scaled_uncertainty = self.uncertainty_weight / weight_scale
        scaled_total = scaled_boundary + scaled_uncertainty
        self._boundary_mix = scaled_boundary / scaled_total
        self._uncertainty_mix = scaled_uncertainty / scaled_total

    def forward(
        self,
        probability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if probability.ndim != 4 or probability.size(1) != 1:
            raise ValueError(
                f"probability must be [B,1,H,W], got {tuple(probability.shape)}"
            )
        work = torch.nan_to_num(
            probability.float(), nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        maximum = F.max_pool2d(work, kernel_size=3, stride=1, padding=1)
        minimum = -F.max_pool2d(-work, kernel_size=3, stride=1, padding=1)
        morph_boundary = (maximum - minimum).clamp(0.0, 1.0)
        uncertainty = (4.0 * work * (1.0 - work)).clamp(0.0, 1.0)
        score = (
            self._boundary_mix * morph_boundary
            + self._uncertainty_mix * uncertainty
        )

        flat_score = score.flatten(2)[:, 0]
        batch_ids: list[torch.Tensor] = []
        flat_indices: list[torch.Tensor] = []
        token_scores: list[torch.Tensor] = []
        for batch_index in range(score.size(0)):
            candidates = torch.arange(
                flat_score.size(1),
                device=score.device,
                dtype=torch.long,
            )
            count = max(
                self.min_tokens,
                int(round(candidates.numel() * self.top_ratio)),
            )
            count = min(count, self.max_tokens, int(candidates.numel()))
            candidate_scores = flat_score[batch_index].index_select(
                0, candidates
            )
            if count == 0:
                selected = candidates[:0]
            else:
                local = torch.topk(candidate_scores, k=count, dim=0).indices
                selected = candidates.index_select(0, local)
            batch_ids.append(
                torch.full(
                    (count,),
                    batch_index,
                    device=score.device,
                    dtype=torch.long,
                )
            )
            flat_indices.append(selected)
            token_scores.append(
                flat_score[batch_index].index_select(0, selected)
            )

        if batch_ids:
            selected_batch_ids = torch.cat(batch_ids)
            selected_flat_indices = torch.cat(flat_indices)
            selected_scores = torch.cat(token_scores).unsqueeze(-1)
        else:
            selected_batch_ids = torch.empty(
                0, device=score.device, dtype=torch.long
            )
            selected_flat_indices = torch.empty(
                0, device=score.device, dtype=torch.long
            )
            selected_scores = torch.empty(
                (0, 1), device=score.device, dtype=torch.float32
            )
        return score.to(dtype=probability.dtype), {
            "batch_ids": selected_batch_ids,
            "flat_indices": selected_flat_indices,
            "token_scores": selected_scores,
        }
