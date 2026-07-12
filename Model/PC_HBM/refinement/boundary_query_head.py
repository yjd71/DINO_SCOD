"""Boundary scoring and bounded per-image token selection."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ..common.utils import finite_or_zero


def _select_tokens(
    score: torch.Tensor,
    *,
    top_ratio: float,
    threshold: float | None,
    min_tokens: int,
    max_tokens: int | None,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = score.shape
    flat_score = score.flatten(2)[:, 0]
    if valid_mask is None:
        flat_valid = torch.ones_like(flat_score, dtype=torch.bool)
    else:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[:, None]
        if valid_mask.shape != score.shape:
            raise ValueError("valid_mask must be [B,1,H,W] and match score")
        flat_valid = valid_mask.flatten(2)[:, 0].to(
            device=score.device, dtype=torch.bool
        )
    batches: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        candidates = torch.nonzero(
            flat_valid[batch_index], as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            continue
        candidate_scores = flat_score[batch_index].index_select(0, candidates)
        default_count = max(
            int(min_tokens),
            int(round(candidates.numel() * float(top_ratio))),
        )
        if max_tokens is not None:
            default_count = min(default_count, int(max_tokens))
        default_count = min(default_count, candidates.numel())
        if threshold is None:
            local = torch.topk(candidate_scores, k=default_count).indices
        else:
            local = torch.nonzero(
                candidate_scores >= float(threshold), as_tuple=False
            ).flatten()
            if local.numel() < min(int(min_tokens), candidates.numel()):
                local = torch.topk(candidate_scores, k=default_count).indices
            elif max_tokens is not None and local.numel() > int(max_tokens):
                order = torch.topk(
                    candidate_scores.index_select(0, local), k=int(max_tokens)
                ).indices
                local = local.index_select(0, order)
        selected = candidates.index_select(0, local)
        batches.append(
            torch.full(
                (selected.numel(),),
                batch_index,
                device=score.device,
                dtype=torch.long,
            )
        )
        indices.append(selected.long())
        values.append(flat_score[batch_index].index_select(0, selected))
    if not batches:
        empty_index = torch.empty(0, device=score.device, dtype=torch.long)
        return empty_index, empty_index, score.new_empty(0)
    return torch.cat(batches), torch.cat(indices), torch.cat(values)


class BoundaryQueryHead(nn.Module):
    """Small convolutional boundary scorer with bounded selection."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 32,
        top_ratio: float = 0.25,
        min_tokens: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if int(hidden_ch) % 4 != 0:
            raise ValueError("hidden_ch must be divisible by four")
        self.top_ratio = float(top_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(hidden_ch), 3, padding=1),
            nn.GroupNorm(4, int(hidden_ch)),
            nn.GELU(),
            nn.Conv2d(int(hidden_ch), int(hidden_ch), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_ch), 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        threshold: float | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        score = torch.sigmoid(finite_or_zero(self.net(x)))
        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[:, None]
            score = score * valid_mask.to(device=score.device, dtype=score.dtype)
        batch_ids, flat_indices, token_scores = _select_tokens(
            score,
            top_ratio=self.top_ratio,
            threshold=threshold,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            valid_mask=valid_mask,
        )
        return score, {
            "batch_ids": batch_ids,
            "flat_indices": flat_indices,
            "token_scores": token_scores,
            "query_valid": torch.ones_like(batch_ids, dtype=torch.bool),
            "height": torch.tensor(score.size(2), device=score.device),
            "width": torch.tensor(score.size(3), device=score.device),
        }


class BoundaryQueryHead3(BoundaryQueryHead):
    """Five-channel p3 boundary scorer on the 28 x 28 grid."""

    def __init__(
        self,
        top_ratio: float = 0.20,
        min_tokens: int = 32,
        max_tokens: int | None = 128,
    ) -> None:
        super().__init__(
            5,
            top_ratio=top_ratio,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )


class BoundaryQueryHead2(BoundaryQueryHead):
    """Eight-channel p2 boundary scorer on the 28 x 28 grid."""

    def __init__(
        self,
        top_ratio: float = 0.20,
        min_tokens: int = 32,
        max_tokens: int | None = 128,
    ) -> None:
        super().__init__(
            8,
            top_ratio=top_ratio,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )


class BoundaryQueryHead1(BoundaryQueryHead):
    """Eight-channel p1 boundary scorer on the 98 x 98 grid."""

    def __init__(
        self,
        top_ratio: float = 0.05,
        min_tokens: int = 96,
        max_tokens: int | None = 384,
    ) -> None:
        super().__init__(
            8,
            top_ratio=top_ratio,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

