"""Balanced two-region parent retrieval for PC-HBM-Lite."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..common.utils import gather_tokens


class BalancedParentRetriever(nn.Module):
    """Retrieve a fixed quota independently from both binary memory regions.

    Region ``0`` is ``fg_boundary`` and region ``1`` is ``bg_near``.  The
    regions are never merged before Top-K, which prevents a large/easy region
    from suppressing the other side of the binary evidence.
    """

    REGION_COUNT = 2

    def __init__(
        self,
        p3_ch: int | None = None,
        dim: int = 128,
        topk_per_region: int = 4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.p3_ch = self.dim if p3_ch is None else int(p3_ch)
        self.topk_per_region = int(topk_per_region)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.p3_ch <= 0:
            raise ValueError("p3_ch must be positive")
        if self.topk_per_region <= 0:
            raise ValueError("topk_per_region must be positive")
        self.proj_parent_q = nn.Conv2d(
            self.p3_ch,
            self.dim,
            kernel_size=1,
            bias=False,
        )

    def encode_q_map(self, p3: torch.Tensor) -> torch.Tensor:
        """Project P3 with the shared 1x1 key map, then normalize in FP32."""

        if p3.ndim != 4 or p3.size(1) != self.p3_ch:
            raise ValueError(
                f"p3 must be [B,{self.p3_ch},H,W], got {tuple(p3.shape)}"
            )
        projected = torch.nan_to_num(
            self.proj_parent_q(p3).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        encoded = F.normalize(
            projected,
            dim=1,
            eps=1.0e-6,
        )
        return encoded.to(dtype=p3.dtype)

    def encode_k_map(self, p3: torch.Tensor) -> torch.Tensor:
        return self.encode_q_map(p3)

    def forward(
        self,
        p3: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        pair_subbank: Mapping[str, Any],
        chunk_size: int = 512,
    ) -> dict[str, torch.Tensor]:
        q3_map = self.encode_q_map(p3)
        q3 = gather_tokens(q3_map, batch_ids, flat_indices)
        result = self.retrieve_q(q3, pair_subbank, chunk_size=chunk_size)
        result["q3_map"] = q3_map
        return result

    def retrieve_q(
        self,
        q3: torch.Tensor,
        pair_subbank: Mapping[str, Any],
        chunk_size: int = 512,
    ) -> dict[str, torch.Tensor]:
        if q3.ndim != 2 or q3.size(1) != self.dim:
            raise ValueError(f"q3 must be [M,{self.dim}], got {tuple(q3.shape)}")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")

        p3_keys = self._float_bank(pair_subbank, "p3_keys", q3)
        p2_keys = self._float_bank(pair_subbank, "p2_keys", q3)
        region_ids = self._long_bank(pair_subbank, "region_ids", q3.device)
        pair_indices = self._pair_indices(
            pair_subbank, p3_keys.size(0), q3.device
        )
        candidate_count = p3_keys.size(0)
        if not (
            p2_keys.size(0)
            == region_ids.numel()
            == pair_indices.numel()
            == candidate_count
        ):
            raise ValueError("Pair subbank tensor lengths do not match")
        if region_ids.numel() and not bool(
            ((region_ids == 0) | (region_ids == 1)).all()
        ):
            raise ValueError("region_ids must contain only 0 or 1")

        result = self.empty_result(q3)
        if q3.size(0) == 0 or candidate_count == 0:
            return result

        normalized_queries = F.normalize(q3.float(), dim=-1, eps=1.0e-6)
        normalized_keys = F.normalize(p3_keys.float(), dim=-1, eps=1.0e-6)
        for region_id in range(self.REGION_COUNT):
            candidate_indices = torch.nonzero(
                region_ids == region_id, as_tuple=False
            ).flatten()
            region_count = int(candidate_indices.numel())
            if region_count == 0:
                continue
            real_k = min(self.topk_per_region, region_count)
            region_keys = normalized_keys.index_select(0, candidate_indices)
            scores: list[torch.Tensor] = []
            local_indices: list[torch.Tensor] = []
            for start in range(0, q3.size(0), int(chunk_size)):
                stop = min(q3.size(0), start + int(chunk_size))
                similarity = (
                    normalized_queries[start:stop]
                    @ region_keys.transpose(0, 1)
                )
                chunk_scores, chunk_indices = torch.topk(
                    similarity, k=real_k, dim=1
                )
                scores.append(chunk_scores)
                local_indices.append(chunk_indices)
            top_scores = torch.cat(scores, dim=0)
            top_local = torch.cat(local_indices, dim=0)
            top_bank = candidate_indices.index_select(
                0, top_local.reshape(-1)
            ).reshape(q3.size(0), real_k)

            result["scores"][:, region_id, :real_k] = top_scores
            result["valid"][:, region_id, :real_k] = True
            result["indices"][:, region_id, :real_k] = pair_indices.index_select(
                0, top_bank.reshape(-1)
            ).reshape(q3.size(0), real_k)
            result["parent_keys"][:, region_id, :real_k] = p3_keys.index_select(
                0, top_bank.reshape(-1)
            ).reshape(q3.size(0), real_k, self.dim)
            result["paired_p2_keys"][:, region_id, :real_k] = p2_keys.index_select(
                0, top_bank.reshape(-1)
            ).reshape(q3.size(0), real_k, self.dim)

        result["query_valid"] = result["valid"].any(dim=-1).all(dim=-1)
        return result

    def empty_result(self, q3: torch.Tensor) -> dict[str, torch.Tensor]:
        query_count = q3.size(0)
        shape = (
            query_count,
            self.REGION_COUNT,
            self.topk_per_region,
        )
        return {
            "q3": q3,
            "parent_keys": q3.new_zeros((*shape, self.dim)),
            "paired_p2_keys": q3.new_zeros((*shape, self.dim)),
            "scores": torch.full(
                shape, -1.0e4, device=q3.device, dtype=torch.float32
            ),
            "indices": torch.full(
                shape, -1, device=q3.device, dtype=torch.long
            ),
            "valid": torch.zeros(
                shape, device=q3.device, dtype=torch.bool
            ),
            "query_valid": torch.zeros(
                query_count, device=q3.device, dtype=torch.bool
            ),
        }

    def _float_bank(
        self,
        bank: Mapping[str, Any],
        key: str,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if key not in bank:
            raise KeyError(f"Pair subbank is missing {key!r}")
        value = bank[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim != 2 or value.size(1) != self.dim:
            raise ValueError(
                f"{key} must be [N,{self.dim}], got {tuple(value.shape)}"
            )
        return value.detach().to(
            device=query.device, dtype=query.dtype, non_blocking=True
        )

    @staticmethod
    def _long_bank(
        bank: Mapping[str, Any],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        if key not in bank:
            raise KeyError(f"Pair subbank is missing {key!r}")
        value = bank[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim != 1:
            raise ValueError(f"{key} must be [N], got {tuple(value.shape)}")
        return value.detach().to(device=device, dtype=torch.long, non_blocking=True)

    @staticmethod
    def _pair_indices(
        bank: Mapping[str, Any],
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = bank.get("pair_indices")
        if value is None:
            return torch.arange(count, device=device, dtype=torch.long)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim != 1 or value.numel() != count:
            raise ValueError("pair_indices must be [N] and align with pair keys")
        return value.detach().to(device=device, dtype=torch.long, non_blocking=True)
