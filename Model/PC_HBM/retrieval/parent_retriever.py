"""Chunked, validity-aware parent retrieval from one routed subbank."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from ..common.utils import (
    EPS,
    REGION_TO_ID,
    entropy_from_probs,
    gather_tokens,
    masked_softmax,
    normalize,
    normalize_prob,
)


class ParentRetriever(nn.Module):
    def __init__(
        self,
        p3_ch: int,
        dim: int = 128,
        topk: int = 16,
        tau: float = 0.07,
        value_dim: int = 8,
        geometry_dim: int = 6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.topk = int(topk)
        self.tau = float(tau)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        self.proj_parent_q = nn.Conv2d(int(p3_ch), self.dim, kernel_size=1, bias=False)

    def encode_q_map(self, p3: torch.Tensor) -> torch.Tensor:
        if p3.ndim != 4:
            raise ValueError(f"p3 must be [B,C,H,W], got {tuple(p3.shape)}")
        return normalize(self.proj_parent_q(p3), dim=1)

    def encode_k_map(self, p3: torch.Tensor) -> torch.Tensor:
        """Use the exact same projection for stored keys and online queries."""

        return self.encode_q_map(p3)

    def forward(
        self,
        p3: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        parent_subbank: Mapping[str, Any],
        chunk_size: int = 512,
    ) -> Dict[str, Any]:
        query_map = self.encode_q_map(p3)
        queries = gather_tokens(query_map, batch_ids, flat_indices)
        result = self.retrieve_q(queries, parent_subbank, chunk_size=chunk_size)
        result["q3_map"] = query_map
        return result

    def retrieve_q(
        self,
        q3: torch.Tensor,
        parent_subbank: Mapping[str, Any],
        chunk_size: int = 512,
    ) -> Dict[str, Any]:
        """Retrieve K hypotheses in bounded query chunks.

        Missing K entries are zeros with score ``-1e4``, pointer/index ``-1``
        and ``top_parent_valid=False``.  They never contribute to attention.
        """

        if q3.ndim != 2 or q3.size(1) != self.dim:
            raise ValueError(f"q3 must be [M,{self.dim}], got {tuple(q3.shape)}")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        keys = self._bank_float(parent_subbank, "p3_keys", self.dim, q3)
        values = self._bank_float(parent_subbank, "p3_values", self.value_dim, q3)
        geometry = self._bank_float(parent_subbank, "p3_geometry", self.geometry_dim, q3)
        child_ptr = self._bank_long(parent_subbank, "child_ptr", q3.device)
        if not (keys.size(0) == values.size(0) == geometry.size(0) == child_ptr.numel()):
            raise ValueError("Parent subbank tensor lengths do not match")
        query_count = q3.size(0)
        candidate_count = keys.size(0)
        if query_count == 0 or candidate_count == 0:
            return self._empty(q3)

        real_k = min(self.topk, candidate_count)
        score_chunks: list[torch.Tensor] = []
        local_index_chunks: list[torch.Tensor] = []
        normalized_keys = normalize(keys, dim=-1)
        for start in range(0, query_count, int(chunk_size)):
            stop = min(query_count, start + int(chunk_size))
            similarity = normalize(q3[start:stop], dim=-1) @ normalized_keys.transpose(0, 1)
            score, local_indices = torch.topk(similarity, k=real_k, dim=1)
            score_chunks.append(score)
            local_index_chunks.append(local_indices)
        real_scores = torch.cat(score_chunks, dim=0)
        real_indices = torch.cat(local_index_chunks, dim=0)

        result = self._empty(q3)
        result["top_parent_scores"][:, :real_k] = real_scores
        result["top_parent_valid"][:, :real_k] = True
        result["top_parent_indices"][:, :real_k] = self._global_parent_indices(
            parent_subbank,
            candidate_count,
            q3.device,
        ).index_select(0, real_indices.reshape(-1)).reshape(query_count, real_k)
        result["top_parent_keys"][:, :real_k] = keys.index_select(
            0, real_indices.reshape(-1)
        ).reshape(query_count, real_k, self.dim)
        result["top_parent_values"][:, :real_k] = values.index_select(
            0, real_indices.reshape(-1)
        ).reshape(query_count, real_k, self.value_dim)
        result["top_parent_geo"][:, :real_k] = geometry.index_select(
            0, real_indices.reshape(-1)
        ).reshape(query_count, real_k, self.geometry_dim)
        result["top_child_ptrs"][:, :real_k] = child_ptr.index_select(
            0, real_indices.reshape(-1)
        ).reshape(query_count, real_k)

        attention = masked_softmax(
            result["top_parent_scores"] / max(self.tau, EPS),
            result["top_parent_valid"],
            dim=1,
        )
        result["A_parent"] = attention
        group = (attention.unsqueeze(-1) * result["top_parent_values"][..., :4]).sum(dim=1)
        result["P3_group"] = normalize_prob(group, dim=1)
        result["S_fg_parent"] = (
            attention * result["top_parent_values"][..., 4]
        ).sum(dim=1, keepdim=True)
        result["S_bg_parent"] = (
            attention * result["top_parent_values"][..., 5]
        ).sum(dim=1, keepdim=True)
        result["M_parent"] = result["S_fg_parent"] - result["S_bg_parent"]
        result["parent_entropy"] = entropy_from_probs(attention, dim=1)
        result["top_parent_reliability"] = (
            result["top_parent_values"][..., 7] * result["top_parent_valid"].to(q3.dtype)
        )
        result["query_valid"] = result["top_parent_valid"].any(dim=1)
        result["top_parent_meta"], result["top_parent_region_ids"] = self._metadata(
            parent_subbank.get("parent_meta", []),
            real_indices,
            query_count,
            real_k,
            q3.device,
        )
        return result

    def _empty(self, q3: torch.Tensor) -> Dict[str, Any]:
        query_count = q3.size(0)
        topk = self.topk
        valid = torch.zeros((query_count, topk), device=q3.device, dtype=torch.bool)
        return {
            "q3": q3,
            "top_parent_keys": q3.new_zeros((query_count, topk, self.dim)),
            "top_parent_values": q3.new_zeros((query_count, topk, self.value_dim)),
            "top_parent_geo": q3.new_zeros((query_count, topk, self.geometry_dim)),
            "top_child_ptrs": torch.full((query_count, topk), -1, device=q3.device, dtype=torch.long),
            "top_parent_indices": torch.full((query_count, topk), -1, device=q3.device, dtype=torch.long),
            "top_parent_scores": q3.new_full((query_count, topk), -1.0e4),
            "top_parent_valid": valid,
            "query_valid": torch.zeros(query_count, device=q3.device, dtype=torch.bool),
            "A_parent": q3.new_zeros((query_count, topk)),
            "P3_group": q3.new_zeros((query_count, 4)),
            "S_fg_parent": q3.new_zeros((query_count, 1)),
            "S_bg_parent": q3.new_zeros((query_count, 1)),
            "M_parent": q3.new_zeros((query_count, 1)),
            "parent_entropy": q3.new_zeros(query_count),
            "top_parent_meta": [[{} for _ in range(topk)] for _ in range(query_count)],
            "top_parent_region_ids": torch.full(
                (query_count, topk), -1, device=q3.device, dtype=torch.long
            ),
            "top_parent_reliability": q3.new_zeros((query_count, topk)),
        }

    @staticmethod
    def _bank_float(
        bank: Mapping[str, Any],
        key: str,
        width: int,
        query: torch.Tensor,
    ) -> torch.Tensor:
        value = bank.get(key)
        if value is None:
            return query.new_empty((0, width))
        value = value.to(device=query.device, dtype=query.dtype, non_blocking=True)
        if value.ndim != 2 or value.size(1) != width:
            raise ValueError(f"{key} must be [N,{width}], got {tuple(value.shape)}")
        return value

    @staticmethod
    def _bank_long(bank: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
        value = bank.get(key)
        if value is None:
            return torch.empty(0, device=device, dtype=torch.long)
        return value.to(device=device, dtype=torch.long, non_blocking=True).view(-1)

    @staticmethod
    def _global_parent_indices(
        bank: Mapping[str, Any],
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = bank.get("parent_indices")
        if indices is None:
            return torch.arange(count, device=device, dtype=torch.long)
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if indices.numel() != count:
            raise ValueError("parent_indices length must match parent subbank")
        return indices

    def _metadata(
        self,
        metadata: Any,
        indices: torch.Tensor,
        query_count: int,
        real_k: int,
        device: torch.device,
    ) -> tuple[list[list[dict[str, Any]]], torch.Tensor]:
        rows = [[{} for _ in range(self.topk)] for _ in range(query_count)]
        region_ids = torch.full((query_count, self.topk), -1, device=device, dtype=torch.long)
        if not isinstance(metadata, list):
            return rows, region_ids
        for row_index, row in enumerate(indices.detach().cpu().tolist()):
            for column_index, item_index in enumerate(row[:real_k]):
                item = dict(metadata[item_index]) if item_index < len(metadata) else {}
                rows[row_index][column_index] = item
                raw_region = item.get("region_id", REGION_TO_ID.get(str(item.get("region", "")), -1))
                try:
                    region_ids[row_index, column_index] = int(raw_region)
                except (TypeError, ValueError):
                    region_ids[row_index, column_index] = -1
        return rows, region_ids


__all__ = ["ParentRetriever"]

