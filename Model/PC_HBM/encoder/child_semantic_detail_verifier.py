from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.utils import gather_local_patches, gather_tokens, masked_softmax
from .encoder_memory import EncoderPCMemory


MEMORY_DIM = 128
VALUE_DIM = 8
GEOMETRY_DIM = 6
PARENT_TOPK = 16
QUERY_CHUNK_SIZE = 512
INVALID_SCORE = -1.0e4


class _LocalPatchEncoder(nn.Module):
    """Encode one fixed-size projected-DINO patch into a 128-D query."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = int(window)
        if self.window not in (3, 5):
            raise ValueError("Encoder child verification windows are fixed to 3 or 5")
        self.net = nn.Sequential(
            nn.Conv2d(MEMORY_DIM, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, MEMORY_DIM, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(MEMORY_DIM, MEMORY_DIM)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        expected = (MEMORY_DIM, self.window, self.window)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"patches must be [M,{MEMORY_DIM},{self.window},{self.window}], "
                f"got {tuple(patches.shape)}"
            )
        if patches.size(0) == 0:
            return patches.new_empty((0, MEMORY_DIM))
        return F.normalize(self.projection(self.net(patches).flatten(1)), dim=-1)


class _PairSupportScorer(nn.Module):
    """Score query/candidate pairs using the prescribed four-way pairing."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MEMORY_DIM * 4, MEMORY_DIM),
            nn.GELU(),
            nn.Linear(MEMORY_DIM, 1),
        )

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or query.size(1) != MEMORY_DIM:
            raise ValueError(f"query must be [M,{MEMORY_DIM}], got {tuple(query.shape)}")
        if candidates.ndim != 3 or candidates.shape[0] != query.shape[0] or candidates.size(2) != MEMORY_DIM:
            raise ValueError(
                f"candidates must be [M,K,{MEMORY_DIM}], got {tuple(candidates.shape)}"
            )
        normalized_query = F.normalize(query, dim=-1)
        normalized_candidates = F.normalize(candidates, dim=-1)
        expanded = normalized_query.unsqueeze(1).expand_as(normalized_candidates)
        pair = torch.cat(
            (
                expanded,
                normalized_candidates,
                expanded * normalized_candidates,
                torch.abs(expanded - normalized_candidates),
            ),
            dim=-1,
        )
        return self.net(pair).squeeze(-1)


class _GeometrySupportScorer(nn.Module):
    """Score SDF, normal, boundary-offset, and reliability agreement."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        query_geometry: torch.Tensor,
        parent_geometry: torch.Tensor,
        child_geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if parent_geometry.shape != child_geometry.shape or parent_geometry.ndim != 3:
            raise ValueError("parent and child geometry must share [M,K,6]")
        if parent_geometry.size(2) != GEOMETRY_DIM:
            raise ValueError("geometry vectors must have width 6")
        if query_geometry.shape != (parent_geometry.size(0), GEOMETRY_DIM):
            raise ValueError(
                f"query_geometry must be [M,{GEOMETRY_DIM}], got {tuple(query_geometry.shape)}"
            )

        query = query_geometry.unsqueeze(1).expand_as(parent_geometry)
        sdf_difference = torch.abs(query[..., 0] - child_geometry[..., 0])
        normal_cosine = F.cosine_similarity(
            query[..., 1:3], child_geometry[..., 1:3], dim=-1, eps=1.0e-6
        )
        offset_difference = torch.abs(
            query[..., 3:5] - child_geometry[..., 3:5]
        ).mean(dim=-1)
        reliability_product = (
            query[..., 5].clamp(0.0, 1.0)
            * child_geometry[..., 5].clamp(0.0, 1.0)
            * parent_geometry[..., 5].clamp(0.0, 1.0)
        )
        # The geometric mean has an infinite derivative at an exact zero.
        # Zero reliability is a normal boundary case (especially under AMP),
        # so evaluate the root at a finite placeholder and mask both its value
        # and gradient back to strict zero for those entries.
        positive_reliability = reliability_product > 0
        safe_product = torch.where(
            positive_reliability,
            reliability_product,
            torch.ones_like(reliability_product),
        )
        geometry_reliability = torch.where(
            positive_reliability,
            safe_product.pow(1.0 / 3.0),
            torch.zeros_like(reliability_product),
        )

        parent_child_sdf_difference = torch.abs(
            parent_geometry[..., 0] - child_geometry[..., 0]
        )
        parent_child_normal_cosine = F.cosine_similarity(
            parent_geometry[..., 1:3],
            child_geometry[..., 1:3],
            dim=-1,
            eps=1.0e-6,
        )
        parent_child_offset_difference = torch.abs(
            parent_geometry[..., 3:5] - child_geometry[..., 3:5]
        ).mean(dim=-1)
        features = torch.stack(
            (
                -sdf_difference,
                normal_cosine,
                -offset_difference,
                geometry_reliability,
                -parent_child_sdf_difference,
                parent_child_normal_cosine,
                -parent_child_offset_difference,
                parent_geometry[..., 5].clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        return {
            "score": self.net(features).squeeze(-1),
            "sdf_difference": sdf_difference,
            "normal_cosine": normal_cosine,
            "offset_difference": offset_difference,
            "geometry_reliability": geometry_reliability,
        }


def _normalize_score(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked per-query z-normalization followed by a bounded sigmoid."""

    if score.shape != valid.shape:
        raise ValueError("score and valid mask shapes must match")
    mask = valid.to(dtype=score.dtype)
    count = mask.sum(dim=-1, keepdim=True)
    mean = (score * mask).sum(dim=-1, keepdim=True) / count.clamp_min(1.0)
    centered = (score - mean) * mask
    variance = centered.square().sum(dim=-1, keepdim=True) / count.clamp_min(1.0)
    normalized = torch.sigmoid(centered / torch.sqrt(variance + 1.0e-6))
    return torch.where(valid, normalized, torch.zeros_like(normalized))


class NormalizedStructuredPrior(nn.Module):
    """Interpretable normalized prior with a zero-gated learned residual."""

    def __init__(self) -> None:
        super().__init__()
        self.a_parent = nn.Parameter(torch.tensor(0.0))
        self.a_semantic = nn.Parameter(torch.tensor(0.0))
        self.a_detail = nn.Parameter(torch.tensor(0.0))
        self.a_geometry = nn.Parameter(torch.tensor(0.0))
        self.a_contradiction = nn.Parameter(torch.tensor(0.0))
        self.prior_residual_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.gamma_prior = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        parent_scores: torch.Tensor,
        semantic_scores: torch.Tensor,
        detail_scores: torch.Tensor,
        geometry_scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        for name, value in (
            ("semantic_scores", semantic_scores),
            ("detail_scores", detail_scores),
            ("geometry_scores", geometry_scores),
            ("valid", valid),
        ):
            if value.shape != parent_scores.shape:
                raise ValueError(f"{name} must match parent score shape")
        valid = valid.bool()
        parent_n = _normalize_score(parent_scores, valid)
        semantic_n = _normalize_score(semantic_scores, valid)
        detail_n = _normalize_score(detail_scores, valid)
        geometry_n = _normalize_score(geometry_scores, valid)
        contradiction = (
            torch.abs(parent_n - semantic_n) + torch.abs(semantic_n - detail_n)
        ) / 2.0
        contradiction = torch.where(valid, contradiction, torch.zeros_like(contradiction))
        prior_base = (
            F.softplus(self.a_parent) * parent_n
            + F.softplus(self.a_semantic) * semantic_n
            + F.softplus(self.a_detail) * detail_n
            + F.softplus(self.a_geometry) * geometry_n
            - F.softplus(self.a_contradiction) * contradiction
        )
        prior_input = torch.stack(
            (parent_n, semantic_n, detail_n, geometry_n, contradiction), dim=-1
        )
        prior_residual = self.prior_residual_mlp(prior_input).squeeze(-1)
        prior_bias = prior_base + torch.tanh(self.gamma_prior) * prior_residual
        zero = torch.zeros((), device=prior_bias.device, dtype=prior_bias.dtype)
        return {
            "parent_normalized": parent_n,
            "semantic_normalized": semantic_n,
            "detail_normalized": detail_n,
            "geometry_normalized": geometry_n,
            "contradiction": contradiction,
            "prior_base": torch.where(valid, prior_base, zero),
            "prior_residual": torch.where(valid, prior_residual, zero),
            "prior_bias": torch.where(valid, prior_bias, zero),
        }


class EncoderParentRetriever(nn.Module):
    """Retrieve F3 parents independently from every routed image subbank."""

    def __init__(
        self,
        dim: int = MEMORY_DIM,
        *,
        topk: int = PARENT_TOPK,
        query_chunk_size: int = QUERY_CHUNK_SIZE,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if int(dim) != MEMORY_DIM:
            raise ValueError("Encoder parent vectors are fixed to 128 dimensions")
        if not 1 <= int(topk) <= PARENT_TOPK:
            raise ValueError("Encoder parent retrieval top-k must be in [1,16]")
        if int(query_chunk_size) <= 0:
            raise ValueError("query_chunk_size must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("parent temperature must be positive")
        self.dim = MEMORY_DIM
        self.topk = int(topk)
        self.query_chunk_size = int(query_chunk_size)
        self.temperature = float(temperature)

    @staticmethod
    def encode_query_map(e3_map: torch.Tensor) -> torch.Tensor:
        if e3_map.ndim != 4 or e3_map.size(1) != MEMORY_DIM:
            raise ValueError(f"e3_map must be [B,{MEMORY_DIM},H,W]")
        return F.normalize(e3_map, dim=1)

    encode_key_map = encode_query_map

    def forward(
        self,
        e3_map: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        parent_subbanks: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, Any]:
        if len(parent_subbanks) != e3_map.size(0):
            raise ValueError("parent_subbanks must contain one independent bank per batch item")
        q3 = gather_tokens(
            self.encode_query_map(e3_map), batch_ids, flat_indices
        )
        return self.retrieve_q(q3, batch_ids, parent_subbanks)

    def retrieve_q(
        self,
        q3: torch.Tensor,
        batch_ids: torch.Tensor,
        parent_subbanks: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, Any]:
        if q3.ndim != 2 or q3.size(1) != MEMORY_DIM:
            raise ValueError(f"q3 must be [M,{MEMORY_DIM}], got {tuple(q3.shape)}")
        if batch_ids.ndim != 1 or batch_ids.numel() != q3.size(0):
            raise ValueError("batch_ids must be [M] and align with q3")
        if batch_ids.numel() and (
            int(batch_ids.min()) < 0 or int(batch_ids.max()) >= len(parent_subbanks)
        ):
            raise IndexError("batch_ids reference a missing routed parent subbank")

        result = self._empty(q3)
        reasons = ["empty_routed_parent_subbank" for _ in range(q3.size(0))]
        for batch_index, subbank in enumerate(parent_subbanks):
            query_positions = torch.nonzero(
                batch_ids.to(device=q3.device, dtype=torch.long) == batch_index,
                as_tuple=False,
            ).flatten()
            if query_positions.numel() == 0:
                continue
            candidate_count = self._candidate_count(subbank)
            if candidate_count == 0:
                continue
            fields = self._prepare_subbank(subbank, q3, candidate_count)
            query = q3.index_select(0, query_positions)
            real_k = min(self.topk, candidate_count)
            local_chunks: list[torch.Tensor] = []
            score_chunks: list[torch.Tensor] = []
            keys = F.normalize(fields["f3_parent_keys"], dim=-1)
            for start in range(0, query.size(0), self.query_chunk_size):
                stop = min(query.size(0), start + self.query_chunk_size)
                similarity = F.normalize(query[start:stop], dim=-1) @ keys.transpose(0, 1)
                scores, local_indices = torch.topk(similarity, k=real_k, dim=-1)
                score_chunks.append(scores)
                local_chunks.append(local_indices)
            scores = torch.cat(score_chunks, dim=0)
            local_indices = torch.cat(local_chunks, dim=0)
            self._copy_retrieval(
                result,
                query_positions,
                scores,
                local_indices,
                fields,
                real_k,
            )
            for position in query_positions.tolist():
                reasons[position] = ""

        valid = result["top_parent_valid"]
        attention = masked_softmax(
            result["top_parent_scores"] / self.temperature, valid, dim=-1
        )
        result["parent_attention"] = attention
        result["parent_evidence"] = (
            attention.unsqueeze(-1) * result["top_parent_keys"]
        ).sum(dim=1)
        result["query_valid"] = valid.any(dim=-1)
        result["reason"] = tuple(reasons)
        result["failure_reasons"] = tuple(reasons)
        return result

    def _empty(self, q3: torch.Tensor) -> dict[str, Any]:
        rows = q3.size(0)
        shape = (rows, self.topk)
        return {
            "q3": q3,
            "top_parent_keys": q3.new_zeros((*shape, MEMORY_DIM)),
            "top_parent_values": q3.new_zeros((*shape, VALUE_DIM)),
            "top_parent_geometry": q3.new_zeros((*shape, GEOMETRY_DIM)),
            "top_parent_scores": q3.new_full(shape, INVALID_SCORE),
            "top_parent_valid": torch.zeros(shape, device=q3.device, dtype=torch.bool),
            "top_parent_indices": torch.full(shape, -1, device=q3.device, dtype=torch.long),
            "top_child_ptrs": torch.full(shape, -1, device=q3.device, dtype=torch.long),
            "top_parent_image_indices": torch.full(shape, -1, device=q3.device, dtype=torch.long),
            "top_parent_region_ids": torch.full(shape, -1, device=q3.device, dtype=torch.long),
            "top_parent_flat_indices": torch.full(shape, -1, device=q3.device, dtype=torch.long),
            "top_parent_reliability": q3.new_zeros(shape),
            "parent_attention": q3.new_zeros(shape),
            "parent_evidence": q3.new_zeros((rows, MEMORY_DIM)),
            "query_valid": torch.zeros(rows, device=q3.device, dtype=torch.bool),
            "reason": tuple("empty_routed_parent_subbank" for _ in range(rows)),
            "failure_reasons": tuple("empty_routed_parent_subbank" for _ in range(rows)),
        }

    @staticmethod
    def _candidate_count(subbank: Mapping[str, torch.Tensor]) -> int:
        if not subbank:
            return 0
        indices = subbank.get("global_parent_indices")
        if indices is not None:
            if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
                raise ValueError("global_parent_indices must be a vector")
            return int(indices.numel())
        keys = subbank.get("f3_parent_keys")
        if keys is None:
            return 0
        if not isinstance(keys, torch.Tensor) or keys.ndim != 2:
            raise ValueError("f3_parent_keys must be a matrix")
        return int(keys.size(0))

    @staticmethod
    def _prepare_subbank(
        subbank: Mapping[str, torch.Tensor],
        query: torch.Tensor,
        count: int,
    ) -> dict[str, torch.Tensor]:
        widths = {
            "f3_parent_keys": MEMORY_DIM,
            "values": VALUE_DIM,
            "geometry": GEOMETRY_DIM,
        }
        vectors = (
            "child_ptr",
            "image_index",
            "region_id",
            "flat_index",
            "reliability",
            "global_parent_indices",
        )
        result: dict[str, torch.Tensor] = {}
        for name, width in widths.items():
            value = subbank.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (count, width):
                raise ValueError(f"routed parent subbank {name} must be [{count},{width}]")
            result[name] = value.to(device=query.device, dtype=query.dtype, non_blocking=True)
        for name in vectors:
            value = subbank.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"routed parent subbank {name} must be [{count}]")
            dtype = query.dtype if name == "reliability" else torch.long
            result[name] = value.to(device=query.device, dtype=dtype, non_blocking=True)
        return result

    @staticmethod
    def _copy_retrieval(
        result: dict[str, Any],
        query_positions: torch.Tensor,
        scores: torch.Tensor,
        local_indices: torch.Tensor,
        fields: Mapping[str, torch.Tensor],
        real_k: int,
    ) -> None:
        rows = query_positions.numel()
        flat = local_indices.reshape(-1)

        def gathered(name: str, width: int | None = None) -> torch.Tensor:
            selected = fields[name].index_select(0, flat)
            if width is None:
                return selected.reshape(rows, real_k)
            return selected.reshape(rows, real_k, width)

        # Autocast may produce FP16/BF16 matmul scores even when q3 (and thus
        # the preallocated result) is FP32.  Preserve the result/query dtype;
        # the differentiable cast keeps retrieval gradients intact.
        score_target = result["top_parent_scores"]
        result["top_parent_scores"][query_positions, :real_k] = scores.to(
            dtype=score_target.dtype
        )
        result["top_parent_valid"][query_positions, :real_k] = True
        result["top_parent_keys"][query_positions, :real_k] = gathered(
            "f3_parent_keys", MEMORY_DIM
        )
        result["top_parent_values"][query_positions, :real_k] = gathered(
            "values", VALUE_DIM
        )
        result["top_parent_geometry"][query_positions, :real_k] = gathered(
            "geometry", GEOMETRY_DIM
        )
        result["top_parent_indices"][query_positions, :real_k] = gathered(
            "global_parent_indices"
        )
        result["top_child_ptrs"][query_positions, :real_k] = gathered("child_ptr")
        result["top_parent_image_indices"][query_positions, :real_k] = gathered(
            "image_index"
        )
        result["top_parent_region_ids"][query_positions, :real_k] = gathered("region_id")
        result["top_parent_flat_indices"][query_positions, :real_k] = gathered("flat_index")
        result["top_parent_reliability"][query_positions, :real_k] = gathered(
            "reliability"
        )


def build_support_targets(
    query_region_ids: torch.Tensor,
    candidate_region_ids: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build labeled-only support targets without changing retrieval evidence."""

    if query_region_ids.ndim != 1 or query_region_ids.numel() != candidate_region_ids.size(0):
        raise ValueError("query_region_ids must be [M]")
    if candidate_region_ids.shape != valid.shape or candidate_region_ids.ndim != 2:
        raise ValueError("candidate_region_ids and valid must share [M,K]")
    query_region_ids = query_region_ids.to(
        device=candidate_region_ids.device, dtype=torch.long
    )
    if query_region_ids.numel() and (
        int(query_region_ids.min()) < 0 or int(query_region_ids.max()) > 3
    ):
        raise ValueError("query_region_ids must be in [0,3]")
    query = query_region_ids.unsqueeze(1)
    candidates = candidate_region_ids.long()
    candidate_valid = valid.bool() & (candidates >= 0) & (candidates <= 3)
    query_foreground = query < 2
    candidate_foreground = candidates < 2
    semantic_target = (query_foreground == candidate_foreground) & candidate_valid
    detail_target = (query == candidates) & candidate_valid
    semantic_hard_negative = (
        (((query == 1) & (candidates == 2)) | ((query == 2) & (candidates == 1)))
        & candidate_valid
    )
    detail_hard_negative = (
        semantic_target & ~detail_target
    )
    return {
        "semantic_target": semantic_target,
        "detail_target": detail_target,
        "semantic_hard_negative_mask": semantic_hard_negative,
        "detail_hard_negative_mask": detail_hard_negative,
    }


def _masked_max(
    score: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    present = mask.any(dim=-1)
    masked = score.masked_fill(~mask, INVALID_SCORE)
    value, index = masked.max(dim=-1)
    value = torch.where(present, value, torch.zeros_like(value))
    index = torch.where(present, index, torch.full_like(index, -1))
    return value, index, present


class ChildSemanticDetailVerifier(nn.Module):
    """Verify retrieved F3 parents with F2 semantics, F1 detail, and geometry."""

    def __init__(
        self,
        dim: int = MEMORY_DIM,
        *,
        parent_topk: int = PARENT_TOPK,
    ) -> None:
        super().__init__()
        if int(dim) != MEMORY_DIM:
            raise ValueError("Encoder verification vectors are fixed to 128 dimensions")
        if not 1 <= int(parent_topk) <= PARENT_TOPK:
            raise ValueError("Encoder verification parent_topk must be in [1,16]")
        self.parent_topk = int(parent_topk)
        self.semantic_local_encoder = _LocalPatchEncoder(window=5)
        self.detail_local_encoder = _LocalPatchEncoder(window=3)
        self.semantic_score_mlp = _PairSupportScorer()
        self.detail_score_mlp = _PairSupportScorer()
        self.geometry_score_mlp = _GeometrySupportScorer()
        self.structured_prior = NormalizedStructuredPrior()

    def forward(
        self,
        e1_map: torch.Tensor,
        e2_map: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        parent_result: Mapping[str, Any],
        memory: EncoderPCMemory,
        query_geometry: torch.Tensor,
        *,
        query_region_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._validate_maps(e1_map, e2_map)
        if not memory.is_ready():
            raise RuntimeError("EncoderPCMemory must be finalized before verification")
        parent_keys = self._parent_tensor(parent_result, "top_parent_keys", 3)
        rows, topk, width = parent_keys.shape
        if (topk, width) != (self.parent_topk, MEMORY_DIM):
            raise ValueError(
                f"top_parent_keys must be [M,{self.parent_topk},{MEMORY_DIM}]"
            )
        if batch_ids.numel() != rows or flat_indices.numel() != rows:
            raise ValueError("boundary token indices must align with retrieved parents")
        if query_geometry.shape != (rows, GEOMETRY_DIM):
            raise ValueError(f"query_geometry must be [M,{GEOMETRY_DIM}]")
        query_geometry = query_geometry.to(device=e2_map.device, dtype=e2_map.dtype)

        q_semantic = self.semantic_local_encoder(
            gather_local_patches(e2_map, batch_ids, flat_indices, window=5)
        )
        q_detail = self.detail_local_encoder(
            gather_local_patches(e1_map, batch_ids, flat_indices, window=3)
        )
        child = self._gather_children(parent_result, memory, q_semantic)
        parent_valid = self._parent_tensor(parent_result, "top_parent_valid", 2).bool()
        valid = parent_valid & child["child_valid"]

        semantic_score = self.semantic_score_mlp(
            q_semantic, child["top_child_semantic_keys"]
        )
        detail_score = self.detail_score_mlp(
            q_detail, child["top_child_detail_keys"]
        )
        geometry = self.geometry_score_mlp(
            query_geometry,
            self._parent_tensor(parent_result, "top_parent_geometry", 3).to(
                device=q_semantic.device, dtype=q_semantic.dtype
            ),
            child["top_child_geometry"],
        )
        parent_scores = self._parent_tensor(parent_result, "top_parent_scores", 2).to(
            device=q_semantic.device, dtype=q_semantic.dtype
        )
        prior = self.structured_prior(
            parent_scores,
            semantic_score,
            detail_score,
            geometry["score"],
            valid,
        )
        hypothesis_attention = masked_softmax(prior["prior_bias"], valid, dim=-1)
        query_valid = valid.any(dim=-1)
        semantic_support = (
            hypothesis_attention * torch.sigmoid(semantic_score)
        ).sum(dim=-1, keepdim=True)
        detail_support = (
            hypothesis_attention * torch.sigmoid(detail_score)
        ).sum(dim=-1, keepdim=True)
        geometry_support = (
            hypothesis_attention * torch.sigmoid(geometry["score"])
        ).sum(dim=-1, keepdim=True)
        contradiction_token = (
            hypothesis_attention * prior["contradiction"]
        ).sum(dim=-1, keepdim=True)
        parent_evidence = (
            hypothesis_attention.unsqueeze(-1) * parent_keys.to(q_semantic)
        ).sum(dim=1)
        semantic_evidence = (
            hypothesis_attention.unsqueeze(-1) * child["top_child_semantic_keys"]
        ).sum(dim=1)
        detail_evidence = (
            hypothesis_attention.unsqueeze(-1) * child["top_child_detail_keys"]
        ).sum(dim=1)
        verified_evidence = F.normalize(
            (parent_evidence + semantic_evidence + detail_evidence) / 3.0,
            dim=-1,
        )
        valid_float = query_valid.unsqueeze(-1).to(dtype=verified_evidence.dtype)
        verified_evidence = verified_evidence * valid_float

        zero_scores = torch.zeros_like(semantic_score)
        result: dict[str, Any] = {
            "q_semantic": q_semantic,
            "q_detail": q_detail,
            **child,
            "top_parent_valid": valid,
            "query_valid": query_valid,
            "S_semantic": torch.where(valid, semantic_score, zero_scores),
            "S_detail": torch.where(valid, detail_score, zero_scores),
            "S_geometry": torch.where(valid, geometry["score"], zero_scores),
            "semantic_support_scores": torch.where(valid, semantic_score, zero_scores),
            "detail_support_scores": torch.where(valid, detail_score, zero_scores),
            "geometry_support_scores": torch.where(valid, geometry["score"], zero_scores),
            **prior,
            "hypothesis_attention": hypothesis_attention,
            "semantic_support": semantic_support,
            "detail_support": detail_support,
            "geometry_support": geometry_support,
            "contradiction_token": contradiction_token,
            "parent_evidence": parent_evidence * valid_float,
            "semantic_evidence": semantic_evidence * valid_float,
            "detail_evidence": detail_evidence * valid_float,
            "verified_evidence": verified_evidence,
            "geometry_sdf_difference": torch.where(
                valid, geometry["sdf_difference"], zero_scores
            ),
            "geometry_normal_cosine": torch.where(
                valid, geometry["normal_cosine"], zero_scores
            ),
            "geometry_offset_difference": torch.where(
                valid, geometry["offset_difference"], zero_scores
            ),
            "geometry_reliability": torch.where(
                valid, geometry["geometry_reliability"], zero_scores
            ),
        }
        result.update(
            self._supervision_diagnostics(
                semantic_score,
                detail_score,
                self._parent_tensor(parent_result, "top_parent_region_ids", 2).to(
                    device=q_semantic.device, dtype=torch.long
                ),
                valid,
                query_region_ids,
            )
        )
        reasons: list[str] = []
        parent_reasons = tuple(parent_result.get("reason", ()))
        parent_query_valid = self._parent_tensor(
            parent_result, "query_valid", 1
        ).bool()
        for index in range(rows):
            if not bool(parent_query_valid[index]):
                reason = (
                    parent_reasons[index]
                    if index < len(parent_reasons) and parent_reasons[index]
                    else "empty_routed_parent_subbank"
                )
            elif not bool(query_valid[index]):
                reason = "all_child_hypotheses_invalid"
            else:
                reason = ""
            reasons.append(reason)
        result["reason"] = tuple(reasons)
        result["failure_reasons"] = tuple(reasons)
        return result

    @staticmethod
    def _validate_maps(e1_map: torch.Tensor, e2_map: torch.Tensor) -> None:
        if e1_map.ndim != 4 or e2_map.ndim != 4:
            raise ValueError("e1_map and e2_map must be BCHW tensors")
        if e1_map.shape != e2_map.shape or e1_map.size(1) != MEMORY_DIM:
            raise ValueError("e1_map and e2_map must share [B,128,H,W]")
        if e1_map.device != e2_map.device or e1_map.dtype != e2_map.dtype:
            raise ValueError("e1_map and e2_map must share device and dtype")

    @staticmethod
    def _parent_tensor(
        parent_result: Mapping[str, Any], name: str, ndim: int
    ) -> torch.Tensor:
        value = parent_result.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"parent_result.{name} must be a {ndim}-D tensor")
        return value

    @staticmethod
    def _gather_children(
        parent_result: Mapping[str, Any],
        memory: EncoderPCMemory,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ptr = ChildSemanticDetailVerifier._parent_tensor(
            parent_result, "top_child_ptrs", 2
        ).to(device=reference.device, dtype=torch.long)
        parent_image = ChildSemanticDetailVerifier._parent_tensor(
            parent_result, "top_parent_image_indices", 2
        ).to(device=reference.device, dtype=torch.long)
        parent_valid = ChildSemanticDetailVerifier._parent_tensor(
            parent_result, "top_parent_valid", 2
        ).to(device=reference.device, dtype=torch.bool)
        rows, topk = ptr.shape
        child_count = memory.num_children
        in_range = (ptr >= 0) & (ptr < child_count)
        safe_ptr = ptr.clamp(min=0, max=max(child_count - 1, 0))

        def gather_matrix(name: str, width: int) -> torch.Tensor:
            if child_count == 0:
                return reference.new_zeros((rows, topk, width))
            source = memory.child[name].to(
                device=reference.device, dtype=reference.dtype, non_blocking=True
            )
            return source.index_select(0, safe_ptr.reshape(-1)).reshape(rows, topk, width)

        def gather_index(name: str) -> torch.Tensor:
            if child_count == 0:
                return torch.full(
                    (rows, topk), -1, device=reference.device, dtype=torch.long
                )
            source = memory.child[name].to(
                device=reference.device, dtype=torch.long, non_blocking=True
            )
            return source.index_select(0, safe_ptr.reshape(-1)).reshape(rows, topk)

        child_image = gather_index("image_index")
        child_valid = parent_valid & in_range & (child_image == parent_image)
        semantic = gather_matrix("f2_child_keys", MEMORY_DIM)
        detail = gather_matrix("f1_detail_keys", MEMORY_DIM)
        geometry = gather_matrix("geometry", GEOMETRY_DIM)
        mask = child_valid.unsqueeze(-1)
        return {
            "top_child_semantic_keys": torch.where(mask, semantic, torch.zeros_like(semantic)),
            "top_child_detail_keys": torch.where(mask, detail, torch.zeros_like(detail)),
            "top_child_geometry": torch.where(mask, geometry, torch.zeros_like(geometry)),
            "top_child_image_indices": torch.where(
                child_valid, child_image, torch.full_like(child_image, -1)
            ),
            "top_child_flat_indices": torch.where(
                child_valid,
                gather_index("flat_index"),
                torch.full_like(child_image, -1),
            ),
            "child_valid": child_valid,
        }

    @staticmethod
    def _supervision_diagnostics(
        semantic_score: torch.Tensor,
        detail_score: torch.Tensor,
        candidate_region_ids: torch.Tensor,
        valid: torch.Tensor,
        query_region_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if query_region_ids is None:
            false = torch.zeros_like(valid)
            supervision = {
                "semantic_target": false,
                "detail_target": false,
                "semantic_hard_negative_mask": false,
                "detail_hard_negative_mask": false,
            }
            available = torch.zeros(
                valid.size(0), device=valid.device, dtype=torch.bool
            )
        else:
            supervision = build_support_targets(
                query_region_ids, candidate_region_ids, valid
            )
            available = torch.ones(
                valid.size(0), device=valid.device, dtype=torch.bool
            )

        semantic_positive_score, semantic_positive_index, semantic_positive_valid = _masked_max(
            semantic_score, supervision["semantic_target"]
        )
        detail_positive_score, detail_positive_index, detail_positive_valid = _masked_max(
            detail_score, supervision["detail_target"]
        )
        semantic_negative_score, semantic_negative_index, semantic_negative_valid = _masked_max(
            semantic_score, supervision["semantic_hard_negative_mask"]
        )
        detail_negative_score, detail_negative_index, detail_negative_valid = _masked_max(
            detail_score, supervision["detail_hard_negative_mask"]
        )
        return {
            **supervision,
            "support_targets_available": available,
            "semantic_positive_score": semantic_positive_score,
            "semantic_positive_index": semantic_positive_index,
            "semantic_positive_valid": semantic_positive_valid,
            "detail_positive_score": detail_positive_score,
            "detail_positive_index": detail_positive_index,
            "detail_positive_valid": detail_positive_valid,
            "semantic_hard_negative_score": semantic_negative_score,
            "semantic_hard_negative_index": semantic_negative_index,
            "semantic_hard_negative_valid": semantic_negative_valid,
            "detail_hard_negative_score": detail_negative_score,
            "detail_hard_negative_index": detail_negative_index,
            "detail_hard_negative_valid": detail_negative_valid,
        }


class EncoderParentChildDetailVerifier(nn.Module):
    """End-to-end F3 parent retrieval followed by F2/F1 verification."""

    def __init__(
        self,
        *,
        parent_topk: int = PARENT_TOPK,
        query_chunk_size: int = QUERY_CHUNK_SIZE,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.parent_retriever = EncoderParentRetriever(
            topk=parent_topk,
            query_chunk_size=query_chunk_size,
            temperature=temperature,
        )
        self.child_verifier = ChildSemanticDetailVerifier(parent_topk=parent_topk)

    def forward(
        self,
        e1_map: torch.Tensor,
        e2_map: torch.Tensor,
        e3_map: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        parent_subbanks: Sequence[Mapping[str, torch.Tensor]],
        memory: EncoderPCMemory,
        query_geometry: torch.Tensor,
        *,
        query_region_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if e3_map.shape != e1_map.shape:
            raise ValueError("e1_map, e2_map, and e3_map must share [B,128,H,W]")
        parent = self.parent_retriever(
            e3_map, batch_ids, flat_indices, parent_subbanks
        )
        verified = self.child_verifier(
            e1_map,
            e2_map,
            batch_ids,
            flat_indices,
            parent,
            memory,
            query_geometry,
            query_region_ids=query_region_ids,
        )
        return {
            **parent,
            "parent_query_valid": parent["query_valid"],
            **verified,
        }


# Short aliases used by downstream adapter code and ablation tests.
EncoderPCVerifier = EncoderParentChildDetailVerifier
EncoderSemanticDetailVerifier = ChildSemanticDetailVerifier


__all__ = [
    "ChildSemanticDetailVerifier",
    "EncoderParentChildDetailVerifier",
    "EncoderParentRetriever",
    "EncoderPCVerifier",
    "EncoderSemanticDetailVerifier",
    "NormalizedStructuredPrior",
    "build_support_targets",
]
