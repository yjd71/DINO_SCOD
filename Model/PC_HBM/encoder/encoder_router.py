"""Encoder-side image routing for PC-HBM schema v3.

The router owns the single route-key encoder used by both online queries and
memory construction.  Ground-truth masks are deliberately absent from its
interface: every descriptor is derived from projected DINO features and
encoder predictions only.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_memory import EncoderPCMemory


class EncoderPCRouter(nn.Module):
    """Route projected DINO evidence to independent per-image parent banks."""

    def __init__(
        self,
        dim: int = 128,
        *,
        top_img_k: int = 8,
        tau_route: float = 0.07,
        margin_temperature: float = 0.03,
        confidence_floor: float = 0.20,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.top_img_k = int(top_img_k)
        self.tau_route = float(tau_route)
        self.margin_temperature = float(margin_temperature)
        self.confidence_floor = float(confidence_floor)
        if self.dim != EncoderPCMemory.MEMORY_DIM:
            raise ValueError("Encoder route width is fixed to 128")
        if self.top_img_k < 0:
            raise ValueError("top_img_k must be non-negative")
        if self.tau_route <= 0.0:
            raise ValueError("tau_route must be positive")
        if self.margin_temperature <= 0.0:
            raise ValueError("margin_temperature must be positive")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0,1]")

        self.route_mlp = nn.Sequential(
            nn.Linear(self.dim * 5, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )

    def encode_route_key(
        self,
        cls4: torch.Tensor,
        e4_map: torch.Tensor,
        e3_map: torch.Tensor,
        coarse_probability: torch.Tensor,
        boundary_probability: torch.Tensor,
        *,
        uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode the GT-free route key shared by memory and online paths.

        Args:
            cls4: Projected block-11 CLS tokens, ``[B,128]``.
            e4_map: Projected block-11 patch map, ``[B,128,H,W]``.
            e3_map: Projected block-8 patch map, ``[B,128,H,W]``.
            coarse_probability: Encoder coarse probability, ``[B,1,H,W]``.
            boundary_probability: Encoder boundary probability, ``[B,1,H,W]``.
            uncertainty: Optional precomputed probability uncertainty.  When
                omitted, ``4*p*(1-p)`` is derived from the coarse probability.

        Returns:
            Normalized route and diagnostic component keys.  The four fields
            consumed by :class:`EncoderPCMemory` are ``route_key``,
            ``cls4_key``, ``f4_global_key`` and ``f3_boundary_key``.
        """

        batch_size = self._validate_feature_inputs(cls4, e4_map, e3_map)
        coarse = self._prepare_probability(
            coarse_probability,
            batch_size=batch_size,
            spatial_size=e3_map.shape[-2:],
            name="coarse_probability",
            dtype=e3_map.dtype,
            device=e3_map.device,
        )
        boundary = self._prepare_probability(
            boundary_probability,
            batch_size=batch_size,
            spatial_size=e3_map.shape[-2:],
            name="boundary_probability",
            dtype=e3_map.dtype,
            device=e3_map.device,
        )
        if uncertainty is None:
            uncertainty_probability = 4.0 * coarse * (1.0 - coarse)
        else:
            uncertainty_probability = self._prepare_probability(
                uncertainty,
                batch_size=batch_size,
                spatial_size=e3_map.shape[-2:],
                name="uncertainty",
                dtype=e3_map.dtype,
                device=e3_map.device,
            )

        cls4_key = F.normalize(cls4, dim=-1)
        f4_global_key = F.normalize(e4_map.mean(dim=(-2, -1)), dim=-1)
        f3_boundary_key = F.normalize(
            self._masked_pool(e3_map, boundary), dim=-1
        )
        f3_uncertainty_key = F.normalize(
            self._masked_pool(e3_map, uncertainty_probability), dim=-1
        )
        environment_weight = (1.0 - coarse) * (1.0 - boundary)
        f3_environment_key = F.normalize(
            self._masked_pool(e3_map, environment_weight), dim=-1
        )
        route_raw = torch.cat(
            (
                cls4_key,
                f4_global_key,
                f3_boundary_key,
                f3_uncertainty_key,
                f3_environment_key,
            ),
            dim=-1,
        )
        route_key = F.normalize(self.route_mlp(route_raw), dim=-1)
        return {
            "route_key": route_key,
            "cls4_key": cls4_key,
            "f4_global_key": f4_global_key,
            "f3_boundary_key": f3_boundary_key,
            "f3_uncertainty_key": f3_uncertainty_key,
            "f3_environment_key": f3_environment_key,
        }

    def route(
        self,
        query_route: torch.Tensor,
        memory: EncoderPCMemory,
        *,
        query_image_ids: Sequence[object] | None = None,
        top_img_k: int | None = None,
        require_same_image_positive: bool = False,
    ) -> dict[str, Any]:
        """Route queries while keeping supervision and retrieval masks separate.

        ``route_logits`` always contains the unmasked full-memory logits used
        for same-image InfoNCE.  Self matches are removed only from the actual
        top-k retrieval.  Parent subbanks are then materialized independently
        for every batch item.
        """

        self._validate_route_inputs(query_route, memory, query_image_ids)
        batch_size = query_route.size(0)
        k = self.top_img_k if top_img_k is None else int(top_img_k)
        if k < 0:
            raise ValueError("top_img_k must be non-negative")

        memory_keys = memory.route["route_keys"].to(
            device=query_route.device,
            dtype=query_route.dtype,
            non_blocking=True,
        )
        similarities = F.normalize(query_route, dim=-1) @ F.normalize(
            memory_keys, dim=-1
        ).transpose(0, 1)
        route_logits = similarities / self.tau_route
        image_ids = tuple(str(image_id) for image_id in memory.route["image_ids"])
        canonical_queries = (
            None
            if query_image_ids is None
            else tuple(str(image_id) for image_id in query_image_ids)
        )

        positive_indices = torch.full(
            (batch_size,), -1, device=query_route.device, dtype=torch.long
        )
        route_info_nce: torch.Tensor | None = None
        if require_same_image_positive:
            if canonical_queries is None:
                raise ValueError(
                    "labeled route supervision requires query_image_ids"
                )
            positive_indices = self._same_image_positive_indices(
                canonical_queries, image_ids, query_route.device
            )
            # Supervision intentionally uses the unmasked logits.  The same
            # positive is masked below only for actual memory retrieval.
            route_info_nce = F.cross_entropy(route_logits, positive_indices)

        top_scores, top_similarities, top_valid, top_indices, top_ids = (
            self._retrieve_rows(
                route_logits,
                similarities,
                image_ids,
                canonical_queries,
                k,
            )
        )
        route_attention = self._masked_softmax(top_scores, top_valid)
        route_entropy, route_entropy_norm = self._route_entropy(
            route_attention, top_valid
        )
        valid_count = top_valid.sum(dim=1)
        route_valid = valid_count >= 2
        route_margin = query_route.new_zeros(batch_size)
        if k >= 2:
            # Confidence is calibrated in cosine-similarity space.  The
            # temperature-scaled scores remain useful for retrieval attention
            # and InfoNCE, but applying tau_route a second time here would make
            # even tiny cosine margins saturate the confidence sigmoid.
            raw_margin = top_similarities[:, 0] - top_similarities[:, 1]
            route_margin = torch.where(
                route_valid, raw_margin, torch.zeros_like(raw_margin)
            )
        margin_confidence = torch.where(
            route_valid,
            torch.sigmoid(route_margin / self.margin_temperature),
            torch.zeros_like(route_margin),
        )
        route_confidence = torch.where(
            route_valid,
            margin_confidence.clamp_min(self.confidence_floor),
            torch.full_like(margin_confidence, self.confidence_floor),
        )

        parent_subbanks = self._build_parent_subbanks(
            memory, top_indices, top_valid, query_route
        )
        return {
            "route_logits": route_logits,
            "positive_memory_image_index": positive_indices,
            "route_info_nce": route_info_nce,
            "top_img_ids": top_ids,
            "top_img_scores": top_scores,
            "top_img_similarities": top_similarities,
            "top_img_valid": top_valid,
            "top_img_indices": top_indices,
            "route_attention": route_attention,
            # Entropy is diagnostic only.  It is deliberately not used in the
            # confidence calculation or as an implicit retrieval gate.
            "route_entropy": route_entropy,
            "route_entropy_norm": route_entropy_norm,
            "route_margin": route_margin,
            "route_margin_confidence": margin_confidence,
            "route_confidence": route_confidence,
            "route_valid": route_valid,
            "routed_parent_indices": tuple(
                subbank["global_parent_indices"] for subbank in parent_subbanks
            ),
            "parent_subbanks": parent_subbanks,
        }

    def forward(
        self,
        cls4: torch.Tensor,
        e4_map: torch.Tensor,
        e3_map: torch.Tensor,
        coarse_probability: torch.Tensor,
        boundary_probability: torch.Tensor,
        memory: EncoderPCMemory,
        *,
        uncertainty: torch.Tensor | None = None,
        query_image_ids: Sequence[object] | None = None,
        top_img_k: int | None = None,
        require_same_image_positive: bool = False,
    ) -> dict[str, Any]:
        encoded = self.encode_route_key(
            cls4,
            e4_map,
            e3_map,
            coarse_probability,
            boundary_probability,
            uncertainty=uncertainty,
        )
        routed = self.route(
            encoded["route_key"],
            memory,
            query_image_ids=query_image_ids,
            top_img_k=top_img_k,
            require_same_image_positive=require_same_image_positive,
        )
        routed.update(encoded)
        return routed

    def _validate_feature_inputs(
        self,
        cls4: torch.Tensor,
        e4_map: torch.Tensor,
        e3_map: torch.Tensor,
    ) -> int:
        if cls4.ndim != 2 or cls4.size(1) != self.dim:
            raise ValueError(f"cls4 must be [B,{self.dim}], got {tuple(cls4.shape)}")
        batch_size = cls4.size(0)
        for name, feature in (("e4_map", e4_map), ("e3_map", e3_map)):
            if (
                feature.ndim != 4
                or feature.size(0) != batch_size
                or feature.size(1) != self.dim
            ):
                raise ValueError(
                    f"{name} must be [B,{self.dim},H,W], got {tuple(feature.shape)}"
                )
            if feature.device != cls4.device or feature.dtype != cls4.dtype:
                raise ValueError(f"{name} must match cls4 device and dtype")
        if e4_map.shape[-2:] != e3_map.shape[-2:]:
            raise ValueError("e4_map and e3_map must share the 28x28 token grid")
        if not torch.isfinite(cls4).all() or not torch.isfinite(e4_map).all() or not torch.isfinite(e3_map).all():
            raise ValueError("route feature inputs must be finite")
        return batch_size

    @staticmethod
    def _prepare_probability(
        probability: torch.Tensor,
        *,
        batch_size: int,
        spatial_size: tuple[int, int],
        name: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if probability.ndim != 4 or probability.shape[:2] != (batch_size, 1):
            raise ValueError(f"{name} must be [B,1,H,W], got {tuple(probability.shape)}")
        if probability.device != device:
            raise ValueError(f"{name} must match projected feature device")
        if not probability.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(probability).all():
            raise ValueError(f"{name} must be finite")
        detached = probability.detach()
        if detached.numel() and (detached.min() < -1.0e-6 or detached.max() > 1.0 + 1.0e-6):
            raise ValueError(f"{name} must contain probabilities in [0,1]")
        # CUDA autocast keeps LayerNorm-ended projected tokens in FP32 while
        # sigmoid heads can emit FP16.  Align without detaching so route
        # supervision remains differentiable across the AMP boundary.
        probability = probability.to(dtype=dtype)
        if probability.shape[-2:] != spatial_size:
            probability = F.interpolate(
                probability,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        return probability.clamp(0.0, 1.0)

    def _validate_route_inputs(
        self,
        query_route: torch.Tensor,
        memory: EncoderPCMemory,
        query_image_ids: Sequence[object] | None,
    ) -> None:
        if query_route.ndim != 2 or query_route.size(1) != self.dim:
            raise ValueError(
                f"query_route must be [B,{self.dim}], got {tuple(query_route.shape)}"
            )
        if not torch.is_floating_point(query_route) or not torch.isfinite(query_route).all():
            raise ValueError("query_route must be a finite floating tensor")
        if not isinstance(memory, EncoderPCMemory):
            raise TypeError("encoder routing requires EncoderPCMemory schema v3")
        if not memory.is_ready():
            raise RuntimeError("EncoderPCMemory is not finalized and ready")
        if query_image_ids is not None and len(query_image_ids) != query_route.size(0):
            raise ValueError("query_image_ids length must match route query batch")

    @staticmethod
    def _same_image_positive_indices(
        query_image_ids: Sequence[str],
        memory_image_ids: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        positions: dict[str, list[int]] = {}
        for index, image_id in enumerate(memory_image_ids):
            positions.setdefault(image_id, []).append(index)
        positive: list[int] = []
        for image_id in query_image_ids:
            matches = positions.get(image_id, [])
            if not matches:
                raise RuntimeError(
                    f"labeled route positive is missing from memory: {image_id!r}"
                )
            if len(matches) != 1:
                raise RuntimeError(
                    f"labeled route positive must be deterministic, found {len(matches)} keys for {image_id!r}"
                )
            positive.append(matches[0])
        return torch.tensor(positive, device=device, dtype=torch.long)

    @staticmethod
    def _retrieve_rows(
        route_logits: torch.Tensor,
        similarities: torch.Tensor,
        memory_image_ids: Sequence[str],
        query_image_ids: Sequence[str] | None,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
        batch_size = route_logits.size(0)
        score_rows: list[torch.Tensor] = []
        similarity_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        index_rows: list[torch.Tensor] = []
        id_rows: list[list[str]] = []
        for batch_index in range(batch_size):
            candidate_valid = torch.ones(
                len(memory_image_ids),
                device=route_logits.device,
                dtype=torch.bool,
            )
            if query_image_ids is not None:
                own_id = query_image_ids[batch_index]
                for memory_index, image_id in enumerate(memory_image_ids):
                    if image_id == own_id:
                        candidate_valid[memory_index] = False
            candidate_indices = torch.nonzero(candidate_valid, as_tuple=False).flatten()
            count = min(k, int(candidate_indices.numel()))
            scores = route_logits.new_full((k,), -1.0e4)
            row_similarities = similarities.new_full((k,), -1.0)
            valid = torch.zeros(k, device=route_logits.device, dtype=torch.bool)
            indices = torch.full(
                (k,), -1, device=route_logits.device, dtype=torch.long
            )
            selected_ids: list[str] = []
            if count:
                candidate_scores = route_logits[batch_index].index_select(
                    0, candidate_indices
                )
                selected_scores, local_indices = torch.topk(
                    candidate_scores, k=count
                )
                selected_indices = candidate_indices.index_select(0, local_indices)
                scores[:count] = selected_scores
                row_similarities[:count] = similarities[batch_index].index_select(
                    0, selected_indices
                )
                valid[:count] = True
                indices[:count] = selected_indices
                selected_ids = [
                    memory_image_ids[index]
                    for index in selected_indices.detach().cpu().tolist()
                ]
            score_rows.append(scores)
            similarity_rows.append(row_similarities)
            valid_rows.append(valid)
            index_rows.append(indices)
            id_rows.append(selected_ids)
        return (
            torch.stack(score_rows),
            torch.stack(similarity_rows),
            torch.stack(valid_rows),
            torch.stack(index_rows),
            id_rows,
        )

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        masked = scores.masked_fill(~valid, -torch.inf)
        has_valid = valid.any(dim=1, keepdim=True)
        safe = torch.where(has_valid, masked, torch.zeros_like(masked))
        attention = torch.softmax(safe, dim=1) * valid.to(dtype=scores.dtype)
        return attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    @staticmethod
    def _route_entropy(
        attention: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention.size(1) == 0:
            empty = attention.new_zeros(attention.size(0))
            return empty, empty
        entropy = -(attention * attention.clamp_min(1.0e-8).log()).sum(dim=1)
        valid_count = valid.sum(dim=1)
        normalizer = valid_count.clamp_min(2).to(attention.dtype).log()
        normalized = torch.where(
            valid_count > 1,
            entropy / normalizer,
            torch.zeros_like(entropy),
        )
        return entropy, normalized

    @staticmethod
    def _build_parent_subbanks(
        memory: EncoderPCMemory,
        top_indices: torch.Tensor,
        top_valid: torch.Tensor,
        query_route: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        parent_image_index = memory.parent["image_index"].to(
            device=query_route.device, dtype=torch.long, non_blocking=True
        )
        subbanks: list[dict[str, torch.Tensor]] = []
        for batch_index in range(query_route.size(0)):
            routed_images = top_indices[batch_index][top_valid[batch_index]]
            if routed_images.numel():
                selected = torch.nonzero(
                    torch.isin(parent_image_index, routed_images),
                    as_tuple=False,
                ).flatten()
            else:
                selected = torch.empty(
                    0, device=query_route.device, dtype=torch.long
                )
            subbank: dict[str, torch.Tensor] = {
                "global_parent_indices": selected,
            }
            for name, value in memory.parent.items():
                source = value.to(device=query_route.device, non_blocking=True)
                if torch.is_floating_point(source):
                    source = source.to(dtype=query_route.dtype)
                else:
                    source = source.to(dtype=torch.long)
                subbank[name] = source.index_select(0, selected)
            subbanks.append(subbank)
        return subbanks

    @staticmethod
    def _masked_pool(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        numerator = (feature * weight).sum(dim=(-2, -1))
        denominator = weight.sum(dim=(-2, -1)).clamp_min(1.0e-6)
        return numerator / denominator


# Short alias retained for callers that name the component by file role.
EncoderRouter = EncoderPCRouter


__all__ = ["EncoderPCRouter", "EncoderRouter"]
