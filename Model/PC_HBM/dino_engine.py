"""DINO PC-HBM-Lite orchestration.

The engine contains exactly one correction path:

dual-context route -> balanced Pair Memory retrieval -> P2 cosine
verification -> binary evidence -> gated P3 residual.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from .common.utils import gather_tokens
from .dino_memory_builder import DinoMemoryBuilder
from .fusion import P3GatedResidual
from .refinement import BoundaryQuerySelector
from .retrieval import (
    BalancedParentRetriever,
    ChildQueryBuilder,
    PairVerifier,
)
from .routing import CamouflageContextRouter


class DinoPCHBMEngine(nn.Module):
    """Compose the complete PC-HBM-Lite path."""

    VALID_MODES = {"verify_only", "full", "teacher_pseudo"}

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate_runtime_contract(cfg)
        self.query_selector = BoundaryQuerySelector(
            top_ratio=cfg.p3_top_ratio,
            min_tokens=cfg.p3_min_tokens,
            max_tokens=cfg.p3_max_tokens,
            boundary_weight=cfg.query_boundary_weight,
            uncertainty_weight=cfg.query_uncertainty_weight,
        )
        self.router = CamouflageContextRouter(
            dim=cfg.memory_dim,
            top_img_k=cfg.route_top_img_k,
            global_weight=cfg.route_global_weight,
            environment_weight=cfg.route_environment_weight,
            min_environment_mass=cfg.route_environment_min_mass,
        )
        self.parent_retriever = BalancedParentRetriever(
            p3_ch=cfg.decoder_dim,
            dim=cfg.memory_dim,
            topk_per_region=cfg.parent_topk_per_region,
        )
        self.child_query = ChildQueryBuilder(
            p2_ch=cfg.decoder_dim,
            dim=cfg.memory_dim,
            window=cfg.child_window_size,
        )
        self.pair_verifier = PairVerifier(
            dim=cfg.memory_dim,
            tau_parent=cfg.tau_parent,
            tau_child=cfg.tau_child,
            child_mix_init_logit=cfg.child_mix_init_logit,
            child_verification_mode=cfg.child_verification_mode,
        )
        self.p3_residual = P3GatedResidual(
            dim=cfg.memory_dim,
            p3_ch=cfg.decoder_dim,
        )
        self.memory_builder = DinoMemoryBuilder(
            cfg,
            self.router,
            self.parent_retriever,
            self.child_query,
        )

    @staticmethod
    def _validate_runtime_contract(cfg) -> None:
        positive_integer_fields = (
            "input_size",
            "encoder_dim",
            "decoder_dim",
            "memory_dim",
            "token_size",
            "output_size",
            "route_top_img_k",
            "parent_topk_per_region",
            "query_chunk_size",
        )
        for name in positive_integer_fields:
            actual = getattr(cfg, name, None)
            if (
                not isinstance(actual, Integral)
                or isinstance(actual, bool)
                or int(actual) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        if int(cfg.memory_dim) != int(cfg.decoder_dim):
            raise ValueError(
                "memory_dim must equal decoder_dim because the parameter-free "
                "Router pools x3 without a channel projection"
            )
        child_window = getattr(cfg, "child_window_size", None)
        if (
            not isinstance(child_window, Integral)
            or isinstance(child_window, bool)
            or int(child_window) <= 0
            or int(child_window) % 2 == 0
        ):
            raise ValueError("child_window_size must be a positive odd integer")
        region_names = tuple(getattr(cfg, "region_names", ()))
        if (
            len(region_names) != 2
            or any(not isinstance(name, str) or not name for name in region_names)
            or len(set(region_names)) != 2
        ):
            raise ValueError(
                "binary region evidence requires two unique non-empty region names"
            )
        layer_indices = tuple(getattr(cfg, "dino_layer_indices", ()))
        if (
            len(layer_indices) != 4
            or any(
                not isinstance(index, Integral)
                or isinstance(index, bool)
                or int(index) < 0
                for index in layer_indices
            )
            or len(set(int(index) for index in layer_indices)) != 4
        ):
            raise ValueError(
                "legacy Decoder integration requires four unique non-negative "
                "DINO layer indices"
            )

    def _validate_inputs(
        self,
        x3: torch.Tensor,
        p3: torch.Tensor,
        p2: torch.Tensor,
        m3: torch.Tensor,
    ) -> None:
        expected_feature = (
            x3.size(0),
            self.cfg.decoder_dim,
            self.cfg.token_size,
            self.cfg.token_size,
        )
        for name, value in (("x3", x3), ("p3", p3), ("p2", p2)):
            if value.ndim != 4 or tuple(value.shape) != expected_feature:
                raise ValueError(
                    f"{name} must be {expected_feature}, got {tuple(value.shape)}"
                )
        expected_mask = (
            x3.size(0),
            1,
            self.cfg.token_size,
            self.cfg.token_size,
        )
        if m3.ndim != 4 or tuple(m3.shape) != expected_mask:
            raise ValueError(
                f"m3 must be {expected_mask}, got {tuple(m3.shape)}"
            )
        if not (x3.device == p3.device == p2.device == m3.device):
            raise ValueError("x3, p3, p2 and m3 must share one device")

    def _routed_retrieval(
        self,
        q3: torch.Tensor,
        batch_ids: torch.Tensor,
        route: Mapping[str, Any],
        memory,
        query_image_ids: Optional[Sequence[str]],
    ) -> dict[str, torch.Tensor]:
        result = self.parent_retriever.empty_result(q3)
        top_img_ids = route.get("top_img_ids")
        if not isinstance(top_img_ids, Sequence) or len(top_img_ids) != int(
            route["route_global"].size(0)
        ):
            raise ValueError(
                "Router top_img_ids must contain one routed list per image"
            )
        for batch_index in range(len(top_img_ids)):
            output_positions = torch.nonzero(
                batch_ids == batch_index, as_tuple=False
            ).flatten()
            if output_positions.numel() == 0:
                continue
            exclude_image_id = None
            if (
                bool(self.cfg.exclude_self_match)
                and query_image_ids is not None
            ):
                exclude_image_id = str(query_image_ids[batch_index])
            pair_subbank = memory.get_pair_subbank(
                top_img_ids[batch_index],
                device=q3.device,
                dtype=q3.dtype,
                exclude_image_id=exclude_image_id,
            )
            selected_q3 = q3.index_select(0, output_positions)
            selected = self.parent_retriever.retrieve_q(
                selected_q3,
                pair_subbank,
                chunk_size=self.cfg.query_chunk_size,
            )
            for key in (
                "parent_keys",
                "paired_p2_keys",
                "scores",
                "indices",
                "valid",
                "query_valid",
            ):
                result[key].index_copy_(
                    0, output_positions, selected[key]
                )
        return result

    @staticmethod
    def _scatter_scalar(
        batch_size: int,
        height: int,
        width: int,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros(
            (batch_size, 1, height, width),
            device=values.device,
            dtype=torch.float32,
        )
        if batch_ids.numel() == 0:
            return output
        row = torch.div(flat_indices, width, rounding_mode="floor")
        col = flat_indices.remainder(width)
        output[batch_ids, 0, row, col] = values.reshape(-1).float()
        return output

    def forward_lite(
        self,
        x3: torch.Tensor,
        p3: torch.Tensor,
        p2: torch.Tensor,
        m3: torch.Tensor,
        memory,
        mode: str,
        *,
        injection_scale: float = 1.0,
        query_image_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, object]:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported Lite engine mode {mode!r}; "
                f"expected one of {sorted(self.VALID_MODES)}"
            )
        self._validate_inputs(x3, p3, p2, m3)
        if memory is None:
            raise ValueError("forward_lite requires a ready V2 Pair Memory")
        if query_image_ids is not None and len(query_image_ids) != x3.size(0):
            raise ValueError("query_image_ids must match the batch size")

        probability = torch.sigmoid(m3.float())
        _, selected = self.query_selector(probability)
        batch_ids = selected["batch_ids"]
        flat_indices = selected["flat_indices"]
        query_scores = selected["token_scores"]

        route = self.router(
            x3,
            probability,
            memory,
            top_img_k=self.cfg.route_top_img_k,
            query_image_ids=query_image_ids,
            exclude_self_match=self.cfg.exclude_self_match,
        )
        q3_map = self.parent_retriever.encode_q_map(p3)
        q3 = gather_tokens(q3_map, batch_ids, flat_indices)
        retrieval = self._routed_retrieval(
            q3,
            batch_ids,
            route,
            memory,
            query_image_ids,
        )
        child = self.child_query(
            p2,
            batch_ids,
            flat_indices,
            p3_hw=p3.shape[-2:],
        )
        verified = self.pair_verifier(
            q3,
            child["q_child"],
            retrieval,
            query_score=query_scores,
        )

        if mode == "verify_only":
            p3_corr = p3
            p3_delta = p3.new_zeros((batch_ids.numel(), p3.size(1)))
            effective_scale = 0.0
        else:
            effective_scale = float(injection_scale)
            p3_corr, p3_delta = self.p3_residual(
                p3,
                batch_ids,
                flat_indices,
                verified["correction"],
                verified["gate"],
                verified["query_valid"],
                injection_scale=effective_scale,
            )

        height, width = p3.shape[-2:]
        query_mask_map = self._scatter_scalar(
            p3.size(0),
            height,
            width,
            batch_ids,
            flat_indices,
            torch.ones_like(query_scores, dtype=torch.float32),
        )
        memory_confidence_map = self._scatter_scalar(
            p3.size(0),
            height,
            width,
            batch_ids,
            flat_indices,
            verified["memory_confidence"],
        )
        gate_map = self._scatter_scalar(
            p3.size(0),
            height,
            width,
            batch_ids,
            flat_indices,
            verified["gate"],
        )

        return {
            "query_batch_ids": batch_ids,
            "query_flat_indices": flat_indices,
            "query_valid": verified["query_valid"],
            "pair_scores": verified["pair_scores"],
            "pair_weight": verified["pair_weight"],
            "pair_logits": verified["pair_logits"],
            "region_prob": verified["region_prob"],
            "parent_cosine": verified["parent_cosine"],
            "parent_scores": verified["parent_scores"],
            "parent_region_logits": verified["parent_region_logits"],
            "parent_region_prob": verified["parent_region_prob"],
            "child_cosine": verified["child_cosine"],
            "child_abs_cosine": verified["child_abs_cosine"],
            "child_relation_cosine": verified["child_relation_cosine"],
            "relation_valid": verified["relation_valid"],
            "child_verify_logits": verified["child_verify_logits"],
            "verification_strength": verified["verification_strength"],
            "verification_abs_weight": verified["verification_abs_weight"],
            "verification_rel_weight": verified["verification_rel_weight"],
            "verification_bias": verified["verification_bias"],
            "verified_scores": verified["verified_scores"],
            "verified_region_logits": verified["verified_region_logits"],
            "verified_region_prob": verified["verified_region_prob"],
            "candidate_entropy": verified["candidate_entropy"],
            "memory_confidence": verified["memory_confidence"],
            "gate": verified["gate"],
            "beta": verified["beta"],
            "retrieval_valid": retrieval["valid"],
            "route": route,
            "query_mask_map": query_mask_map,
            "memory_confidence_map": memory_confidence_map,
            "gate_map": gate_map,
            "p3_delta": p3_delta,
            "p3_corr": p3_corr,
        }

    def build_memory_entries(
        self,
        features: dict[str, torch.Tensor],
        gt: torch.Tensor,
        image_ids: Sequence[str],
    ) -> dict[str, Any]:
        return self.memory_builder(features, gt, image_ids)
