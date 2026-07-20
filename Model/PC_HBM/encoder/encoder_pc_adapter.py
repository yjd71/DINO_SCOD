"""End-to-end encoder-side PC-HBM adapter before the unchanged decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.common.utils import gather_tokens

from .child_semantic_detail_verifier import EncoderParentChildDetailVerifier
from .contracts import DinoFeatureBundle, Tensor4
from .encoder_global_fusion import EncoderBootstrap, EncoderBootstrapOutput
from .encoder_memory import EncoderPCMemory
from .encoder_router import EncoderPCRouter
from .encoder_level_propagation import EncoderLevelPropagation
from .encoder_feature_injector import (
    EncoderF4F3InjectionOutput,
    EncoderFeatureInjector,
)
from .route_context_adapter import (
    EncoderRouteContextAdapter,
    EncoderRouteContextOutput,
)


@dataclass(frozen=True)
class EncoderPCStageFlags:
    enable_f4_f3: bool = True
    f4_f3_progress: float = 1.0
    enable_f2_f1: bool = True
    f2_f1_progress: float = 1.0
    require_same_image_positive: bool = False


@dataclass(frozen=True)
class EncoderPCAdapterOutput:
    features: Tensor4
    aux: Mapping[str, Any]


class EncoderPCHBMAdapter(nn.Module):
    VALID_MODES = {"off", "bootstrap", "parent_only", "full"}

    def __init__(self, config: EncoderPCHBMConfig | None = None) -> None:
        super().__init__()
        self.config = EncoderPCHBMConfig() if config is None else config
        cfg = self.config
        self.bootstrap = EncoderBootstrap(
            encoder_dim=cfg.encoder_dim,
            memory_dim=cfg.memory_dim,
            token_size=cfg.token_size,
            boundary_token_ratio=cfg.boundary_token_ratio,
            boundary_min_tokens=cfg.boundary_min_tokens,
            boundary_max_tokens=cfg.boundary_max_tokens,
        )
        self.router = EncoderPCRouter(
            cfg.memory_dim,
            top_img_k=cfg.route_top_img_k,
            tau_route=cfg.tau_route,
            margin_temperature=cfg.route_margin_temperature,
            confidence_floor=cfg.route_confidence_floor,
        )
        self.verifier = EncoderParentChildDetailVerifier()
        self.route_context = EncoderRouteContextAdapter(
            cfg.memory_dim,
            num_heads=cfg.attention_heads,
            token_size=cfg.token_size,
        )
        self.injector = EncoderFeatureInjector(
            cfg.memory_dim,
            cfg.encoder_dim,
            max_f4=cfg.max_f4_injection,
            max_f3=cfg.max_f3_injection,
            alpha_init=cfg.injection_alpha_init,
        )
        self.propagation = EncoderLevelPropagation(
            cfg.memory_dim,
            cfg.encoder_dim,
            num_heads=cfg.attention_heads,
            window_size=cfg.propagation_window_size,
            max_f2=cfg.max_f2_injection,
            max_f1=cfg.max_f1_injection,
            alpha_init=cfg.injection_alpha_init,
            detach_f3_refs=cfg.detach_f3_refs_for_f2,
            detach_f2_refs=cfg.detach_f2_refs_for_f1,
        )

    def forward_memory_features(
        self, bundle: DinoFeatureBundle
    ) -> Mapping[str, torch.Tensor]:
        """Return only raw-bundle keys consumed by the labeled memory builder."""

        bootstrap = self.bootstrap(bundle)
        e1, e2, e3, e4 = bootstrap.projected.maps
        route = self.router.encode_route_key(
            bootstrap.projected.cls_tokens[3],
            e4,
            e3,
            bootstrap.global_output.coarse_probability,
            bootstrap.boundary_output.boundary_probability,
        )
        return {
            "route_keys": route["route_key"],
            "cls4_keys": route["cls4_key"],
            "f4_global_keys": route["f4_global_key"],
            "f3_boundary_keys": route["f3_boundary_key"],
            "f3_parent_keys": bootstrap.projected.patch_tokens[2],
            "f2_child_keys": bootstrap.projected.patch_tokens[1],
            "f1_detail_keys": bootstrap.projected.patch_tokens[0],
        }

    def forward(
        self,
        bundle: DinoFeatureBundle,
        *,
        memory: EncoderPCMemory | None = None,
        mode: str = "off",
        stage: EncoderPCStageFlags | None = None,
        query_image_ids: Sequence[object] | None = None,
        allow_memory_fallback: bool = False,
    ) -> EncoderPCAdapterOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported encoder PC mode={mode!r}; expected {sorted(self.VALID_MODES)}."
            )
        if mode == "off":
            return EncoderPCAdapterOutput(
                features=bundle.patch_tokens,
                aux={"mode": "off", "pc_active": False, "fallback_reason": None},
            )

        bootstrap = self.bootstrap(bundle)
        aux: dict[str, Any] = {
            "mode": mode,
            "pc_active": False,
            "fallback_reason": None,
            "bootstrap": bootstrap,
            "coarse_logits": bootstrap.global_output.coarse_logits,
            "coarse_probability": bootstrap.global_output.coarse_probability,
            "boundary_logits": bootstrap.boundary_output.boundary_logits,
            "boundary_probability": bootstrap.boundary_output.boundary_probability,
        }
        if mode == "bootstrap":
            return EncoderPCAdapterOutput(bundle.patch_tokens, aux)

        if memory is None or not isinstance(memory, EncoderPCMemory) or not memory.is_ready():
            reason = "missing_encoder_pc_memory" if memory is None else "incompatible_encoder_pc_memory"
            if self.training or not allow_memory_fallback:
                raise RuntimeError(f"Encoder PC-HBM requires ready schema-v3 memory: {reason}.")
            aux["fallback_reason"] = reason
            return EncoderPCAdapterOutput(bundle.patch_tokens, aux)

        flags = EncoderPCStageFlags() if stage is None else stage
        e1, e2, e3, e4 = bootstrap.projected.maps
        route_keys = self.router.encode_route_key(
            bootstrap.projected.cls_tokens[3],
            e4,
            e3,
            bootstrap.global_output.coarse_probability,
            bootstrap.boundary_output.boundary_probability,
        )
        route = self.router.route(
            route_keys["route_key"],
            memory,
            query_image_ids=query_image_ids,
            require_same_image_positive=flags.require_same_image_positive,
        )
        aux["route"] = route
        batch_ids, flat_indices = self._sparse_indices(
            bootstrap.boundary_output.selected_indices
        )
        parent = self.verifier.parent_retriever(
            e3, batch_ids, flat_indices, route["parent_subbanks"]
        )
        q3 = parent["q3"]
        aux["parent"] = parent
        if mode == "parent_only":
            aux["pc_active"] = bool(parent["query_valid"].any())
            return EncoderPCAdapterOutput(bundle.patch_tokens, aux)

        query_geometry_map = self._predicted_geometry(
            bootstrap.global_output.coarse_probability,
            bootstrap.boundary_output.boundary_probability,
        )
        query_geometry = gather_tokens(query_geometry_map, batch_ids, flat_indices)
        verification = self.verifier.child_verifier(
            e1,
            e2,
            batch_ids,
            flat_indices,
            parent,
            memory,
            query_geometry,
        )
        verification = {
            **parent,
            "parent_query_valid": parent["query_valid"],
            **verification,
        }
        uncertainty_map = 4.0 * bootstrap.global_output.coarse_probability * (
            1.0 - bootstrap.global_output.coarse_probability
        )
        uncertainty_tokens = gather_tokens(
            uncertainty_map, batch_ids, flat_indices
        )
        boundary_confidence_tokens = gather_tokens(
            bootstrap.boundary_output.boundary_probability,
            batch_ids,
            flat_indices,
        )
        sparse_route_context = route_keys["route_key"].index_select(0, batch_ids)
        sparse_route_confidence = route["route_confidence"].index_select(0, batch_ids)
        context = self.route_context(
            q3=q3,
            verification=verification,
            route_context=sparse_route_context,
            route_confidence=sparse_route_confidence,
            uncertainty=uncertainty_tokens,
            boundary_confidence=boundary_confidence_tokens,
            batch_ids=batch_ids,
            flat_indices=flat_indices,
            batch_size=e1.shape[0],
        )
        routed_evidence = self._routed_memory_evidence(route, memory, e1)
        progress = flags.f4_f3_progress if flags.enable_f4_f3 else 0.0
        injection = self.injector(
            f3_tokens=bundle.patch_tokens[2],
            f4_tokens=bundle.patch_tokens[3],
            route_evidence=routed_evidence,
            route_confidence=route["route_confidence"],
            route_valid=route["route_valid"],
            verified_f3_map=context.verified_f3_map,
            f3_gate_map=context.gate_map,
            progress=progress,
        )
        propagation = None
        f1_tokens, f2_tokens = bundle.patch_tokens[:2]
        if flags.enable_f2_f1:
            propagation = self.propagation(
                f1_tokens=f1_tokens,
                f2_tokens=f2_tokens,
                e1_map=e1,
                e2_map=e2,
                corrected_f3_state=e3 + context.gate_map * context.verified_f3_map,
                verified_f2_map=context.verified_f2_map,
                verified_f1_map=context.verified_f1_map,
                valid2_map=context.valid2_map,
                valid1_map=context.valid1_map,
                progress=flags.f2_f1_progress,
            )
            f1_tokens, f2_tokens = propagation.f1_tokens, propagation.f2_tokens
        enhanced: Tensor4 = (
            f1_tokens,
            f2_tokens,
            injection.f3_tokens,
            injection.f4_tokens,
        )
        aux.update(
            {
                "pc_active": bool(context.valid3_map.any()),
                "verification": verification,
                "route_context": context,
                "injection": injection,
                "propagation": propagation,
                "C23_map": context.c23_map,
                "semantic_support_map": context.semantic_support_map,
                "detail_support_map": context.detail_support_map,
                "geometry_support_map": context.geometry_support_map,
            }
        )
        return EncoderPCAdapterOutput(enhanced, aux)

    @staticmethod
    def _sparse_indices(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if selected.ndim != 2:
            raise ValueError("selected boundary indices must be [B,Q].")
        batch, count = selected.shape
        batch_ids = torch.arange(batch, device=selected.device).repeat_interleave(count)
        return batch_ids, selected.reshape(-1).long()

    @staticmethod
    def _predicted_geometry(
        coarse_probability: torch.Tensor,
        boundary_probability: torch.Tensor,
    ) -> torch.Tensor:
        probability = coarse_probability.float().clamp(0.0, 1.0)
        dx = F.pad(probability[..., 1:] - probability[..., :-1], (0, 1, 0, 0))
        dy = F.pad(probability[..., 1:, :] - probability[..., :-1, :], (0, 0, 0, 1))
        magnitude = torch.sqrt(dx.square() + dy.square() + 1e-8)
        normal_x = dx / magnitude
        normal_y = dy / magnitude
        offset_x = -normal_x * boundary_probability.float()
        offset_y = -normal_y * boundary_probability.float()
        reliability = (1.0 - 4.0 * probability * (1.0 - probability)) * (
            1.0 - boundary_probability.float()
        )
        return torch.cat(
            (
                2.0 * probability - 1.0,
                normal_x,
                normal_y,
                offset_x,
                offset_y,
                reliability.clamp(0.0, 1.0),
            ),
            dim=1,
        ).to(dtype=coarse_probability.dtype)

    @staticmethod
    def _routed_memory_evidence(
        route: Mapping[str, Any],
        memory: EncoderPCMemory,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        indices = route["top_img_indices"].to(reference.device)
        valid = route["top_img_valid"].to(reference.device)
        attention = route["route_attention"].to(reference)
        keys = memory.route["route_keys"].to(reference)
        safe = indices.clamp_min(0)
        gathered = keys.index_select(0, safe.reshape(-1)).view(
            indices.shape[0], indices.shape[1], -1
        )
        weights = attention * valid.to(attention.dtype)
        return (weights.unsqueeze(-1) * gathered).sum(dim=1)


EncoderPCAdapter = EncoderPCHBMAdapter


__all__ = [
    "EncoderPCAdapter",
    "EncoderPCAdapterOutput",
    "EncoderPCHBMAdapter",
    "EncoderPCStageFlags",
]
