"""Training contracts for the encoder-side PC-HBM v3 profile.

This module is deliberately trainer-agnostic.  It owns the five-stage Base
curriculum, stage-specific trainability, labeled core losses, optimizer groups,
and the EMA Adapter used to produce labeled memory.  It never builds route
keys from GT and never mutates the unchanged Decoder implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.pc_hbm_dino_config import EncoderPCHBMConfig

from ..common.utils import masked_softmax
from ..encoder.child_semantic_detail_verifier import build_support_targets
from ..encoder.encoder_pc_adapter import EncoderPCHBMAdapter, EncoderPCStageFlags
from .ema import make_ema_copy, update_ema_module
from .encoder_losses import encoder_bootstrap_loss
from .losses import decoder_base_loss, probability_bce, zero_like_loss
from .supervision import (
    build_geometry_target,
    build_gt_boundary,
    build_region_label_map,
    gather_by_boundary_indices,
    normalize_boundary_indices,
)


_MISSING = object()


@dataclass(frozen=True)
class EncoderPCStage:
    """Decision-complete state for one 1-based Base-training epoch."""

    epoch: int
    index: int
    name: str
    mode: str
    enable_route_parent: bool
    enable_verification: bool
    enable_f4_f3: bool
    enable_f2_f1: bool
    enable_refiner: bool
    f4_f3_progress: float
    f2_f1_progress: float

    @classmethod
    def for_epoch(
        cls,
        epoch: int,
        config: EncoderPCHBMConfig | None = None,
    ) -> "EncoderPCStage":
        cfg = EncoderPCHBMConfig() if config is None else config
        epoch = int(epoch)
        name = str(cfg.stage_for_epoch(epoch))
        if name == "bootstrap":
            index, mode = 1, "bootstrap"
        elif name == "parent_only":
            index, mode = 2, "parent_only"
        elif name == "parent_child_f3":
            index, mode = 3, "full"
        elif name == "hierarchical_full":
            index, mode = 4, "full"
        elif name == "hierarchical_refiner":
            index, mode = 5, "full"
        else:
            raise ValueError(f"Unsupported encoder PC stage name: {name!r}")

        enable_route_parent = epoch > int(cfg.bootstrap_end_epoch)
        enable_verification = epoch > int(cfg.parent_end_epoch)
        enable_f4_f3 = enable_verification
        enable_f2_f1 = epoch > int(cfg.f4_f3_end_epoch)
        enable_refiner = epoch > int(cfg.hierarchy_end_epoch)
        f4_f3_progress = (
            float(cfg.stage_progress(epoch, level="f4_f3")) if enable_f4_f3 else 0.0
        )
        f2_f1_progress = (
            float(cfg.stage_progress(epoch, level="f2_f1")) if enable_f2_f1 else 0.0
        )
        return cls(
            epoch=epoch,
            index=index,
            name=name,
            mode=mode,
            enable_route_parent=enable_route_parent,
            enable_verification=enable_verification,
            enable_f4_f3=enable_f4_f3,
            enable_f2_f1=enable_f2_f1,
            enable_refiner=enable_refiner,
            f4_f3_progress=f4_f3_progress,
            f2_f1_progress=f2_f1_progress,
        )

    def adapter_flags(
        self,
        require_same_image_positive: bool = True,
    ) -> EncoderPCStageFlags:
        """Translate trainer stage state into the Adapter's narrow interface."""

        return EncoderPCStageFlags(
            enable_f4_f3=self.enable_f4_f3,
            f4_f3_progress=self.f4_f3_progress,
            enable_f2_f1=self.enable_f2_f1,
            f2_f1_progress=self.f2_f1_progress,
            require_same_image_positive=(
                bool(require_same_image_positive) and self.enable_route_parent
            ),
        )


def _unwrap_module(module: nn.Module) -> nn.Module:
    current = module
    for _ in range(3):
        nested = getattr(current, "module", None)
        if not isinstance(nested, nn.Module):
            break
        current = nested
    return current


def _resolve_encoder_adapter(model_or_adapter: nn.Module) -> EncoderPCHBMAdapter:
    candidate = _unwrap_module(model_or_adapter)
    if isinstance(candidate, EncoderPCHBMAdapter):
        return candidate
    for name in ("encoder_pc_hbm", "encoder_pc_adapter", "encoder_adapter", "adapter"):
        nested = getattr(candidate, name, None)
        if isinstance(nested, EncoderPCHBMAdapter):
            return nested
    raise TypeError("Expected EncoderPCHBMAdapter or a module that owns one")


def _set_trainable(module: nn.Module, enabled: bool) -> int:
    module.train(bool(enabled))
    count = 0
    for parameter in module.parameters():
        parameter.requires_grad_(bool(enabled))
        if enabled:
            count += parameter.numel()
    return count


def configure_encoder_pc_stage(
    model_or_adapter: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module | None,
    stage: EncoderPCStage,
) -> dict[str, int]:
    """Apply the exact Base-stage gradient ownership matrix.

    Decoder training is always enabled.  The Adapter progresses through
    bootstrap, route/parent, F4/F3 verification, and F2/F1 propagation.
    Refiner parameters are enabled only for epochs 21--30.
    """

    if not isinstance(stage, EncoderPCStage):
        raise TypeError("stage must be EncoderPCStage")
    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be nn.Module")
    adapter = _resolve_encoder_adapter(model_or_adapter)
    adapter.train(True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    enabled = {
        "bootstrap": True,
        "router": stage.enable_route_parent,
        "verifier": stage.enable_verification,
        "route_context": stage.enable_verification,
        "injector": stage.enable_f4_f3,
        "propagation": stage.enable_f2_f1,
    }
    trainable: dict[str, int] = {}
    for name, is_enabled in enabled.items():
        module = getattr(adapter, name, None)
        if not isinstance(module, nn.Module):
            raise AttributeError(f"Encoder Adapter is missing required module {name!r}")
        trainable[name] = _set_trainable(module, is_enabled)

    trainable["decoder"] = _set_trainable(decoder, True)
    if pseudo_refiner is not None:
        if not isinstance(pseudo_refiner, nn.Module):
            raise TypeError("pseudo_refiner must be nn.Module or None")
        trainable["pseudo_refiner"] = _set_trainable(
            pseudo_refiner, stage.enable_refiner
        )
    else:
        trainable["pseudo_refiner"] = 0
    return trainable


def _unique_parameters(
    module: nn.Module,
    *,
    group_name: str,
    seen: set[int],
) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for parameter in module.parameters():
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"Parameter is shared across optimizer groups: {group_name}")
        seen.add(identity)
        parameters.append(parameter)
    if not parameters:
        raise ValueError(f"Optimizer group {group_name!r} has no parameters")
    return parameters


def build_encoder_pc_optimizer(
    adapter: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module | None = None,
    decoder_warm_started: bool = False,
    *,
    adapter_lr: float = 1.0e-4,
    decoder_warm_lr: float = 3.0e-5,
    decoder_scratch_lr: float = 1.0e-4,
    refiner_lr: float = 1.0e-4,
) -> torch.optim.Adam:
    """Build fixed Adam groups while retaining future-stage parameters.

    Parameters are included regardless of their current ``requires_grad``
    state, so later curriculum stages can unfreeze them without rebuilding the
    optimizer.  Passing a decoder-side PC module is rejected.
    """

    resolved_adapter = _resolve_encoder_adapter(adapter)
    decoder = _unwrap_module(decoder)
    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be nn.Module")
    if any(name.startswith("pc_hbm.") for name, _ in decoder.named_parameters()):
        raise ValueError("Encoder-side training requires a Decoder without pc_hbm parameters")
    learning_rates = {
        "encoder_pc_hbm": float(adapter_lr),
        "decoder": float(decoder_warm_lr if decoder_warm_started else decoder_scratch_lr),
        "pseudo_refiner": float(refiner_lr),
    }
    if any(value <= 0.0 for value in learning_rates.values()):
        raise ValueError("All encoder PC optimizer learning rates must be positive")

    seen: set[int] = set()
    groups: list[dict[str, Any]] = [
        {
            "name": "decoder",
            "params": _unique_parameters(decoder, group_name="decoder", seen=seen),
            "lr": learning_rates["decoder"],
            "weight_decay": 0.0,
        },
        {
            "name": "encoder_pc_hbm",
            "params": _unique_parameters(
                resolved_adapter, group_name="encoder_pc_hbm", seen=seen
            ),
            "lr": learning_rates["encoder_pc_hbm"],
            "weight_decay": 0.0,
        },
    ]
    if pseudo_refiner is not None:
        pseudo_refiner = _unwrap_module(pseudo_refiner)
        groups.append(
            {
                "name": "pseudo_refiner",
                "params": _unique_parameters(
                    pseudo_refiner, group_name="pseudo_refiner", seen=seen
                ),
                "lr": learning_rates["pseudo_refiner"],
                "weight_decay": 0.0,
            }
        )
    return torch.optim.Adam(groups, weight_decay=0.0)


def make_ema_encoder_adapter(adapter: nn.Module) -> EncoderPCHBMAdapter:
    """Create the frozen EMA Adapter used exclusively for memory production."""

    resolved = _resolve_encoder_adapter(adapter)
    ema = make_ema_copy(resolved)
    if not isinstance(ema, EncoderPCHBMAdapter):
        raise TypeError("EMA copy did not preserve EncoderPCHBMAdapter type")
    return ema


@torch.no_grad()
def update_ema_encoder_adapter(
    ema: nn.Module,
    online: nn.Module,
    decay: float = 0.995,
) -> None:
    """EMA every Adapter parameter by name and copy every registered buffer."""

    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must be in [0,1]")
    ema_adapter = _resolve_encoder_adapter(ema)
    online_adapter = _resolve_encoder_adapter(online)
    update_ema_module(online_adapter, ema_adapter, momentum=decay)
    ema_adapter.eval()
    ema_adapter.requires_grad_(False)


def _field(container: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(container, Mapping):
        if name in container:
            return container[name]
    elif container is not None and hasattr(container, name):
        return getattr(container, name)
    if default is _MISSING:
        raise KeyError(f"Missing encoder PC auxiliary field: {name}")
    return default


def _aux_mapping(aux: Any) -> Mapping[str, Any]:
    if isinstance(aux, Mapping):
        return aux
    nested = getattr(aux, "aux", None)
    if isinstance(nested, Mapping):
        return nested
    raise TypeError("encoder_pc_labeled_loss requires auxiliary mappings")


def _encoder_aux(aux: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("encoder", "encoder_aux", "encoder_pc_hbm", "adapter"):
        nested = aux.get(name)
        if isinstance(nested, Mapping):
            return nested
        nested_aux = getattr(nested, "aux", None)
        if isinstance(nested_aux, Mapping):
            return nested_aux
    if "coarse_logits" in aux:
        return aux
    raise KeyError("Missing encoder-side auxiliary mapping")


def _decoder_aux(aux: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("decoder", "decoder_aux"):
        nested = aux.get(name)
        if isinstance(nested, Mapping):
            return nested
    return aux


def _required_tensor(container: Any, name: str, ndim: int | None = None) -> torch.Tensor:
    value = _field(container, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"encoder PC auxiliary field {name!r} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"encoder PC auxiliary field {name!r} must be {ndim}-D")
    return value


def _target_like(gt: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(gt, torch.Tensor):
        raise TypeError("gt must be a tensor")
    target = gt.unsqueeze(1) if gt.ndim == 3 else gt
    if target.ndim != 4 or target.size(1) != 1:
        raise ValueError("gt must be [B,H,W] or [B,1,H,W]")
    if target.size(0) != reference.size(0):
        raise ValueError("gt batch size must match predictions")
    if target.shape[-2:] != reference.shape[-2:]:
        target = F.interpolate(target.float(), size=reference.shape[-2:], mode="nearest")
    return target.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)


def _boundary_indices(
    encoder_aux: Mapping[str, Any],
    *,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    indices = encoder_aux.get("boundary_indices")
    if indices is None:
        bootstrap = encoder_aux.get("bootstrap")
        boundary_output = _field(bootstrap, "boundary_output", None)
        selected = _field(boundary_output, "selected_indices", None)
        if isinstance(selected, torch.Tensor):
            if selected.ndim != 2:
                raise ValueError("selected boundary indices must be [B,Q]")
            batch, count = selected.shape
            indices = {
                "batch_ids": torch.arange(
                    batch, device=selected.device, dtype=torch.long
                ).repeat_interleave(count),
                "flat_indices": selected.reshape(-1).long(),
            }
    normalized = normalize_boundary_indices(indices, device=device)
    if normalized is None:
        raise KeyError("Encoder auxiliary output must expose boundary_indices")
    return normalized


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = torch.broadcast_to(weight, value.shape)
    denominator = weight.sum()
    return (value * weight).sum() / denominator.clamp_min(1.0)


def _weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    if weight.shape != value.shape:
        weight = torch.broadcast_to(weight, value.shape)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _query_region_ids(
    gt: torch.Tensor,
    indices: Mapping[str, torch.Tensor],
    *,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    region_map = build_region_label_map(gt, size=size).to(device=device)
    return gather_by_boundary_indices(region_map, indices).long()


def _parent_loss(
    parent: Any,
    query_region_ids: torch.Tensor,
    reference: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    values = _required_tensor(parent, "top_parent_values", 3).to(reference)
    valid = _required_tensor(parent, "top_parent_valid", 2).to(
        device=reference.device, dtype=torch.bool
    )
    if values.shape[:2] != valid.shape or values.size(-1) != 8:
        raise ValueError("Parent values/valid tensors violate [M,K,8]/[M,K]")
    if query_region_ids.shape != (values.size(0),):
        raise ValueError("GT query regions must align with parent queries")
    attention = _field(parent, "parent_attention", None)
    if not isinstance(attention, torch.Tensor):
        scores = _required_tensor(parent, "top_parent_scores", 2).to(reference)
        attention = masked_softmax(scores / max(float(temperature), 1.0e-6), valid)
    else:
        attention = attention.to(reference)
    if attention.shape != valid.shape:
        raise ValueError("parent_attention must be [M,K]")
    region_probability = (
        attention.unsqueeze(-1) * values[..., :4].clamp(0.0, 1.0)
    ).sum(dim=1)
    selected = region_probability.gather(1, query_region_ids[:, None]).squeeze(1)
    loss = -selected.float().clamp_min(1.0e-6).log()
    query_valid = valid.any(dim=1)
    return _masked_mean(loss, query_valid)


def _child_losses(
    verification: Any,
    query_region_ids: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = _required_tensor(verification, "S_semantic", 2).to(reference)
    detail = _required_tensor(verification, "S_detail", 2).to(reference)
    candidate_regions = _required_tensor(
        verification, "top_parent_region_ids", 2
    ).to(device=reference.device, dtype=torch.long)
    valid = _required_tensor(verification, "top_parent_valid", 2).to(
        device=reference.device, dtype=torch.bool
    )
    if not (
        semantic.shape == detail.shape == candidate_regions.shape == valid.shape
    ):
        raise ValueError("Child verification score/region/valid shapes must match")
    targets = build_support_targets(query_region_ids, candidate_regions, valid)
    semantic_element = F.binary_cross_entropy_with_logits(
        semantic.float(), targets["semantic_target"].float(), reduction="none"
    )
    detail_element = F.binary_cross_entropy_with_logits(
        detail.float(), targets["detail_target"].float(), reduction="none"
    )
    semantic_weight = valid.float() * (
        1.0 + targets["semantic_hard_negative_mask"].float()
    )
    detail_weight = valid.float() * (
        1.0 + targets["detail_hard_negative_mask"].float()
    )
    return (
        _weighted_mean(semantic_element, semantic_weight),
        _weighted_mean(detail_element, detail_weight),
    )


def _geometry_loss(
    encoder_aux: Mapping[str, Any],
    verification: Any,
    gt: torch.Tensor,
    indices: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    predicted = encoder_aux.get("query_geometry")
    if not isinstance(predicted, torch.Tensor):
        predicted = _field(verification, "query_geometry", None)
    if not isinstance(predicted, torch.Tensor) or predicted.ndim != 2 or predicted.size(1) != 6:
        raise KeyError("Full encoder training requires query_geometry [M,6]")
    predicted = predicted.to(reference)
    target_maps = build_geometry_target(gt, size=size)
    target_map = torch.cat(
        (
            target_maps["sdf"],
            target_maps["normal"],
            target_maps["offset"],
            target_maps["reliability"],
        ),
        dim=1,
    ).to(device=reference.device, dtype=reference.dtype)
    target = gather_by_boundary_indices(target_map, indices)
    if target.shape != predicted.shape:
        raise ValueError("query_geometry must align with boundary query tokens")
    valid = _required_tensor(verification, "query_valid", 1).to(
        device=reference.device, dtype=torch.bool
    )
    sdf = torch.abs(predicted[:, 0] - target[:, 0])
    normal = 1.0 - F.cosine_similarity(
        predicted[:, 1:3].float(), target[:, 1:3].float(), dim=-1, eps=1.0e-6
    )
    offset = torch.abs(predicted[:, 3:5] - target[:, 3:5]).mean(dim=-1)
    reliability = target[:, 5].float().clamp(0.0, 1.0)
    weight = valid.float() * reliability
    return _weighted_mean(sdf.float() + 0.5 * normal + 0.5 * offset.float(), weight)


def _require_probability(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.numel() and bool(((value.detach() < 0.0) | (value.detach() > 1.0)).any()):
        raise ValueError(f"{name} must contain sigmoid probabilities in [0,1]")
    return value


def _gate_loss(
    encoder_aux: Mapping[str, Any],
    verification: Any,
    context: Any,
    gt: torch.Tensor,
    indices: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    gate_map = _require_probability(
        _required_tensor(context, "gate_map", 4).to(reference), "gate_map"
    )
    coarse_probability = _require_probability(
        _required_tensor(encoder_aux, "coarse_probability", 4).to(reference),
        "coarse_probability",
    )
    c23_map = _field(encoder_aux, "C23_map", None)
    if not isinstance(c23_map, torch.Tensor):
        c23_map = _required_tensor(context, "c23_map", 4)
    c23_map = c23_map.to(reference).clamp(0.0, 1.0)
    valid_map = _required_tensor(context, "valid3_map", 4).to(
        device=reference.device, dtype=torch.bool
    )
    target_map = _target_like(gt, coarse_probability)
    need_correction = (
        torch.abs(coarse_probability.detach() - target_map) > 0.25
    ).to(reference.dtype)
    gate_target_map = need_correction * (1.0 - c23_map.detach())
    gate = gather_by_boundary_indices(gate_map, indices)
    gate_target = gather_by_boundary_indices(gate_target_map, indices)
    valid = gather_by_boundary_indices(valid_map.float(), indices) > 0.5
    query_valid = _required_tensor(verification, "query_valid", 1).to(
        device=reference.device, dtype=torch.bool
    )
    valid = valid & query_valid[:, None]
    element = probability_bce(gate, gate_target, reduction="none")
    return _masked_mean(element, valid)


def _injection_loss(
    encoder_aux: Mapping[str, Any],
    gt: torch.Tensor,
    reference: torch.Tensor,
    *,
    include_f2_f1: bool,
    size: tuple[int, int],
) -> torch.Tensor:
    injection = _field(encoder_aux, "injection")
    deltas = [
        _required_tensor(injection, "f4_delta"),
        _required_tensor(injection, "f3_delta"),
    ]
    if include_f2_f1:
        propagation = _field(encoder_aux, "propagation")
        if propagation is None:
            raise KeyError("Hierarchical stage requires propagation auxiliary output")
        deltas.extend(
            (
                _required_tensor(propagation, "f2_delta"),
                _required_tensor(propagation, "f1_delta"),
            )
        )
    boundary = build_gt_boundary(gt, size=size).to(
        device=reference.device, dtype=reference.dtype
    )
    magnitude = zero_like_loss(reference)
    stable = zero_like_loss(reference)
    smooth = zero_like_loss(reference)
    for delta in deltas:
        delta = delta.to(reference)
        if delta.ndim == 3:
            batch, tokens, channels = delta.shape
            if tokens != int(size[0]) * int(size[1]):
                raise ValueError(
                    "Injection token delta must match the coarse spatial grid"
                )
            delta_map = delta.transpose(1, 2).reshape(
                batch, channels, int(size[0]), int(size[1])
            )
        elif delta.ndim == 4:
            delta_map = delta
            if delta_map.shape[-2:] != size:
                raise ValueError(
                    "Injection map delta must match the coarse spatial grid"
                )
        else:
            raise ValueError("Injection deltas must be [B,N,C] or [B,C,H,W]")
        if delta_map.size(0) != boundary.size(0):
            raise ValueError("Injection delta batch size must match GT")

        magnitude = magnitude + delta.abs().mean()
        stable = stable + (
            (1.0 - boundary) * delta_map.abs().mean(dim=1, keepdim=True)
        ).mean()
        if delta_map.size(-2) > 1:
            smooth = smooth + (
                delta_map[:, :, 1:, :] - delta_map[:, :, :-1, :]
            ).abs().mean()
        if delta_map.size(-1) > 1:
            smooth = smooth + (
                delta_map[:, :, :, 1:] - delta_map[:, :, :, :-1]
            ).abs().mean()
    return magnitude + 0.5 * stable + 0.25 * smooth


def encoder_pc_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any] | Any,
    gt: torch.Tensor,
    config: EncoderPCHBMConfig,
    stage: EncoderPCStage,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Base Decoder supervision plus the stage-specific encoder matrix.

    Route supervision is consumed exclusively from ``aux.route_info_nce``.
    Raw coarse/boundary logits use BCE-with-logits; structured gate
    probabilities use probability BCE.  GT is only consulted inside this loss
    function to construct mask, region, geometry, and gate targets.
    """

    if not isinstance(stage, EncoderPCStage):
        raise TypeError("stage must be EncoderPCStage")
    if not isinstance(config, EncoderPCHBMConfig):
        raise TypeError("config must be EncoderPCHBMConfig")
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("Decoder outputs must be (m4, m3, m2, z_core, global_logit)")
    if not all(isinstance(value, torch.Tensor) for value in outputs):
        raise TypeError("Every Decoder output must be a tensor")
    reference = outputs[3]
    combined_aux = _aux_mapping(aux)
    encoder_aux = _encoder_aux(combined_aux)
    base = decoder_base_loss(outputs, _decoder_aux(combined_aux), gt, config)

    coarse_logits = _required_tensor(encoder_aux, "coarse_logits", 4)
    boundary_logits = _required_tensor(encoder_aux, "boundary_logits", 4)
    target = _target_like(gt, coarse_logits)
    boundary_target = build_gt_boundary(
        gt, size=tuple(int(item) for item in coarse_logits.shape[-2:])
    ).to(device=coarse_logits.device, dtype=coarse_logits.dtype)
    bootstrap = encoder_bootstrap_loss(
        coarse_logits=coarse_logits,
        boundary_logits=boundary_logits,
        mask_target=target,
        boundary_target=boundary_target,
        coarse_weight=1.0,
        boundary_weight=1.0,
    )

    zero = zero_like_loss(reference)
    raw: dict[str, torch.Tensor] = {
        "L_decoder": base,
        "L_base": base,
        "L_encoder_coarse": bootstrap["coarse"],
        "L_encoder_boundary": bootstrap["boundary"],
        "L_route": zero,
        "L_parent": zero,
        "L_child_semantic": zero,
        "L_child_detail": zero,
        "L_geometry": zero,
        "L_gate": zero,
        "L_injection": zero,
    }
    indices: Mapping[str, torch.Tensor] | None = None
    query_regions: torch.Tensor | None = None
    if stage.enable_route_parent:
        route = _field(encoder_aux, "route")
        route_info_nce = _field(route, "route_info_nce", None)
        if not isinstance(route_info_nce, torch.Tensor):
            raise RuntimeError(
                "Route/parent stage requires aux['route']['route_info_nce']; "
                "route loss must not be recomputed from GT"
            )
        raw["L_route"] = route_info_nce.mean()
        indices = _boundary_indices(encoder_aux, device=reference.device)
        query_regions = _query_region_ids(
            gt,
            indices,
            size=tuple(int(item) for item in coarse_logits.shape[-2:]),
            device=reference.device,
        )
        parent = _field(encoder_aux, "parent")
        raw["L_parent"] = _parent_loss(
            parent,
            query_regions,
            reference,
            temperature=float(config.tau_parent),
        )

    if stage.enable_verification:
        if indices is None or query_regions is None:
            raise RuntimeError("Verification stage requires parent supervision state")
        verification = _field(encoder_aux, "verification")
        raw["L_child_semantic"], raw["L_child_detail"] = _child_losses(
            verification, query_regions, reference
        )
        raw["L_geometry"] = _geometry_loss(
            encoder_aux,
            verification,
            gt,
            indices,
            reference,
            size=tuple(int(item) for item in coarse_logits.shape[-2:]),
        )
        context = _field(encoder_aux, "route_context")
        raw["L_gate"] = _gate_loss(
            encoder_aux,
            verification,
            context,
            gt,
            indices,
            reference,
            size=tuple(int(item) for item in coarse_logits.shape[-2:]),
        )
        raw["L_injection"] = _injection_loss(
            encoder_aux,
            gt,
            reference,
            include_f2_f1=stage.enable_f2_f1,
            size=tuple(int(item) for item in coarse_logits.shape[-2:]),
        )

    total = (
        raw["L_decoder"]
        + float(config.lambda_coarse) * raw["L_encoder_coarse"]
        + float(config.lambda_boundary) * raw["L_encoder_boundary"]
        + float(config.lambda_route) * raw["L_route"]
        + float(config.lambda_parent) * raw["L_parent"]
        + float(config.lambda_child_semantic) * raw["L_child_semantic"]
        + float(config.lambda_child_detail) * raw["L_child_detail"]
        + float(config.lambda_geometry) * raw["L_geometry"]
        + float(config.lambda_gate) * raw["L_gate"]
        + float(config.lambda_injection) * raw["L_injection"]
    )
    raw["loss_labeled"] = total
    raw["f4_f3_progress"] = reference.new_tensor(stage.f4_f3_progress)
    raw["f2_f1_progress"] = reference.new_tensor(stage.f2_f1_progress)
    raw["refiner_enabled"] = reference.new_tensor(float(stage.enable_refiner))
    return total, {name: value.detach() for name, value in raw.items()}


__all__ = [
    "EncoderPCStage",
    "build_encoder_pc_optimizer",
    "configure_encoder_pc_stage",
    "encoder_pc_labeled_loss",
    "make_ema_encoder_adapter",
    "update_ema_encoder_adapter",
]
