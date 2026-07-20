"""Labeled-only memory-loader, rebuild and compatibility helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from Model.PC_HBM.encoder.encoder_memory import (
    build_encoder_memory_compat_meta as _build_encoder_memory_compat_meta,
)
from Model.PC_HBM.encoder.encoder_memory_builder import EncoderMemoryBuilder
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.dataloader import build_labeled_memory_loader


def module_fingerprint(module: nn.Module) -> str:
    """Return a deterministic SHA-256 fingerprint of a producer state dict."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(repr(value).encode("utf-8"))
            continue
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_memory_compat_meta(
    config: Any,
    producer: nn.Module | None = None,
    producer_source: str = "ema_decoder",
) -> dict[str, Any]:
    """Build the shared schema plus producer provenance for a memory export."""

    fingerprint = module_fingerprint(producer) if producer is not None else None
    if hasattr(config, "expected_memory_meta"):
        meta = dict(config.expected_memory_meta(producer_fingerprint=fingerprint))
    else:
        meta = {
            "architecture": getattr(config, "memory_architecture", "DINO_SCOD_PC_HBM"),
            "schema_version": int(getattr(config, "memory_schema_version", 1)),
            "input_size": int(getattr(config, "input_size", 392)),
            "token_hw": (int(getattr(config, "token_size", 28)),) * 2,
            "output_hw": (int(getattr(config, "output_size", 98)),) * 2,
            "dino_layer_indices": tuple(getattr(config, "dino_layer_indices", (2, 5, 8, 11))),
            "encoder_dim": int(getattr(config, "encoder_dim", 768)),
            "decoder_dim": int(getattr(config, "decoder_dim", 128)),
            "memory_dim": int(getattr(config, "memory_dim", 128)),
            "value_dim": int(getattr(config, "value_dim", 8)),
            "geometry_dim": int(getattr(config, "geometry_dim", 6)),
            "storage_dtype": str(getattr(config, "memory_storage_dtype", "float16")),
            "source": str(getattr(config, "memory_source", "labeled_only")),
        }
        if fingerprint is not None:
            meta["producer_fingerprint"] = fingerprint
    meta["producer_source"] = str(producer_source)
    if meta.get("source") != "labeled_only":
        raise ValueError("PC-HBM memory compatibility source must be labeled_only")
    return meta


def unpack_memory_batch(batch: Any) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Normalize memory batches while respecting the live RSBL dataset order."""

    if isinstance(batch, Mapping):
        image_ids = batch.get("image_ids", batch.get("sample_keys", batch.get("names")))
        images = batch.get("images", batch.get("image"))
        gts = batch.get("gts", batch.get("gt", batch.get("masks")))
    elif isinstance(batch, (tuple, list)) and len(batch) == 3:
        # Dedicated LabeledMemoryDataset contract: names, normalized images, GT.
        image_ids, images, gts = batch
    elif isinstance(batch, (tuple, list)) and len(batch) == 4:
        # PCLabeledTrainDataset contract: original, normalized images, GT, ids.
        _, images, gts, image_ids = batch
    else:
        raise TypeError(
            "Memory batch must be (names, images, gts), PC labeled four-tuple, or a mapping"
        )
    if not torch.is_tensor(images) or not torch.is_tensor(gts):
        raise TypeError("Memory images and GT masks must be tensors")
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    elif isinstance(image_ids, Sequence):
        image_ids = [str(value) for value in image_ids]
    else:
        raise TypeError("Memory image ids must be strings or a sequence of strings")
    if len(image_ids) != images.size(0) or gts.size(0) != images.size(0):
        raise ValueError("Memory batch ids/images/GT batch dimensions differ")
    if any(not value for value in image_ids):
        raise ValueError("Memory image ids must be non-empty stable sample keys")
    return image_ids, images, gts


@torch.inference_mode()
def rebuild_memory(
    model: nn.Module,
    memory_decoder: nn.Module,
    memory_loader,
    memory,
    device: torch.device | str,
    *,
    config: Any | None = None,
    compat_meta: Mapping[str, Any] | None = None,
    entry_builder: Callable[..., Mapping[str, Any]] | None = None,
    use_amp: bool = True,
):
    """Rebuild one rank's complete CPU-FP16 memory from labeled data only."""

    device = torch.device(device)
    _validate_memory_loader(memory_loader)
    feature_model = _unwrap_module(model)
    if not hasattr(feature_model, "extract_features"):
        raise AttributeError("Memory rebuild model must provide extract_features(images)")
    decoder = _unwrap_module(memory_decoder)
    if not hasattr(decoder, "forward_memory_features"):
        raise AttributeError(
            "memory_decoder must provide forward_memory_features(features, image_rgb)"
        )
    if entry_builder is None:
        engine = getattr(decoder, "pc_hbm", None)
        entry_builder = getattr(engine, "build_memory_entries", None)
    if entry_builder is None:
        raise AttributeError("PC-HBM engine must provide build_memory_entries")
    if config is not None:
        if str(getattr(config, "memory_source", "labeled_only")) != "labeled_only":
            raise ValueError("Memory rebuild only accepts labeled_only configuration")
        if bool(getattr(config, "use_unlabeled_memory_update", False)):
            raise ValueError("Unlabeled pseudo labels cannot update PC-HBM memory")

    memory.clear()
    decoder.eval()
    seen_ids: set[str] = set()
    for batch in memory_loader:
        image_ids, images, gts = unpack_memory_batch(batch)
        duplicate = seen_ids.intersection(image_ids)
        if duplicate:
            raise ValueError(f"Memory loader repeated stable image ids: {sorted(duplicate)[:5]}")
        seen_ids.update(image_ids)
        images = images.to(device=device, non_blocking=True)
        gts = gts.to(device=device, non_blocking=True)
        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            features = feature_model.extract_features(images)
            image_rgb = (
                feature_model.prepare_rgb(images)
                if hasattr(feature_model, "prepare_rgb")
                else None
            )
            memory_features = decoder.forward_memory_features(
                features, image_rgb=image_rgb
            )
            entries = entry_builder(
                features=memory_features,
                gt=gts,
                image_ids=image_ids,
            )
        if not isinstance(entries, Mapping):
            raise TypeError("build_memory_entries must return a mapping")
        memory.append(entries)

    if not seen_ids:
        raise RuntimeError("Cannot finalize an empty PC-HBM labeled memory")
    if compat_meta is None and config is not None:
        compat_meta = build_memory_compat_meta(config, decoder)
    memory.finalize(
        device=torch.device("cpu"),
        dtype=torch.float16,
        compat_meta=dict(compat_meta or {}),
    )
    if not memory.is_ready():
        raise RuntimeError("PC-HBM memory did not become ready after finalize")
    return memory


@torch.inference_mode()
def rebuild_encoder_memory(
    model: nn.Module,
    memory_adapter: nn.Module,
    memory_loader,
    memory,
    device: torch.device | str,
    *,
    config: Any | None = None,
    compat_meta: Mapping[str, Any] | None = None,
    use_amp: bool = True,
    producer_source: str = "ema_encoder_adapter",
):
    """Rebuild schema-v3 memory from raw DINO bundles and an EMA Adapter only.

    This API intentionally has no Decoder argument. GT is passed only to the
    tensor-only entry builder after the Adapter has produced every route and
    retrieval key from the unenhanced DINO feature bundle.
    """

    device = torch.device(device)
    _validate_memory_loader(memory_loader)
    feature_model = _unwrap_module(model)
    extract_bundle = getattr(feature_model, "extract_feature_bundle", None)
    if not callable(extract_bundle):
        raise AttributeError(
            "Encoder memory rebuild model must provide extract_feature_bundle(images)"
        )
    dino = getattr(feature_model, "dino", None)
    if not isinstance(dino, nn.Module):
        raise AttributeError(
            "Encoder memory rebuild model must expose the frozen DINO module"
        )
    if any(parameter.requires_grad for parameter in dino.parameters()):
        raise RuntimeError("Encoder memory rebuild requires frozen DINO weights")
    dino_weight_fingerprint = module_fingerprint(dino)
    adapter = _unwrap_module(memory_adapter)
    forward_memory_features = getattr(adapter, "forward_memory_features", None)
    if not callable(forward_memory_features):
        raise AttributeError(
            "memory_adapter must provide forward_memory_features(raw_bundle)"
        )
    if config is not None:
        if str(getattr(config, "memory_source", "labeled_only")) != "labeled_only":
            raise ValueError("Encoder memory rebuild only accepts labeled_only configuration")
        if bool(getattr(config, "use_unlabeled_memory_update", False)):
            raise ValueError("Unlabeled pseudo labels cannot update encoder PC-HBM memory")
        if int(getattr(config, "memory_schema_version", 3)) != 3:
            raise ValueError("Encoder memory rebuild requires schema v3")
        if str(getattr(config, "memory_device", "cpu")) != "cpu":
            raise ValueError("Encoder memory rebuild storage device must be CPU")
        storage_dtype = str(getattr(config, "memory_storage_dtype", "float16"))
        if storage_dtype.lower().replace("torch.", "") not in {"float16", "fp16", "half"}:
            raise ValueError("Encoder memory rebuild storage dtype must be float16")
    entry_builder = EncoderMemoryBuilder(config)

    memory.clear()
    adapter.eval()
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    for batch in memory_loader:
        image_ids, images, gts = unpack_memory_batch(batch)
        duplicate = seen_ids.intersection(image_ids)
        if duplicate:
            raise ValueError(
                f"Memory loader repeated stable image ids: {sorted(duplicate)[:5]}"
            )
        seen_ids.update(image_ids)
        ordered_ids.extend(image_ids)
        images = images.to(device=device, non_blocking=True)
        gts = gts.to(device=device, non_blocking=True)
        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            raw_bundle = extract_bundle(images)
            memory_features = forward_memory_features(raw_bundle)
            if not isinstance(memory_features, Mapping):
                raise TypeError("forward_memory_features must return a mapping")
            entries = entry_builder(
                features=memory_features,
                gt=gts,
                image_ids=image_ids,
            )
        if not isinstance(entries, Mapping):
            raise TypeError("Encoder memory entry builder must return a mapping")
        memory.append(entries)

    if not seen_ids:
        raise RuntimeError("Cannot finalize an empty encoder PC-HBM labeled memory")
    producer_fingerprint = module_fingerprint(adapter)
    split_fingerprint = compute_labeled_split_fingerprint(ordered_ids)
    generated_meta = _build_encoder_memory_compat_meta(
        dino_weight_fingerprint=dino_weight_fingerprint,
        producer_fingerprint=producer_fingerprint,
        labeled_split_fingerprint=split_fingerprint,
    )
    generated_meta.update(
        {
            "producer_source": str(producer_source),
            "labeled_image_count": len(seen_ids),
        }
    )
    if compat_meta:
        for key, value in compat_meta.items():
            if key in generated_meta and not _compat_values_equal(generated_meta[key], value):
                raise ValueError(
                    f"Encoder memory compatibility metadata conflicts with rebuilt {key}"
                )
            generated_meta[key] = value
    memory.finalize(
        device=torch.device("cpu"),
        dtype=torch.float16,
        compat_meta=generated_meta,
    )
    if not memory.is_ready():
        raise RuntimeError("Encoder PC-HBM memory did not become ready after finalize")
    compatible = memory.validate_compat(generated_meta)
    if not compatible:
        raise RuntimeError(
            f"Rebuilt encoder PC-HBM memory is incompatible: {compatible.reason}"
        )
    return memory


def _validate_memory_loader(memory_loader) -> None:
    if bool(getattr(memory_loader, "drop_last", False)):
        raise ValueError("Memory loader must use drop_last=False")
    sampler = getattr(memory_loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        raise ValueError("Each rank must iterate the complete memory set; DistributedSampler is forbidden")
    if isinstance(sampler, RandomSampler):
        raise ValueError("Memory loader must use shuffle=False")


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _compat_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _compat_values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _compat_values_equal(left[key], right[key]) for key in left
        )
    return left == right


# Public alias used by training entry points.
build_pc_memory_loader = build_labeled_memory_loader


__all__ = [
    "build_labeled_memory_loader",
    "build_memory_compat_meta",
    "build_pc_memory_loader",
    "module_fingerprint",
    "rebuild_encoder_memory",
    "rebuild_memory",
    "unpack_memory_batch",
]
