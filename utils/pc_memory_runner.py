"""Labeled-only memory-loader, rebuild and compatibility helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from utils.checkpoint_pc_hbm import (
    CANONICAL_LABELED_SPLIT_COUNT,
    CANONICAL_LABELED_SPLIT_FINGERPRINT,
    LabeledSplitIdentity,
    validate_labeled_sample_keys,
)


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
    producer: nn.Module,
) -> dict[str, Any]:
    """Build the exact Schema V2 contract with a required producer hash."""

    if not isinstance(producer, nn.Module):
        raise TypeError("Memory producer must be an nn.Module")
    fingerprint = module_fingerprint(producer)
    if len(fingerprint) != 64:
        raise RuntimeError("Memory producer fingerprint must be SHA-256")
    if hasattr(config, "expected_memory_meta"):
        meta = dict(config.expected_memory_meta(producer_fingerprint=fingerprint))
    else:
        meta = {
            "architecture": getattr(
                config,
                "memory_architecture",
                "DINO_SCOD_PC_HBM_LITE",
            ),
            "schema_version": int(getattr(config, "memory_schema_version", 2)),
            "input_size": int(getattr(config, "input_size", 392)),
            "token_hw": (int(getattr(config, "token_size", 28)),) * 2,
            "output_hw": (int(getattr(config, "output_size", 98)),) * 2,
            "dino_layer_indices": tuple(getattr(config, "dino_layer_indices", (2, 5, 8, 11))),
            "encoder_dim": int(getattr(config, "encoder_dim", 768)),
            "decoder_dim": int(getattr(config, "decoder_dim", 128)),
            "memory_dim": int(getattr(config, "memory_dim", 128)),
            "child_window_size": int(getattr(config, "child_window_size", 3)),
            "region_names": tuple(
                getattr(config, "region_names", ("fg_boundary", "bg_near"))
            ),
            "storage_dtype": str(getattr(config, "memory_storage_dtype", "float16")),
            "source": str(getattr(config, "memory_source", "labeled_only")),
            "producer_role": str(
                getattr(config, "memory_producer_role", "labeled_student")
            ),
            "route_environment_min_mass": float(
                getattr(config, "route_environment_min_mass", 1.0e-3)
            ),
            "fg_boundary_kernel": int(
                getattr(config, "fg_boundary_kernel", 3)
            ),
            "bg_near_kernel": int(
                getattr(config, "bg_near_kernel", 7)
            ),
            "gt_binary_threshold": float(
                getattr(config, "gt_binary_threshold", 0.5)
            ),
            "region_max_quota": tuple(
                getattr(config, "region_max_quota", (48, 48))
            ),
            "region_min_quota": tuple(
                getattr(config, "region_min_quota", (8, 8))
            ),
            "region_sampling_ratio": tuple(
                getattr(config, "region_sampling_ratio", (0.5, 0.5))
            ),
        }
        meta["producer_fingerprint"] = fingerprint
    if meta.get("source") != "labeled_only":
        raise ValueError("PC-HBM memory compatibility source must be labeled_only")
    if meta.get("producer_role") != "labeled_student":
        raise ValueError("PC-HBM memory producer_role must be labeled_student")
    if meta.get("architecture") != "DINO_SCOD_PC_HBM_LITE":
        raise ValueError(
            "PC-HBM-Lite compatibility requires architecture "
            "'DINO_SCOD_PC_HBM_LITE'"
        )
    if int(meta.get("schema_version", -1)) != 2:
        raise ValueError("PC-HBM-Lite compatibility requires schema version 2")
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
    expected_split_count: int | None = None,
    expected_split_fingerprint: str | None = None,
):
    """Rebuild one rank's CPU-resident memory from labeled data only."""

    device = torch.device(device)
    _validate_memory_loader(memory_loader)
    _validate_memory_split(
        memory_loader,
        expected_count=expected_split_count,
        expected_fingerprint=expected_split_fingerprint,
    )
    feature_model = _unwrap_module(model)
    if not hasattr(feature_model, "extract_features"):
        raise AttributeError("Memory rebuild model must provide extract_features(images)")
    decoder = _unwrap_module(memory_decoder)
    if not hasattr(decoder, "forward_memory_features"):
        raise AttributeError("memory_decoder must provide forward_memory_features(features)")
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
        if str(getattr(config, "memory_producer_role", "labeled_student")) != "labeled_student":
            raise ValueError("Memory rebuild requires producer_role=labeled_student")

    decoder_was_training = decoder.training
    memory.clear()
    decoder.eval()
    seen_ids: set[str] = set()
    try:
        for batch in memory_loader:
            image_ids, images, gts = unpack_memory_batch(batch)
            duplicate = seen_ids.intersection(image_ids)
            if duplicate:
                raise ValueError(
                    f"Memory loader repeated stable image ids: {sorted(duplicate)[:5]}"
                )
            seen_ids.update(image_ids)
            images = images.to(device=device, non_blocking=True)
            gts = gts.to(device=device, non_blocking=True)
            amp_enabled = bool(use_amp and device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                features = feature_model.extract_features(images)
                memory_features = decoder.forward_memory_features(features)
                entries = entry_builder(
                    features=memory_features,
                    gt=gts,
                    image_ids=image_ids,
                )
            if not isinstance(entries, Mapping):
                raise TypeError("build_memory_entries must return a mapping")
            if str(entries.get("source", "labeled_only")) != "labeled_only":
                raise ValueError("Memory entries must be sourced from labeled data")
            memory.append(entries)

        if not seen_ids:
            raise RuntimeError("Cannot finalize an empty PC-HBM labeled memory")
        if compat_meta is None and config is not None:
            compat_meta = build_memory_compat_meta(config, decoder)
        if dict(compat_meta or {}).get("producer_role") != "labeled_student":
            raise ValueError("Memory compat metadata must identify labeled_student")
        memory.finalize(
            device=torch.device("cpu"),
            dtype=memory.storage_dtype,
            compat_meta=dict(compat_meta or {}),
        )
        if not memory.is_ready():
            raise RuntimeError("PC-HBM memory did not become ready after finalize")
        return memory
    finally:
        decoder.train(decoder_was_training)


def _validate_memory_loader(memory_loader) -> None:
    if bool(getattr(memory_loader, "drop_last", False)):
        raise ValueError("Memory loader must use drop_last=False")
    sampler = getattr(memory_loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        raise ValueError("Each rank must iterate the complete memory set; DistributedSampler is forbidden")
    if isinstance(sampler, RandomSampler):
        raise ValueError("Memory loader must use shuffle=False")


def _validate_memory_split(
    memory_loader,
    *,
    expected_count: int | None = None,
    expected_fingerprint: str | None = None,
) -> LabeledSplitIdentity:
    dataset = getattr(memory_loader, "dataset", None)
    sample_keys = getattr(dataset, "sample_keys", None)
    if sample_keys is None:
        raise ValueError(
            "Memory loader dataset must expose stable sample_keys"
        )
    return validate_labeled_sample_keys(
        sample_keys,
        expected_count=expected_count,
        expected_fingerprint=expected_fingerprint,
    )


def _validate_canonical_memory_split(memory_loader) -> None:
    """Retain the fixed 202-key check for reproducible benchmark tools."""

    dataset = getattr(memory_loader, "dataset", None)
    sample_keys = getattr(dataset, "sample_keys", None)
    if sample_keys is not None and len(sample_keys) != CANONICAL_LABELED_SPLIT_COUNT:
        raise RuntimeError(
            "Benchmark memory rebuild requires exactly "
            f"{CANONICAL_LABELED_SPLIT_COUNT} labeled samples, "
            f"got {len(sample_keys)}"
        )
    _validate_memory_split(
        memory_loader,
        expected_count=CANONICAL_LABELED_SPLIT_COUNT,
        expected_fingerprint=CANONICAL_LABELED_SPLIT_FINGERPRINT,
    )


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


__all__ = [
    "build_memory_compat_meta",
    "module_fingerprint",
    "rebuild_memory",
    "unpack_memory_batch",
    "_validate_canonical_memory_split",
    "_validate_memory_split",
]
