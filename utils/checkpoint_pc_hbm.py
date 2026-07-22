"""Strict legacy-compatible PC-HBM checkpoint and resume utilities."""

from __future__ import annotations

import os
import random
import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ARTIFACT_METADATA_VERSION = 1
ARTIFACT_METADATA_KEYS = (
    "training_design",
    "artifact_role",
    "labeled_split_fingerprint",
    "baseline_fingerprint",
    "pc_frozen",
)
TRAINING_DESIGNS = frozenset({"teacher_only", "two_stage", "joint"})
DECODER_ARCHITECTURES = frozenset({"legacy_transformer"})
DECODER_CONTRACT_VERSION = 1

ENCODER_PC_FORMAT_VERSION = 3
ENCODER_PC_ARCHITECTURE = "DINO_SCOD_ENCODER_PC_HBM"
ENCODER_PC_MODEL_ARCHITECTURE = "dino_encoder_pc_original_decoder_v1"
ENCODER_PC_ARTIFACT_KIND = "encoder_pc_model"
ENCODER_PC_RESUME_KIND = "encoder_pc_training_resume"
ENCODER_PC_MODEL_ROLES = frozenset({"base", "student"})


def _config_decoder_identity(config: Any | None) -> tuple[str | None, int | None]:
    if config is None:
        return None, None
    if isinstance(config, Mapping):
        architecture = config.get("decoder_arch") or config.get("decoder_architecture")
        contract = config.get("decoder_contract_version")
    else:
        architecture = getattr(config, "decoder_arch", None) or getattr(
            config, "decoder_architecture", None
        )
        contract = getattr(config, "decoder_contract_version", None)
    if architecture is None:
        return None, None
    architecture = str(architecture)
    if architecture not in DECODER_ARCHITECTURES:
        raise ValueError(f"Unsupported decoder architecture metadata: {architecture!r}")
    return architecture, int(contract if contract is not None else DECODER_CONTRACT_VERSION)


def _module_decoder_identity(module: nn.Module | None) -> tuple[str | None, int | None]:
    if module is None:
        return None, None
    root = _unwrap(module)
    identities: set[tuple[str, int]] = set()
    for candidate in root.modules():
        architecture = getattr(candidate, "decoder_arch", None) or getattr(
            candidate, "decoder_architecture", None
        )
        class_name = type(candidate).__name__.lower()
        if architecture is None and (
            "legacytransformerdecoder" in class_name or class_name == "decoder"
        ):
            architecture = "legacy_transformer"
        if architecture is None:
            continue
        architecture = str(architecture)
        if architecture not in DECODER_ARCHITECTURES:
            raise ValueError(f"Unsupported decoder architecture on module: {architecture!r}")
        contract = int(
            getattr(candidate, "decoder_contract_version", DECODER_CONTRACT_VERSION)
        )
        identities.add((architecture, contract))
    if not identities:
        return _config_decoder_identity(getattr(root, "pc_cfg", None))
    if len(identities) != 1:
        raise RuntimeError(f"Model contains conflicting decoder identities: {sorted(identities)}")
    return next(iter(identities))


def _checkpoint_decoder_identity(checkpoint: Mapping[str, Any]) -> tuple[str | None, int | None]:
    architecture = checkpoint.get("decoder_architecture")
    contract = checkpoint.get("decoder_contract_version")
    if architecture is None:
        return _config_decoder_identity(checkpoint.get("pc_cfg"))
    architecture = str(architecture)
    if architecture not in DECODER_ARCHITECTURES:
        raise RuntimeError(f"Checkpoint declares unsupported decoder architecture {architecture!r}")
    return architecture, int(contract if contract is not None else DECODER_CONTRACT_VERSION)


def _resolved_save_decoder_identity(
    module: nn.Module, config: Any | None
) -> tuple[str | None, int | None]:
    module_identity = _module_decoder_identity(module)
    config_identity = _config_decoder_identity(config)
    if module_identity[0] is not None and config_identity[0] is not None:
        if module_identity != config_identity:
            raise RuntimeError(
                "Decoder module/config architecture mismatch: "
                f"module={module_identity}, config={config_identity}"
            )
    return module_identity if module_identity[0] is not None else config_identity


def _attach_decoder_identity(
    payload: dict[str, Any], module: nn.Module, config: Any | None
) -> None:
    architecture, contract = _resolved_save_decoder_identity(module, config)
    if architecture is None:
        return
    payload["decoder_architecture"] = architecture
    payload["decoder_contract_version"] = int(contract)


def _validate_decoder_identity(checkpoint: Mapping[str, Any], target: nn.Module) -> None:
    # Validate a declared checkpoint identity even when the target is a small
    # wrapper without architecture attributes.  This keeps removed decoder
    # artifacts from bypassing the clean-break contract through an untagged
    # target module.
    actual_arch, actual_contract = _checkpoint_decoder_identity(checkpoint)
    expected_arch, expected_contract = _module_decoder_identity(target)
    if expected_arch is None:
        return
    if actual_arch is None:
        return
    if actual_arch != expected_arch:
        raise RuntimeError(
            f"Decoder architecture mismatch: checkpoint={actual_arch!r}, target={expected_arch!r}"
        )
    if expected_contract is not None and actual_contract != expected_contract:
        raise RuntimeError(
            "Decoder contract mismatch: "
            f"checkpoint={actual_contract!r}, target={expected_contract!r}"
        )


def load_decoder_compatible(
    decoder: nn.Module,
    source: str | os.PathLike | Mapping[str, Any],
    *,
    require_pc_complete: bool = False,
    expected_artifact_meta: Mapping[str, Any] | None = None,
):
    """Load raw/nested Decoder weights with a single ``module.`` normalization.

    A truly legacy checkpoint may omit every ``pc_hbm.*`` key.  Once a
    checkpoint contains any PC-HBM key, partial PC state is rejected.
    Non-PC missing keys and every unexpected key are always errors.
    """

    checkpoint = _load_source(source)
    if expected_artifact_meta is not None:
        validate_artifact_metadata(checkpoint, expected_artifact_meta)
    _validate_decoder_identity(checkpoint, decoder)
    state = extract_decoder_state(checkpoint)
    result = decoder.load_state_dict(state, strict=False)
    invalid_missing = [key for key in result.missing_keys if not key.startswith("pc_hbm.")]
    if invalid_missing:
        raise RuntimeError(f"Unexpected missing decoder keys: {invalid_missing}")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected decoder checkpoint keys: {result.unexpected_keys}")
    missing_pc = [key for key in result.missing_keys if key.startswith("pc_hbm.")]
    checkpoint_has_pc = any(key.startswith("pc_hbm.") for key in state)
    if missing_pc and (require_pc_complete or checkpoint_has_pc):
        raise RuntimeError(f"Incomplete PC-HBM decoder checkpoint; missing keys: {missing_pc}")
    return result


def save_decoder_checkpoint(
    path: str | os.PathLike,
    decoder: nn.Module,
    pc_cfg: Any,
    epoch: int,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    extra: Mapping[str, Any] | None = None,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save the version-2 standalone Decoder artifact."""

    payload: dict[str, Any] = {
        "format_version": 2,
        "epoch": int(epoch),
        "decoder": _unwrap(decoder).state_dict(),
        "pc_cfg": _config_dict(pc_cfg),
    }
    _attach_decoder_identity(payload, decoder, pc_cfg)
    _optional_state(payload, "optimizer", optimizer)
    _optional_state(payload, "scheduler", scheduler)
    _optional_state(payload, "scaler", scaler)
    if extra:
        payload["extra"] = dict(extra)
    _attach_artifact_metadata(payload, artifact_meta)
    _atomic_torch_save(payload, path)
    return payload


def save_memory_checkpoint(
    path: str | os.PathLike,
    memory,
    compat_meta: Mapping[str, Any] | None = None,
    *,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save memory separately so inference can load it without trainer state."""

    state = memory.state_dict()
    resolved_meta = dict(compat_meta or state.get("compat_meta", {}) or {})
    payload = {
        "format_version": 1,
        "memory": state,
        "compat_meta": resolved_meta,
    }
    _attach_artifact_metadata(payload, artifact_meta)
    _atomic_torch_save(payload, path)
    return payload


def load_memory_checkpoint(
    path: str | os.PathLike | Mapping[str, Any],
    memory,
    expected_compat: Mapping[str, Any] | None = None,
    require_producer_match: bool = True,
) -> dict[str, Any]:
    """Load CPU memory and reject an incompatible schema when requested."""

    checkpoint = _load_source(path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Memory checkpoint must be a mapping")
    memory.load_state_dict(checkpoint)
    if expected_compat is not None:
        result = memory.validate_compat(
            dict(expected_compat), require_producer_match=bool(require_producer_match)
        )
        if isinstance(result, tuple):
            compatible, reason = result
        else:
            compatible, reason = bool(result), "memory compatibility validation failed"
        if not compatible:
            raise RuntimeError(f"Incompatible PC-HBM memory: {reason}")
    if not memory.is_ready():
        raise RuntimeError("Loaded PC-HBM memory is not finalized/ready")
    return dict(checkpoint)


def save_training_resume(
    path: str | os.PathLike,
    *,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler=None,
    scaler=None,
    ema_model: nn.Module | None = None,
    pc_cfg: Any | None = None,
    extra: Mapping[str, Any] | None = None,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save exact optimizer/AMP/EMA/config/RNG state for deterministic resume."""

    payload: dict[str, Any] = {
        "format_version": 2,
        "epoch": int(epoch),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "pc_cfg": _config_dict(pc_cfg),
        "rng_state": capture_rng_state(),
    }
    _attach_decoder_identity(payload, model, pc_cfg)
    _optional_state(payload, "scheduler", scheduler)
    _optional_state(payload, "scaler", scaler)
    if ema_model is not None:
        payload["ema_model"] = _unwrap(ema_model).state_dict()
    if extra:
        payload["extra"] = dict(extra)
    _attach_artifact_metadata(payload, artifact_meta)
    _atomic_torch_save(payload, path)
    return payload


def load_training_resume(
    path: str | os.PathLike | Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema_model: nn.Module | None = None,
    restore_rng: bool = True,
    expected_artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a versioned training resume checkpoint without silent omissions."""

    checkpoint = _load_source(path)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise TypeError("Training resume checkpoint must contain a model state")
    if expected_artifact_meta is not None:
        validate_artifact_metadata(checkpoint, expected_artifact_meta)
    target_model = _unwrap(model)
    _validate_decoder_identity(checkpoint, target_model)
    state = _align_module_prefix(checkpoint["model"], target_model.state_dict())
    target_model.load_state_dict(state, strict=True)
    _restore_optional_state(checkpoint, "optimizer", optimizer)
    _restore_optional_state(checkpoint, "scheduler", scheduler)
    _restore_optional_state(checkpoint, "scaler", scaler)
    if ema_model is not None:
        if "ema_model" not in checkpoint:
            raise RuntimeError("Resume requested ema_model but checkpoint has none")
        target_ema = _unwrap(ema_model)
        ema_state = _align_module_prefix(checkpoint["ema_model"], target_ema.state_dict())
        target_ema.load_state_dict(ema_state, strict=True)
    if restore_rng:
        if "rng_state" not in checkpoint:
            raise RuntimeError("Resume checkpoint has no RNG state")
        restore_rng_state(checkpoint["rng_state"])
    return dict(checkpoint)


def save_encoder_pc_checkpoint(
    path: str | os.PathLike,
    *,
    epoch: int,
    encoder_pc_hbm: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module,
    config: Any,
    model_role: str,
    training_design: str,
    artifact_meta: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a standalone encoder-side PC-HBM artifact in format v3.

    The three trainable subsystems deliberately have independent state-dict
    entries.  This prevents a decoder-side PC-HBM checkpoint from being
    mistaken for an encoder-side artifact and makes inference loading strict.
    """

    decoder_state = _encoder_pc_decoder_state(decoder)
    payload: dict[str, Any] = {
        "format_version": ENCODER_PC_FORMAT_VERSION,
        "architecture": ENCODER_PC_ARCHITECTURE,
        "model_architecture": ENCODER_PC_MODEL_ARCHITECTURE,
        "artifact_kind": ENCODER_PC_ARTIFACT_KIND,
        "epoch": int(epoch),
        "encoder_pc_hbm": _unwrap(encoder_pc_hbm).state_dict(),
        "decoder": decoder_state,
        "pseudo_refiner": _unwrap(pseudo_refiner).state_dict(),
        "config": _encoder_pc_config_dict(config),
        "artifact_meta": _encoder_pc_artifact_metadata(
            artifact_meta,
            model_role=model_role,
            training_design=training_design,
        ),
    }
    if extra:
        payload["extra"] = dict(extra)
    _atomic_torch_save(payload, path)
    return payload


def load_encoder_pc_checkpoint(
    source: str | os.PathLike | Mapping[str, Any],
    *,
    encoder_pc_hbm: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module,
    expected_model_role: str,
    expected_training_design: str,
    expected_config: Any | Mapping[str, Any] | None = None,
    expected_artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly restore a standalone encoder-side format-v3 artifact."""

    checkpoint = _load_source(source)
    _validate_encoder_pc_checkpoint(
        checkpoint,
        expected_kind=ENCODER_PC_ARTIFACT_KIND,
        expected_model_role=expected_model_role,
        expected_training_design=expected_training_design,
        expected_artifact_meta=expected_artifact_meta,
    )
    _validate_encoder_pc_expected_config(checkpoint, expected_config)
    _load_encoder_pc_modules(
        checkpoint,
        encoder_pc_hbm=encoder_pc_hbm,
        decoder=decoder,
        pseudo_refiner=pseudo_refiner,
    )
    return dict(checkpoint)


def save_encoder_pc_training_resume(
    path: str | os.PathLike,
    *,
    epoch: int,
    encoder_pc_hbm: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module,
    optimizer,
    config: Any,
    stage_state: Any,
    split_state: Any,
    memory_profile: Any,
    model_role: str,
    training_design: str,
    scheduler=None,
    scaler=None,
    ema_adapter: nn.Module | None = None,
    ema_decoder: nn.Module | None = None,
    ema_refiner: nn.Module | None = None,
    artifact_meta: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    rng_state_by_rank: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save exact encoder-side Base/TS training state in format v3."""

    if optimizer is None:
        raise TypeError("encoder-PC training resume requires an optimizer")
    payload: dict[str, Any] = {
        "format_version": ENCODER_PC_FORMAT_VERSION,
        "architecture": ENCODER_PC_ARCHITECTURE,
        "model_architecture": ENCODER_PC_MODEL_ARCHITECTURE,
        "artifact_kind": ENCODER_PC_RESUME_KIND,
        "epoch": int(epoch),
        "encoder_pc_hbm": _unwrap(encoder_pc_hbm).state_dict(),
        "decoder": _encoder_pc_decoder_state(decoder),
        "pseudo_refiner": _unwrap(pseudo_refiner).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": _encoder_pc_config_dict(config),
        "stage_state": _plain_checkpoint_state(stage_state, name="stage_state"),
        "split_state": _plain_checkpoint_state(split_state, name="split_state"),
        "memory_profile": _plain_checkpoint_state(memory_profile, name="memory_profile"),
        "rng_state": capture_rng_state(),
        "artifact_meta": _encoder_pc_artifact_metadata(
            artifact_meta,
            model_role=model_role,
            training_design=training_design,
        ),
    }
    if rng_state_by_rank is not None:
        states = [dict(state) for state in rng_state_by_rank]
        if not states:
            raise ValueError("rng_state_by_rank must contain at least one rank state")
        payload["rng_state_by_rank"] = states
        payload["rng_state"] = states[0]
    _optional_state(payload, "scheduler", scheduler)
    _optional_state(payload, "scaler", scaler)
    _optional_module_state(payload, "ema_adapter", ema_adapter)
    _optional_module_state(payload, "ema_decoder", ema_decoder)
    _optional_module_state(payload, "ema_refiner", ema_refiner)
    if extra:
        payload["extra"] = dict(extra)
    _atomic_torch_save(payload, path)
    return payload


def load_encoder_pc_training_resume(
    source: str | os.PathLike | Mapping[str, Any],
    *,
    encoder_pc_hbm: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module,
    expected_model_role: str,
    expected_training_design: str,
    expected_config: Any | Mapping[str, Any] | None = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema_adapter: nn.Module | None = None,
    ema_decoder: nn.Module | None = None,
    ema_refiner: nn.Module | None = None,
    restore_rng: bool = True,
    expected_split_state: Any | None = None,
    expected_memory_profile: Any | None = None,
    expected_artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly restore a format-v3 Base/TS resume checkpoint."""

    checkpoint = _load_source(source)
    _validate_encoder_pc_checkpoint(
        checkpoint,
        expected_kind=ENCODER_PC_RESUME_KIND,
        expected_model_role=expected_model_role,
        expected_training_design=expected_training_design,
        expected_artifact_meta=expected_artifact_meta,
    )
    _validate_encoder_pc_expected_config(checkpoint, expected_config)
    for name in ("optimizer", "stage_state", "split_state", "memory_profile", "rng_state"):
        if name not in checkpoint:
            raise RuntimeError(f"Encoder-PC resume is missing required {name!r} state")
    if expected_split_state is not None:
        _validate_plain_checkpoint_state(
            checkpoint["split_state"], expected_split_state, name="split_state"
        )
    if expected_memory_profile is not None:
        _validate_plain_checkpoint_state(
            checkpoint["memory_profile"], expected_memory_profile, name="memory_profile"
        )
    _load_encoder_pc_modules(
        checkpoint,
        encoder_pc_hbm=encoder_pc_hbm,
        decoder=decoder,
        pseudo_refiner=pseudo_refiner,
    )
    _restore_optional_state(checkpoint, "optimizer", optimizer)
    _restore_optional_state(checkpoint, "scheduler", scheduler)
    _restore_optional_state(checkpoint, "scaler", scaler)
    _restore_optional_module_state(checkpoint, "ema_adapter", ema_adapter)
    _restore_optional_module_state(checkpoint, "ema_decoder", ema_decoder)
    _restore_optional_module_state(checkpoint, "ema_refiner", ema_refiner)
    if restore_rng:
        rank_states = checkpoint.get("rng_state_by_rank")
        if rank_states is not None:
            if not isinstance(rank_states, Sequence) or not rank_states:
                raise RuntimeError("rng_state_by_rank must be a non-empty sequence")
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )
            if len(rank_states) != world_size or rank >= len(rank_states):
                raise RuntimeError(
                    "Resume RNG rank count differs from the current distributed world"
                )
            restore_rng_state(rank_states[rank])
        else:
            restore_rng_state(checkpoint["rng_state"])
    return dict(checkpoint)


def load_original_decoder_warm_start(
    decoder: nn.Module,
    source: str | os.PathLike | Mapping[str, Any],
    *,
    drop_prefixes: tuple[str, ...] = ("pc_hbm.",),
    strict_non_pc: bool = True,
) -> dict[str, Any]:
    """Warm-start a detached original Decoder from complete non-PC weights only.

    ``pc_hbm.*`` is the only source namespace that may be discarded.  Missing
    or additional non-PC tensors are errors, so legacy optimizer, memory, and
    PC state can never be silently migrated into the new encoder-side profile.
    """

    if tuple(drop_prefixes) != ("pc_hbm.",):
        raise ValueError("Encoder-PC warm-start may drop only the 'pc_hbm.' prefix")
    if strict_non_pc is not True:
        raise ValueError("Encoder-PC warm-start requires strict_non_pc=True")

    target = _unwrap(decoder)
    target_state = target.state_dict()
    target_pc = sorted(key for key in target_state if key.startswith("pc_hbm."))
    if target_pc:
        raise RuntimeError(
            "Encoder-side warm-start requires a detached original Decoder; "
            f"target contains PC-HBM keys: {target_pc[:5]}"
        )
    target_architecture, _ = _module_decoder_identity(target)
    if target_architecture != "legacy_transformer":
        raise RuntimeError(
            "Encoder-side warm-start target is not the original Decoder: "
            f"{target_architecture!r}"
        )

    checkpoint = _load_source(source)
    source_architecture, _ = _checkpoint_decoder_identity(checkpoint)
    if source_architecture not in (None, "legacy_transformer"):
        raise RuntimeError(
            "Warm-start source is not an original Decoder checkpoint: "
            f"{source_architecture!r}"
        )
    source_state = extract_decoder_state(checkpoint)
    ignored_pc = sorted(key for key in source_state if key.startswith("pc_hbm."))
    non_pc_state = {
        key: value for key, value in source_state.items() if not key.startswith("pc_hbm.")
    }
    missing = sorted(set(target_state) - set(non_pc_state))
    unexpected = sorted(set(non_pc_state) - set(target_state))
    if missing:
        raise RuntimeError(f"Original Decoder warm-start is missing non-PC keys: {missing}")
    if unexpected:
        raise RuntimeError(
            f"Original Decoder warm-start has unexpected non-PC keys: {unexpected}"
        )
    target.load_state_dict(non_pc_state, strict=True)
    return {
        "loaded_keys": tuple(sorted(non_pc_state)),
        "ignored_pc_keys": tuple(ignored_pc),
        "source_format_version": checkpoint.get("format_version")
        if isinstance(checkpoint, Mapping)
        else None,
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    for key in ("python", "numpy", "torch"):
        if key not in state:
            raise RuntimeError(f"RNG state is missing {key!r}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def normalize_sample_key(value: str) -> str:
    """Normalize a dataset sample key without depending on the host OS."""

    if not isinstance(value, str):
        raise TypeError(f"sample key must be a string, got {type(value).__name__}")
    value = unicodedata.normalize("NFC", value.strip()).replace("\\", "/")
    parts = [part for part in value.split("/") if part and part != "."]
    normalized = "/".join(parts)
    if not normalized:
        raise ValueError("sample key must not be empty")
    return normalized


def compute_labeled_split_fingerprint(sample_keys) -> str:
    """Hash a labeled sample-key set deterministically and order-independently."""

    if isinstance(sample_keys, str):
        values = [sample_keys]
    else:
        try:
            values = list(sample_keys)
        except TypeError as error:
            raise TypeError("sample_keys must be an iterable of strings") from error
    normalized = sorted({normalize_sample_key(value) for value in values})
    if not normalized:
        raise ValueError("labeled split must contain at least one sample key")
    encoded = json.dumps(
        {"schema": "pc_hbm_labeled_split_v1", "sample_keys": normalized},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_labeled_split_fingerprint_from_indices_pt(
    indices_pt: str | os.PathLike,
    *,
    all_sample_keys=None,
) -> str:
    """Hash the string keys or integer indices stored in a labeled ``.pt`` file.

    Passing ``all_sample_keys`` resolves integer indices to stable dataset keys.
    Without it, integer identities are hashed in an explicit index namespace;
    this is still deterministic for comparing two runs using the same catalog.
    """

    try:
        values = torch.load(indices_pt, map_location="cpu", weights_only=False)
    except TypeError:
        values = torch.load(indices_pt, map_location="cpu")
    if isinstance(values, Mapping):
        candidates = [
            values.get(name)
            for name in ("sample_keys", "labeled_sample_keys", "indices", "labeled_indices")
            if name in values
        ]
        if len(candidates) != 1:
            raise TypeError(
                "labeled indices mapping must contain exactly one supported key: "
                "sample_keys, labeled_sample_keys, indices, or labeled_indices"
            )
        values = candidates[0]
    if torch.is_tensor(values):
        values = values.detach().cpu().flatten().tolist()
    elif isinstance(values, (list, tuple, set)):
        values = list(values)
    else:
        raise TypeError(
            f"unsupported labeled indices format: {type(values).__name__}; "
            "expected tensor, list, tuple, set, or supported mapping"
        )
    if not values:
        raise ValueError("labeled indices file must not be empty")
    if all(isinstance(value, str) for value in values):
        return compute_labeled_split_fingerprint(values)
    if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in values):
        raise TypeError("labeled indices must be uniformly strings or integers")
    indices = [int(value) for value in values]
    if any(index < 0 for index in indices):
        raise IndexError("labeled indices must be non-negative")
    if all_sample_keys is None:
        return compute_labeled_split_fingerprint([f"@index/{index}" for index in indices])
    catalog = list(all_sample_keys)
    out_of_range = [index for index in indices if index >= len(catalog)]
    if out_of_range:
        raise IndexError(
            f"labeled index {out_of_range[0]} is outside sample-key catalog of size {len(catalog)}"
        )
    return compute_labeled_split_fingerprint([catalog[index] for index in indices])


def build_artifact_metadata(
    *,
    training_design: str,
    artifact_role: str,
    labeled_split_fingerprint: str,
    baseline_fingerprint: str,
    pc_frozen: bool,
) -> dict[str, Any]:
    """Build validated metadata shared by Decoder, memory, and resume artifacts."""

    return _normalize_artifact_metadata(
        {
            "artifact_metadata_version": ARTIFACT_METADATA_VERSION,
            "training_design": training_design,
            "artifact_role": artifact_role,
            "labeled_split_fingerprint": labeled_split_fingerprint,
            "baseline_fingerprint": baseline_fingerprint,
            "pc_frozen": pc_frozen,
        }
    )


def read_artifact_metadata(
    source: str | os.PathLike | Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read canonical metadata from a checkpoint; return ``None`` when untagged."""

    checkpoint = _load_source(source)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    candidate = checkpoint.get("artifact_meta")
    if candidate is None and isinstance(checkpoint.get("extra"), Mapping):
        candidate = checkpoint["extra"].get("artifact_meta")
    if candidate is None and any(key in checkpoint for key in ARTIFACT_METADATA_KEYS):
        candidate = checkpoint
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise TypeError("artifact_meta must be a mapping")
    return _normalize_artifact_metadata(candidate)


def validate_artifact_metadata(
    source: str | os.PathLike | Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    allow_untagged_joint: bool = True,
) -> dict[str, Any]:
    """Validate artifact identity and prevent silent cross-design loading.

    Untagged legacy checkpoints are accepted only when the caller explicitly
    expects the ``joint`` training design.  Tagged callers may provide a
    collection of accepted designs for an explicit compatibility boundary.
    """

    if not isinstance(expected, Mapping):
        raise TypeError("expected artifact metadata must be a mapping")
    expected = dict(expected)
    expected_design = expected.get("training_design")
    if isinstance(expected_design, str):
        expected_designs = frozenset({expected_design})
    else:
        try:
            expected_designs = frozenset(expected_design)
        except TypeError as error:
            raise ValueError(
                "expected metadata must specify training_design as a supported "
                "string or a non-empty collection of supported strings"
            ) from error
    if not expected_designs or not expected_designs.issubset(TRAINING_DESIGNS):
        raise ValueError(
            "expected metadata must specify training_design using only: "
            f"{sorted(TRAINING_DESIGNS)}"
        )
    metadata = read_artifact_metadata(source)
    if metadata is None:
        # Preserve the legacy exception exactly: a disjunctive expectation that
        # happens to include ``joint`` must not make an untagged artifact valid.
        if allow_untagged_joint and expected_design == "joint":
            return {}
        raise RuntimeError(
            "Untagged legacy checkpoint is allowed only with training_design='joint'"
        )
    for key, expected_value in expected.items():
        if key not in ARTIFACT_METADATA_KEYS and key != "artifact_metadata_version":
            raise KeyError(f"unsupported expected artifact metadata key: {key}")
        if key == "training_design":
            matches = metadata.get(key) in expected_designs
        else:
            matches = expected_value is None or metadata.get(key) == expected_value
        if not matches:
            raise RuntimeError(
                f"Artifact metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {metadata.get(key)!r}"
            )
    return metadata


def extract_decoder_state(
    source: str | os.PathLike | Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract and normalize a raw or nested Decoder state dict."""

    checkpoint = _load_source(source)
    return _strip_single_module_prefix(_extract_decoder_state(checkpoint))


def extract_non_pc_decoder_state(
    source: str | os.PathLike | Mapping[str, Any],
    *,
    clone: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the legacy/raw Student portion of a Decoder checkpoint."""

    state = extract_decoder_state(source)
    legacy = {key: value for key, value in state.items() if not key.startswith("pc_hbm.")}
    if not legacy:
        raise RuntimeError("Decoder checkpoint contains no non-PC parameters or buffers")
    if clone:
        legacy = {key: value.detach().clone() for key, value in legacy.items()}
    return legacy


def state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for a normalized state mapping."""

    digest = hashlib.sha256()
    for name, value in sorted(_strip_single_module_prefix(state).items()):
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


def _extract_decoder_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Decoder checkpoint must be a state-dict mapping")
    for key in ("decoder", "student", "teacher", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and candidate and all(
            isinstance(name, str) for name in candidate
        ):
            return candidate
    if checkpoint and all(isinstance(name, str) for name in checkpoint) and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return checkpoint
    raise TypeError("Checkpoint does not contain a raw or nested Decoder state_dict")


def _strip_single_module_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        if new_key in normalized:
            raise RuntimeError(f"module. prefix normalization collided at {new_key!r}")
        normalized[new_key] = value
    return normalized


def _align_module_prefix(state, target_state):
    state = dict(state)
    target_keys = set(target_state)
    if set(state) == target_keys:
        return state
    stripped = _strip_single_module_prefix(state)
    if set(stripped) == target_keys:
        return stripped
    raise RuntimeError("Resume model keys do not exactly match the target model")


def _config_dict(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"Unsupported PC config type: {type(config).__name__}")


def _normalize_artifact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    missing = [key for key in ARTIFACT_METADATA_KEYS if key not in metadata]
    if missing:
        raise RuntimeError(f"Artifact metadata is incomplete; missing keys: {missing}")
    design = metadata["training_design"]
    if design not in TRAINING_DESIGNS:
        raise ValueError(f"Unsupported training_design: {design!r}")
    role = metadata["artifact_role"]
    if not isinstance(role, str) or not role.strip():
        raise TypeError("artifact_role must be a non-empty string")
    for key in ("labeled_split_fingerprint", "baseline_fingerprint"):
        value = metadata[key]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{key} must be a non-empty string")
    if not isinstance(metadata["pc_frozen"], bool):
        raise TypeError("pc_frozen must be a bool")
    version = metadata.get("artifact_metadata_version", ARTIFACT_METADATA_VERSION)
    if version != ARTIFACT_METADATA_VERSION:
        raise RuntimeError(
            f"Unsupported artifact metadata version {version!r}; "
            f"expected {ARTIFACT_METADATA_VERSION}"
        )
    return {
        "artifact_metadata_version": ARTIFACT_METADATA_VERSION,
        **{key: metadata[key] for key in ARTIFACT_METADATA_KEYS},
    }


def _attach_artifact_metadata(
    payload: dict[str, Any], metadata: Mapping[str, Any] | None
) -> None:
    if metadata is not None:
        payload["artifact_meta"] = _normalize_artifact_metadata(metadata)


def _encoder_pc_config_dict(config: Any) -> dict[str, Any]:
    state = _config_dict(config)
    if not isinstance(state, dict) or not state:
        raise TypeError("Encoder-PC format v3 requires a non-empty config mapping")
    architecture = state.get("architecture")
    if architecture is not None and architecture != ENCODER_PC_ARCHITECTURE:
        raise RuntimeError(
            "Encoder-PC config architecture mismatch: "
            f"expected {ENCODER_PC_ARCHITECTURE!r}, got {architecture!r}"
        )
    schema = state.get("memory_schema_version", state.get("schema_version"))
    if schema is not None and int(schema) != ENCODER_PC_FORMAT_VERSION:
        raise RuntimeError(
            "Encoder-PC config must use memory schema v3; "
            f"got {schema!r}"
        )
    return state


def _validate_encoder_pc_expected_config(
    checkpoint: Mapping[str, Any],
    expected_config: Any | Mapping[str, Any] | None,
) -> None:
    if expected_config is None:
        return
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise RuntimeError("Encoder-PC checkpoint has no valid config mapping")
    expected = _encoder_pc_config_dict(expected_config)
    saved_state = dict(saved)
    missing = sorted(set(expected) - set(saved_state))
    unexpected = sorted(set(saved_state) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected).intersection(saved_state)
        if saved_state[key] != expected[key]
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatched:
            preview = {
                key: (saved_state[key], expected[key]) for key in mismatched[:10]
            }
            details.append(f"mismatched={preview}")
        raise RuntimeError(
            "Encoder-PC checkpoint config differs from the live contract: "
            + "; ".join(details)
        )


def _encoder_pc_artifact_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    model_role: str,
    training_design: str,
) -> dict[str, Any]:
    if model_role not in ENCODER_PC_MODEL_ROLES:
        raise ValueError(
            f"Unsupported encoder-PC model_role {model_role!r}; "
            f"expected one of {sorted(ENCODER_PC_MODEL_ROLES)}"
        )
    if not isinstance(training_design, str) or not training_design.strip():
        raise TypeError("encoder-PC training_design must be a non-empty string")
    resolved = dict(metadata or {})
    for key, expected in (
        ("model_role", model_role),
        ("training_design", training_design),
    ):
        actual = resolved.get(key, expected)
        if actual != expected:
            raise RuntimeError(
                f"Encoder-PC artifact metadata {key} mismatch: "
                f"argument={expected!r}, metadata={actual!r}"
            )
        resolved[key] = expected
    resolved.setdefault("encoder_pc_metadata_version", 1)
    if resolved["encoder_pc_metadata_version"] != 1:
        raise RuntimeError(
            "Unsupported encoder-PC artifact metadata version: "
            f"{resolved['encoder_pc_metadata_version']!r}"
        )
    return resolved


def _validate_encoder_pc_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_model_role: str,
    expected_training_design: str,
    expected_artifact_meta: Mapping[str, Any] | None,
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Encoder-PC checkpoint must be a mapping")
    version = checkpoint.get("format_version")
    if version != ENCODER_PC_FORMAT_VERSION:
        raise RuntimeError(
            "Encoder-PC loader requires checkpoint format v3; "
            f"got {version!r}. Legacy format v1/v2 is not migrated."
        )
    architecture = checkpoint.get("architecture")
    if architecture != ENCODER_PC_ARCHITECTURE:
        raise RuntimeError(
            "Encoder-PC checkpoint architecture mismatch: "
            f"expected {ENCODER_PC_ARCHITECTURE!r}, got {architecture!r}"
        )
    model_architecture = checkpoint.get("model_architecture")
    if model_architecture != ENCODER_PC_MODEL_ARCHITECTURE:
        raise RuntimeError(
            "Encoder-PC model architecture mismatch: "
            f"expected {ENCODER_PC_MODEL_ARCHITECTURE!r}, "
            f"got {model_architecture!r}"
        )
    kind = checkpoint.get("artifact_kind")
    if kind != expected_kind:
        raise RuntimeError(
            f"Encoder-PC checkpoint kind mismatch: expected {expected_kind!r}, got {kind!r}"
        )
    if "config" not in checkpoint:
        raise RuntimeError("Encoder-PC checkpoint is missing required 'config' state")
    _encoder_pc_config_dict(checkpoint["config"])
    metadata = checkpoint.get("artifact_meta")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Encoder-PC checkpoint is missing artifact_meta")
    normalized = _encoder_pc_artifact_metadata(
        metadata,
        model_role=expected_model_role,
        training_design=expected_training_design,
    )
    for key, expected in dict(expected_artifact_meta or {}).items():
        if key not in normalized:
            raise RuntimeError(f"Encoder-PC artifact metadata is missing {key!r}")
        if normalized[key] != expected:
            raise RuntimeError(
                f"Encoder-PC artifact metadata mismatch for {key}: "
                f"checkpoint={normalized[key]!r}, expected={expected!r}"
            )


def _encoder_pc_decoder_state(decoder: nn.Module) -> Mapping[str, torch.Tensor]:
    target = _unwrap(decoder)
    architecture, _ = _module_decoder_identity(target)
    if architecture != "legacy_transformer":
        raise RuntimeError(
            "Encoder-PC format v3 requires the original Decoder, got "
            f"{architecture!r}"
        )
    state = target.state_dict()
    pc_keys = sorted(key for key in state if key.startswith("pc_hbm."))
    if pc_keys:
        raise RuntimeError(
            "Encoder-PC format v3 requires Decoder attach_pc=False; "
            f"found PC-HBM state keys: {pc_keys[:5]}"
        )
    return state


def _load_encoder_pc_modules(
    checkpoint: Mapping[str, Any],
    *,
    encoder_pc_hbm: nn.Module,
    decoder: nn.Module,
    pseudo_refiner: nn.Module,
) -> None:
    targets = {
        "encoder_pc_hbm": _unwrap(encoder_pc_hbm),
        "decoder": _unwrap(decoder),
        "pseudo_refiner": _unwrap(pseudo_refiner),
    }
    for name, target in targets.items():
        if name == "decoder":
            _encoder_pc_decoder_state(target)
        state = checkpoint.get(name)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"Encoder-PC checkpoint is missing {name!r} state")
        if name == "decoder":
            checkpoint_pc = sorted(key for key in state if key.startswith("pc_hbm."))
            target_pc = sorted(
                key for key in target.state_dict() if key.startswith("pc_hbm.")
            )
            if checkpoint_pc or target_pc:
                raise RuntimeError(
                    "Encoder-PC v3 Decoder must not contain pc_hbm state; "
                    f"checkpoint={checkpoint_pc[:5]}, target={target_pc[:5]}"
                )
        aligned = _align_module_prefix(state, target.state_dict())
        target.load_state_dict(aligned, strict=True)


def _plain_checkpoint_state(value: Any, *, name: str) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _plain_checkpoint_state(item, name=f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_checkpoint_state(item, name=name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"{name} must contain only dataclass/mapping/list/scalar state, "
        f"got {type(value).__name__}"
    )


def _validate_plain_checkpoint_state(actual: Any, expected: Any, *, name: str) -> None:
    expected_plain = _plain_checkpoint_state(expected, name=name)
    if actual != expected_plain:
        raise RuntimeError(
            f"Encoder-PC resume {name} mismatch: "
            f"checkpoint={actual!r}, expected={expected_plain!r}"
        )


def _optional_module_state(payload, name, module):
    if module is not None:
        payload[name] = _unwrap(module).state_dict()


def _restore_optional_module_state(checkpoint, name, module):
    if module is None:
        return
    if name not in checkpoint:
        raise RuntimeError(f"Resume requested {name} but checkpoint has none")
    target = _unwrap(module)
    state = _align_module_prefix(checkpoint[name], target.state_dict())
    target.load_state_dict(state, strict=True)


def _optional_state(payload, name, object_with_state):
    if object_with_state is not None:
        payload[name] = object_with_state.state_dict()


def _restore_optional_state(checkpoint, name, object_with_state):
    if object_with_state is None:
        return
    if name not in checkpoint:
        raise RuntimeError(f"Resume requested {name} but checkpoint has none")
    object_with_state.load_state_dict(checkpoint[name])


def _atomic_torch_save(payload, path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _load_source(source):
    if isinstance(source, Mapping):
        return source
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


# Concise aliases used by the new CLI entry points.
save_pc_hbm_decoder = save_decoder_checkpoint
save_pc_hbm_memory = save_memory_checkpoint
load_pc_hbm_memory = load_memory_checkpoint


__all__ = [
    "ARTIFACT_METADATA_KEYS",
    "ARTIFACT_METADATA_VERSION",
    "ENCODER_PC_ARCHITECTURE",
    "ENCODER_PC_ARTIFACT_KIND",
    "ENCODER_PC_FORMAT_VERSION",
    "ENCODER_PC_MODEL_ARCHITECTURE",
    "ENCODER_PC_MODEL_ROLES",
    "ENCODER_PC_RESUME_KIND",
    "TRAINING_DESIGNS",
    "build_artifact_metadata",
    "capture_rng_state",
    "compute_labeled_split_fingerprint",
    "compute_labeled_split_fingerprint_from_indices_pt",
    "extract_decoder_state",
    "extract_non_pc_decoder_state",
    "load_decoder_compatible",
    "load_encoder_pc_checkpoint",
    "load_encoder_pc_training_resume",
    "load_original_decoder_warm_start",
    "load_memory_checkpoint",
    "load_pc_hbm_memory",
    "load_training_resume",
    "normalize_sample_key",
    "read_artifact_metadata",
    "restore_rng_state",
    "save_decoder_checkpoint",
    "save_encoder_pc_checkpoint",
    "save_encoder_pc_training_resume",
    "save_memory_checkpoint",
    "save_pc_hbm_decoder",
    "save_pc_hbm_memory",
    "save_training_resume",
    "state_dict_fingerprint",
    "validate_artifact_metadata",
]
