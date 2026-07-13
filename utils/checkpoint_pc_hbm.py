"""Strict legacy-compatible PC-HBM checkpoint and resume utilities."""

from __future__ import annotations

import os
import random
import hashlib
import json
import unicodedata
from collections.abc import Mapping
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
    require_producer_match: bool = False,
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
    "TRAINING_DESIGNS",
    "build_artifact_metadata",
    "capture_rng_state",
    "compute_labeled_split_fingerprint",
    "compute_labeled_split_fingerprint_from_indices_pt",
    "extract_decoder_state",
    "extract_non_pc_decoder_state",
    "load_decoder_compatible",
    "load_memory_checkpoint",
    "load_pc_hbm_memory",
    "load_training_resume",
    "normalize_sample_key",
    "read_artifact_metadata",
    "restore_rng_state",
    "save_decoder_checkpoint",
    "save_memory_checkpoint",
    "save_pc_hbm_decoder",
    "save_pc_hbm_memory",
    "save_training_resume",
    "state_dict_fingerprint",
    "validate_artifact_metadata",
]
