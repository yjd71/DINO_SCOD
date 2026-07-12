"""Strict legacy-compatible PC-HBM checkpoint and resume utilities."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def load_decoder_compatible(
    decoder: nn.Module,
    source: str | os.PathLike | Mapping[str, Any],
    *,
    require_pc_complete: bool = False,
):
    """Load raw/nested Decoder weights with a single ``module.`` normalization.

    A truly legacy checkpoint may omit every ``pc_hbm.*`` key.  Once a
    checkpoint contains any PC-HBM key, partial PC state is rejected.
    Non-PC missing keys and every unexpected key are always errors.
    """

    checkpoint = _load_source(source)
    state = _extract_decoder_state(checkpoint)
    state = _strip_single_module_prefix(state)
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
    _atomic_torch_save(payload, path)
    return payload


def save_memory_checkpoint(
    path: str | os.PathLike,
    memory,
    compat_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save memory separately so inference can load it without trainer state."""

    state = memory.state_dict()
    resolved_meta = dict(compat_meta or state.get("compat_meta", {}) or {})
    payload = {
        "format_version": 1,
        "memory": state,
        "compat_meta": resolved_meta,
    }
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
) -> dict[str, Any]:
    """Restore a versioned training resume checkpoint without silent omissions."""

    checkpoint = _load_source(path)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise TypeError("Training resume checkpoint must contain a model state")
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
    "capture_rng_state",
    "load_decoder_compatible",
    "load_memory_checkpoint",
    "load_pc_hbm_memory",
    "load_training_resume",
    "restore_rng_state",
    "save_decoder_checkpoint",
    "save_memory_checkpoint",
    "save_pc_hbm_decoder",
    "save_pc_hbm_memory",
    "save_training_resume",
]
