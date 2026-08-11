"""Strict legacy-compatible PC-HBM checkpoint and resume utilities."""

from __future__ import annotations

import os
import copy
import random
import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ARTIFACT_METADATA_VERSION = 2
PC_HBM_ARCHITECTURE = "DINO_SCOD_PC_HBM_LITE"
PC_HBM_SCHEMA_VERSION = 2
CHILD_VERIFIER_VERSION = 2
CANONICAL_LABELED_SPLIT_COUNT = 202
CANONICAL_LABELED_SPLIT_FINGERPRINT = (
    "1f7cbfa5cd9f3afcc72910d482a762fb5bdb81b35585285d5626be6d1a2698b0"
)
ARTIFACT_METADATA_KEYS = (
    "architecture",
    "schema_version",
    "training_design",
    "artifact_role",
    "labeled_split_fingerprint",
    "baseline_fingerprint",
    "pc_frozen",
)
TRAINING_DESIGNS = frozenset({"teacher_only", "two_stage"})


@dataclass(frozen=True)
class LabeledSplitIdentity:
    """Validated identity of one run's labeled stable-key split."""

    count: int
    fingerprint: str


def load_decoder_compatible(
    decoder: nn.Module,
    source: str | os.PathLike | Mapping[str, Any],
    *,
    require_pc_complete: bool = False,
    expected_artifact_meta: Mapping[str, Any] | None = None,
    expected_pc_cfg: Any | None = None,
):
    """Load a baseline or complete V2 Lite Decoder after full preflight."""

    checkpoint = _load_source(source)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Decoder checkpoint must be a mapping")
    if expected_artifact_meta is not None:
        validate_artifact_metadata(checkpoint, expected_artifact_meta)
    state = extract_decoder_state(checkpoint)
    target_state = decoder.state_dict()
    unexpected = sorted(set(state) - set(target_state))
    if unexpected:
        raise RuntimeError(f"Unexpected decoder checkpoint keys: {unexpected}")
    invalid_missing = sorted(
        key
        for key in target_state
        if key not in state and not key.startswith("pc_hbm.")
    )
    if invalid_missing:
        raise RuntimeError(f"Unexpected missing decoder keys: {invalid_missing}")
    missing_pc = sorted(
        key for key in target_state if key.startswith("pc_hbm.") and key not in state
    )
    checkpoint_has_pc = any(key.startswith("pc_hbm.") for key in state)
    if missing_pc and (require_pc_complete or checkpoint_has_pc):
        raise RuntimeError(
            f"Incomplete PC-HBM-Lite decoder checkpoint; missing keys: {missing_pc}"
        )
    if require_pc_complete and not checkpoint_has_pc:
        raise RuntimeError("A complete PC-HBM-Lite Decoder checkpoint is required")
    if checkpoint_has_pc:
        _preflight_pc_v2(checkpoint, context="Decoder checkpoint")
        _validate_pc_config_match(
            checkpoint.get("pc_cfg"),
            (
                expected_pc_cfg
                if expected_pc_cfg is not None
                else getattr(decoder, "pc_cfg", None)
            ),
            context="Decoder checkpoint",
        )
    _validate_state_compatible(
        state,
        {key: target_state[key] for key in state},
        context="Decoder checkpoint",
    )
    candidate_decoder = copy.deepcopy(decoder)
    try:
        candidate_decoder.load_state_dict(copy.deepcopy(state), strict=False)
    except Exception as error:
        raise RuntimeError("Decoder checkpoint failed load preflight") from error
    return decoder.load_state_dict(state, strict=False)


def read_pc_config(
    source: str | os.PathLike | Mapping[str, Any],
    *,
    context: str = "PC-HBM checkpoint",
):
    """Reconstruct the canonical runtime config from a V2 Decoder/resume."""

    checkpoint = _load_source(source)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{context} must be a mapping")
    _preflight_pc_v2(checkpoint, context=context)
    raw_config = checkpoint.get("pc_cfg")
    _validate_lite_config(raw_config, context=context)
    from configs.pc_hbm_dino_config import DinoPCHBMConfig

    return DinoPCHBMConfig(**dict(raw_config))


load_pc_config_from_checkpoint = read_pc_config


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

    config_state = _config_dict(pc_cfg)
    _validate_lite_config(config_state, context="Decoder save")
    payload: dict[str, Any] = {
        "format_version": 2,
        "schema_version": PC_HBM_SCHEMA_VERSION,
        "architecture": PC_HBM_ARCHITECTURE,
        "checkpoint_type": "decoder",
        "child_verifier_version": CHILD_VERIFIER_VERSION,
        "child_verification_mode": config_state["child_verification_mode"],
        "epoch": int(epoch),
        "decoder": _unwrap(decoder).state_dict(),
        "pc_cfg": config_state,
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
    if int(state.get("format_version", -1)) != PC_HBM_SCHEMA_VERSION:
        raise RuntimeError("Memory save requires a V2 PCMemory state")
    payload = {
        "format_version": 2,
        "schema_version": PC_HBM_SCHEMA_VERSION,
        "architecture": PC_HBM_ARCHITECTURE,
        "checkpoint_type": "memory",
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
    state = checkpoint.get("memory", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Memory checkpoint state must be a mapping")
    if "memory" in checkpoint:
        _preflight_pc_v2(checkpoint, context="Memory checkpoint")
    candidate = copy.deepcopy(memory)
    candidate.load_state_dict(state)
    if expected_compat is not None:
        result = candidate.validate_compat(
            dict(expected_compat), require_producer_match=bool(require_producer_match)
        )
        if isinstance(result, tuple):
            compatible, reason = result
        else:
            compatible, reason = bool(result), "memory compatibility validation failed"
        if not compatible:
            raise RuntimeError(f"Incompatible PC-HBM memory: {reason}")
    if not candidate.is_ready():
        raise RuntimeError("Loaded PC-HBM memory is not finalized/ready")
    memory.load_state_dict(state)
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

    config_state = _config_dict(pc_cfg)
    _validate_lite_config(config_state, context="Training resume save")
    payload: dict[str, Any] = {
        "format_version": 2,
        "schema_version": PC_HBM_SCHEMA_VERSION,
        "architecture": PC_HBM_ARCHITECTURE,
        "checkpoint_type": "training_resume",
        "child_verifier_version": CHILD_VERIFIER_VERSION,
        "child_verification_mode": config_state["child_verification_mode"],
        "epoch": int(epoch),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "pc_cfg": config_state,
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
    pc_cfg: Any | None = None,
    restore_rng: bool = True,
    expected_artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a versioned training resume checkpoint without silent omissions."""

    checkpoint = _load_source(path)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise TypeError("Training resume checkpoint must contain a model state")
    _preflight_pc_v2(checkpoint, context="Training resume")
    target_model = _unwrap(model)
    resolved_pc_cfg = (
        pc_cfg
        if pc_cfg is not None
        else getattr(target_model, "pc_cfg", None)
    )
    _validate_pc_config_match(
        checkpoint.get("pc_cfg"),
        resolved_pc_cfg,
        context="Training resume",
    )
    if expected_artifact_meta is not None:
        validate_artifact_metadata(checkpoint, expected_artifact_meta)
    state = _align_module_prefix(checkpoint["model"], target_model.state_dict())
    _validate_state_compatible(
        state, target_model.state_dict(), context="Training resume model"
    )
    _preflight_module_state(
        target_model, state, context="Training resume model"
    )
    _preflight_optional_state(checkpoint, "optimizer", optimizer)
    _preflight_optional_state(checkpoint, "scheduler", scheduler)
    _preflight_optional_state(checkpoint, "scaler", scaler)
    target_ema = None
    ema_state = None
    if ema_model is not None:
        if "ema_model" not in checkpoint:
            raise RuntimeError("Resume requested ema_model but checkpoint has none")
        target_ema = _unwrap(ema_model)
        ema_state = _align_module_prefix(checkpoint["ema_model"], target_ema.state_dict())
        _validate_state_compatible(
            ema_state, target_ema.state_dict(), context="Training resume EMA"
        )
        _preflight_module_state(
            target_ema, ema_state, context="Training resume EMA"
        )
    if restore_rng:
        if "rng_state" not in checkpoint:
            raise RuntimeError("Resume checkpoint has no RNG state")
        _preflight_rng_state(checkpoint["rng_state"])

    # Mutation starts only after every model, optional state, metadata and RNG
    # contract has passed the checks above.
    target_model.load_state_dict(state, strict=True)
    _restore_optional_state(checkpoint, "optimizer", optimizer)
    _restore_optional_state(checkpoint, "scheduler", scheduler)
    _restore_optional_state(checkpoint, "scaler", scaler)
    if target_ema is not None and ema_state is not None:
        target_ema.load_state_dict(ema_state, strict=True)
    if restore_rng:
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

    values = _load_labeled_indices_values(indices_pt)
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


def _load_labeled_indices_values(
    indices_pt: str | os.PathLike,
) -> list[Any]:
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
    return values


def validate_labeled_sample_keys(
    sample_keys,
    *,
    expected_count: int | None = None,
    expected_fingerprint: str | None = None,
) -> LabeledSplitIdentity:
    """Validate a non-empty unique stable-key split for a training run."""

    if isinstance(sample_keys, str):
        raise TypeError("sample_keys must be an iterable of stable-key strings")
    try:
        values = list(sample_keys)
    except TypeError as error:
        raise TypeError(
            "sample_keys must be an iterable of stable-key strings"
        ) from error
    if not values:
        raise ValueError("labeled split must contain at least one sample key")
    if not all(isinstance(value, str) for value in values):
        raise TypeError("labeled split must store stable sample-key strings")

    normalized = [normalize_sample_key(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Labeled sample keys must be unique after normalization")
    identity = LabeledSplitIdentity(
        count=len(normalized),
        fingerprint=compute_labeled_split_fingerprint(normalized),
    )
    if expected_count is not None and identity.count != int(expected_count):
        raise RuntimeError(
            "Labeled split count differs from the current run contract: "
            f"expected={int(expected_count)}, got={identity.count}"
        )
    if (
        expected_fingerprint is not None
        and identity.fingerprint != str(expected_fingerprint)
    ):
        raise RuntimeError(
            "Labeled split fingerprint differs from the current run contract: "
            f"expected={expected_fingerprint}, got={identity.fingerprint}"
        )
    return identity


def validate_labeled_indices_pt(
    indices_pt: str | os.PathLike,
    *,
    expected_count: int | None = None,
    expected_fingerprint: str | None = None,
) -> LabeledSplitIdentity:
    """Validate a labeled key file without imposing a fixed dataset size."""

    return validate_labeled_sample_keys(
        _load_labeled_indices_values(indices_pt),
        expected_count=expected_count,
        expected_fingerprint=expected_fingerprint,
    )


def validate_labeled_sample_txt(
    sample_txt: str | os.PathLike,
    *,
    expected_count: int | None = None,
    expected_fingerprint: str | None = None,
) -> LabeledSplitIdentity:
    """Validate stable sample keys stored one per non-empty text line."""

    if sample_txt is None:
        raise ValueError("sample_txt is required when labeled_indices_pt is not set")
    sample_txt = Path(sample_txt)
    try:
        values = [
            line.strip()
            for line in sample_txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as error:
        raise FileNotFoundError(
            f"Cannot read labeled sample text file: {sample_txt}"
        ) from error
    return validate_labeled_sample_keys(
        values,
        expected_count=expected_count,
        expected_fingerprint=expected_fingerprint,
    )


def validate_labeled_split_source(
    labeled_indices_pt: str | os.PathLike | None,
    sample_txt: str | os.PathLike | None,
    *,
    expected_count: int | None = None,
    expected_fingerprint: str | None = None,
) -> LabeledSplitIdentity:
    """Match Dataset priority: a PT split overrides TXT, otherwise use TXT."""

    if labeled_indices_pt is not None:
        return validate_labeled_indices_pt(
            labeled_indices_pt,
            expected_count=expected_count,
            expected_fingerprint=expected_fingerprint,
        )
    return validate_labeled_sample_txt(
        sample_txt,
        expected_count=expected_count,
        expected_fingerprint=expected_fingerprint,
    )


def validate_labeled_split_fingerprint(
    fingerprint: str,
    *,
    expected_fingerprint: str | None = None,
) -> str:
    """Validate a SHA-256 split fingerprint and optionally its run identity."""

    fingerprint = str(fingerprint)
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError(
            "labeled split fingerprint must be a lowercase SHA-256 hex digest"
        )
    if (
        expected_fingerprint is not None
        and fingerprint != str(expected_fingerprint)
    ):
        raise RuntimeError(
            "Labeled split fingerprint differs from the current run contract: "
            f"expected={expected_fingerprint}, got={fingerprint}"
        )
    return fingerprint


def validate_canonical_labeled_split_fingerprint(
    fingerprint: str,
) -> str:
    """Reject any labeled split other than the fixed 202-key protocol."""

    fingerprint = str(fingerprint)
    if fingerprint != CANONICAL_LABELED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            "PC-HBM-Lite requires the canonical 202-key labeled split: "
            f"expected={CANONICAL_LABELED_SPLIT_FINGERPRINT}, "
            f"got={fingerprint}"
        )
    return fingerprint


def validate_canonical_labeled_indices_pt(
    indices_pt: str | os.PathLike,
) -> str:
    """Validate exact content, cardinality and identity of the fixed key file."""

    values = _load_labeled_indices_values(indices_pt)
    if len(values) != CANONICAL_LABELED_SPLIT_COUNT:
        raise RuntimeError(
            "PC-HBM-Lite benchmark key file must contain exactly "
            f"{CANONICAL_LABELED_SPLIT_COUNT} entries, got {len(values)}"
        )
    identity = validate_labeled_sample_keys(
        values,
        expected_fingerprint=CANONICAL_LABELED_SPLIT_FINGERPRINT,
    )
    return identity.fingerprint


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
            "architecture": PC_HBM_ARCHITECTURE,
            "schema_version": PC_HBM_SCHEMA_VERSION,
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
) -> dict[str, Any]:
    """Validate V2 artifact identity before any state is restored."""

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
        raise RuntimeError("Untagged or pre-V2 PC-HBM artifact is not loadable")
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
    if metadata["architecture"] != PC_HBM_ARCHITECTURE:
        raise RuntimeError(
            "Artifact architecture mismatch: "
            f"expected {PC_HBM_ARCHITECTURE!r}, "
            f"got {metadata['architecture']!r}"
        )
    if int(metadata["schema_version"]) != PC_HBM_SCHEMA_VERSION:
        raise RuntimeError(
            "Artifact schema mismatch: "
            f"expected {PC_HBM_SCHEMA_VERSION}, "
            f"got {metadata['schema_version']!r}"
        )
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


def _preflight_pc_v2(checkpoint: Mapping[str, Any], *, context: str) -> None:
    if int(checkpoint.get("format_version", -1)) != 2:
        raise RuntimeError(f"{context} must use format_version=2")
    if int(checkpoint.get("schema_version", -1)) != PC_HBM_SCHEMA_VERSION:
        raise RuntimeError(
            f"{context} must use schema_version={PC_HBM_SCHEMA_VERSION}"
        )
    if checkpoint.get("architecture") != PC_HBM_ARCHITECTURE:
        raise RuntimeError(
            f"{context} architecture must be {PC_HBM_ARCHITECTURE!r}"
        )
    if "pc_cfg" in checkpoint:
        config_state = checkpoint.get("pc_cfg")
        _validate_lite_config(config_state, context=context)
        if int(checkpoint.get("child_verifier_version", -1)) != (
            CHILD_VERIFIER_VERSION
        ):
            raise RuntimeError(
                f"{context} must use child_verifier_version="
                f"{CHILD_VERIFIER_VERSION}"
            )
        expected_mode = config_state["child_verification_mode"]
        if checkpoint.get("child_verification_mode") != expected_mode:
            raise RuntimeError(
                f"{context} child_verification_mode metadata must match pc_cfg"
            )


def _validate_lite_config(config: Any, *, context: str) -> None:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context} requires serialized PC-HBM-Lite config")
    raw = dict(config)
    if raw.get("memory_source") != "labeled_only":
        raise RuntimeError(
            f"{context} requires memory_source='labeled_only'"
        )
    if raw.get("use_unlabeled_memory_update") is not False:
        raise RuntimeError(
            f"{context} requires use_unlabeled_memory_update=False"
        )
    if raw.get("memory_device") != "cpu":
        raise RuntimeError(f"{context} requires memory_device='cpu'")
    if int(raw.get("memory_format_version", -1)) != 2:
        raise RuntimeError(
            f"{context} contains an incompatible PC memory format"
        )
    if int(raw.get("memory_schema_version", -1)) != PC_HBM_SCHEMA_VERSION:
        raise RuntimeError(
            f"{context} contains an incompatible PC memory schema"
        )
    if raw.get("memory_architecture") != PC_HBM_ARCHITECTURE:
        raise RuntimeError(
            f"{context} contains an incompatible PC architecture"
        )
    try:
        from configs.pc_hbm_dino_config import DinoPCHBMConfig

        normalized = _config_dict(DinoPCHBMConfig(**raw))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{context} contains an invalid PC-HBM-Lite config: {error}"
        ) from error
    if normalized != raw:
        differing = sorted(
            key
            for key in set(normalized or {}) | set(raw)
            if (normalized or {}).get(key) != raw.get(key)
        )
        raise RuntimeError(
            f"{context} contains a non-canonical PC-HBM-Lite config; "
            f"differing keys: {differing}"
        )


def _validate_pc_config_match(
    saved_config: Any,
    current_config: Any,
    *,
    context: str,
) -> None:
    """Require the complete runtime config before any checkpoint mutation."""

    _validate_lite_config(saved_config, context=context)
    current = _config_dict(current_config)
    if current is None:
        raise RuntimeError(
            f"{context} target must expose the complete PC-HBM-Lite config"
        )
    _validate_lite_config(current, context=f"{context} target")
    saved = dict(saved_config)
    if saved != current:
        differing = sorted(
            key
            for key in set(saved) | set(current)
            if saved.get(key) != current.get(key)
        )
        raise RuntimeError(
            f"{context} PC-HBM-Lite config mismatch; "
            f"differing keys: {differing}"
        )


def _validate_state_compatible(
    state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{context} state must be a mapping")
    missing = sorted(set(target_state) - set(state))
    unexpected = sorted(set(state) - set(target_state))
    if missing or unexpected:
        raise RuntimeError(
            f"{context} keys mismatch; missing={missing}, unexpected={unexpected}"
        )
    mismatched = []
    for key, target in target_state.items():
        value = state[key]
        if not torch.is_tensor(value) or value.shape != target.shape:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            mismatched.append(f"{key}: expected {tuple(target.shape)}, got {shape}")
    if mismatched:
        raise RuntimeError(f"{context} tensor mismatch: {mismatched}")


def _preflight_optional_state(checkpoint, name, object_with_state) -> None:
    if object_with_state is None:
        return
    if name not in checkpoint:
        raise RuntimeError(f"Resume requested {name} but checkpoint has none")
    candidate = copy.deepcopy(object_with_state)
    try:
        candidate.load_state_dict(copy.deepcopy(checkpoint[name]))
    except Exception as error:
        raise RuntimeError(f"Invalid resume {name} state") from error


def _preflight_module_state(
    module: nn.Module,
    state: Mapping[str, Any],
    *,
    context: str,
) -> None:
    candidate = copy.deepcopy(module)
    try:
        candidate.load_state_dict(copy.deepcopy(state), strict=True)
    except Exception as error:
        raise RuntimeError(f"{context} failed load preflight") from error


def _preflight_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("rng_state must be a mapping")
    required = {"python", "numpy", "torch"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"rng_state is incomplete; missing {missing}")
    if not torch.is_tensor(state["torch"]):
        raise TypeError("rng_state['torch'] must be a tensor")
    current = capture_rng_state()
    try:
        restore_rng_state(copy.deepcopy(state))
    except Exception as error:
        raise RuntimeError("Invalid resume RNG state") from error
    finally:
        restore_rng_state(current)


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


__all__ = [
    "ARTIFACT_METADATA_KEYS",
    "ARTIFACT_METADATA_VERSION",
    "CANONICAL_LABELED_SPLIT_COUNT",
    "CANONICAL_LABELED_SPLIT_FINGERPRINT",
    "LabeledSplitIdentity",
    "TRAINING_DESIGNS",
    "build_artifact_metadata",
    "capture_rng_state",
    "compute_labeled_split_fingerprint",
    "compute_labeled_split_fingerprint_from_indices_pt",
    "extract_decoder_state",
    "extract_non_pc_decoder_state",
    "load_decoder_compatible",
    "load_pc_config_from_checkpoint",
    "load_memory_checkpoint",
    "load_training_resume",
    "normalize_sample_key",
    "read_artifact_metadata",
    "read_pc_config",
    "restore_rng_state",
    "save_decoder_checkpoint",
    "save_memory_checkpoint",
    "save_training_resume",
    "state_dict_fingerprint",
    "validate_artifact_metadata",
    "validate_canonical_labeled_indices_pt",
    "validate_canonical_labeled_split_fingerprint",
    "validate_labeled_indices_pt",
    "validate_labeled_sample_txt",
    "validate_labeled_sample_keys",
    "validate_labeled_split_source",
    "validate_labeled_split_fingerprint",
]
