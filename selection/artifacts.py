"""Deterministic, KMeans-free artifact I/O for sampling pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    normalize_sample_key,
)


def _canonicalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return _canonicalize_for_json(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_for_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonicalize_for_json(value.item())
    if isinstance(value, torch.Tensor):
        return _canonicalize_for_json(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata contains NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _canonicalize_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"fingerprint source is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_unique_keys(sample_keys: Sequence[str], *, name: str) -> list[str]:
    if isinstance(sample_keys, (str, bytes)) or not isinstance(sample_keys, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    keys: list[str] = []
    for value in sample_keys:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings")
        keys.append(normalize_sample_key(value))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains duplicate keys")
    return keys


def compute_catalog_fingerprint(sample_keys: Sequence[str]) -> str:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    return stable_fingerprint({"version": 1, "sample_keys": sorted(keys)})


def compute_key_order_fingerprint(sample_keys: Sequence[str]) -> str:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    return stable_fingerprint({"version": 1, "sample_keys": keys})


def compute_image_fingerprint(
    sample_keys: Sequence[str],
    image_paths: Sequence[str | os.PathLike[str]]
    | Mapping[str, str | os.PathLike[str]],
) -> str:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    if isinstance(image_paths, Mapping):
        normalized_paths: dict[str, Path] = {}
        for raw_key, raw_path in image_paths.items():
            key = normalize_sample_key(str(raw_key))
            if key in normalized_paths:
                raise ValueError(f"image path mapping contains duplicate key: {key}")
            normalized_paths[key] = Path(raw_path)
        if set(normalized_paths) != set(keys):
            missing = sorted(set(keys) - set(normalized_paths))
            extra = sorted(set(normalized_paths) - set(keys))
            raise ValueError(
                "image path mapping does not match sample keys; "
                f"missing={missing[:3]!r}, extra={extra[:3]!r}"
            )
        pairs = [(key, normalized_paths[key]) for key in keys]
    else:
        paths = [Path(path) for path in image_paths]
        if len(paths) != len(keys):
            raise ValueError("image path count does not match sample key count")
        pairs = list(zip(keys, paths))

    digest = hashlib.sha256(b"selection-image-content-v1\0")
    for key, path in sorted(pairs, key=lambda pair: pair[0]):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _torch_load(path: str | os.PathLike[str]) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _payload_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError, RuntimeError):
        return False


def atomic_torch_save(
    payload: Any,
    path: str | os.PathLike[str],
    *,
    refuse_mismatch: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = _torch_load(target)
        if _payload_equal(existing, payload):
            return
        if refuse_mismatch:
            raise FileExistsError(f"refusing to overwrite different artifact: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_save(
    payload: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    refuse_mismatch: bool = True,
) -> None:
    canonical = _canonicalize_for_json(payload)
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"existing JSON artifact is invalid: {target}") from error
        if existing == canonical:
            return
        if refuse_mismatch:
            raise FileExistsError(f"refusing to overwrite different artifact: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent), text=True
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_split_keys(
    sample_keys: Sequence[str],
    *,
    catalog_keys: Sequence[str] | None = None,
    expected_count: int | None = None,
    require_sorted: bool = True,
) -> list[str]:
    keys = _normalize_unique_keys(sample_keys, name="split")
    if require_sorted and keys != sorted(keys):
        raise ValueError("split keys must be sorted")
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError(
            f"split must contain exactly {expected_count} keys, got {len(keys)}"
        )
    if catalog_keys is not None:
        catalog = set(_normalize_unique_keys(catalog_keys, name="catalog_keys"))
        missing = sorted(set(keys) - catalog)
        if missing:
            raise ValueError(f"split keys are absent from the catalog: {missing[:3]!r}")
    return keys


def save_split_keys(
    path: str | os.PathLike[str], sample_keys: Sequence[str]
) -> str:
    normalized = sorted(_normalize_unique_keys(sample_keys, name="split"))
    if not normalized:
        raise ValueError("split must contain at least one sample key")
    atomic_torch_save(normalized, path, refuse_mismatch=True)
    return compute_labeled_split_fingerprint(normalized)


def load_split_keys(
    path: str | os.PathLike[str],
    *,
    catalog_keys: Sequence[str] | None = None,
    expected_count: int | None = None,
) -> list[str]:
    payload = _torch_load(path)
    if not isinstance(payload, list) or not all(isinstance(key, str) for key in payload):
        raise ValueError("split artifact must contain a plain list[str]")
    return validate_split_keys(
        payload,
        catalog_keys=catalog_keys,
        expected_count=expected_count,
        require_sorted=True,
    )


__all__ = [
    "atomic_json_save",
    "atomic_torch_save",
    "compute_catalog_fingerprint",
    "compute_image_fingerprint",
    "compute_key_order_fingerprint",
    "file_sha256",
    "load_split_keys",
    "save_split_keys",
    "stable_fingerprint",
    "validate_split_keys",
]
