from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sklearn
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from configs.pc_bacs_config import PCBACSConfig
from Model.base_model import BaseModel
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    extract_non_pc_decoder_state,
    read_artifact_metadata,
    state_dict_fingerprint,
)
from utils.dataloader import SelectionPoolDataset
from utils.pc_bacs import (
    atomic_json_save,
    atomic_torch_save,
    build_nested_splits,
    compute_pc_bacs_score,
    fit_dino_kmeans,
    normalize_dino_features,
    save_split_keys,
)


REPO_ROOT = Path(__file__).resolve().parent
DINO_WEIGHT_PATH = REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth"
FEATURE_CACHE_VERSION = 1
SCORE_CACHE_VERSION = 1
SELECTOR_EPOCHS = 5
SELECTOR_TRAINING_SEED_COUNT = 40
PREPROCESS_SPEC = {
    "color": "opencv_bgr_to_rgb_to_pil",
    "input_size": 392,
    "resize": "bilinear",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline PC-HBM-oriented Boundary-Aware Coverage Sampling"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-seed-only",
        action="store_true",
        help="Build the keyed DINO cache and one KMeans-center seed per cluster.",
    )
    mode.add_argument(
        "--convert-split-only",
        action="store_true",
        help="Convert a legacy txt/pt split to canonical stable string keys.",
    )

    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument(
        "--train-sets", nargs="+", default=["TR-CAMO", "TR-COD10K"]
    )
    parser.add_argument("--seed-split", type=Path)
    parser.add_argument(
        "--selector-seed-split",
        type=Path,
        help=(
            "Seed split used to train the selector checkpoint. Defaults to "
            "--seed-split; pass the full 40-key seed for a reduced smoke pool."
        ),
    )
    parser.add_argument("--selector-checkpoint", type=Path)

    parser.add_argument("--features-path", type=Path)
    parser.add_argument("--scores-path", type=Path)
    parser.add_argument("--rebuild-features", action="store_true")
    score_cache_mode = parser.add_mutually_exclusive_group()
    score_cache_mode.add_argument("--reuse-scores", action="store_true")
    score_cache_mode.add_argument("--rebuild-scores", action="store_true")

    parser.add_argument("--n-clusters", type=int, default=40)
    parser.add_argument(
        "--target-counts", nargs="+", type=int, default=[41, 202, 404]
    )
    parser.add_argument("--dedup-threshold", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Deterministic smoke-only pool size. Never use for formal selection.",
    )
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_fingerprint(items: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"pc_bacs_catalog_v1\n")
    seen: set[str] = set()
    for item in sorted(items, key=lambda record: str(record["key"])):
        key = str(item["key"]).replace("\\", "/")
        if key in seen:
            raise ValueError(f"Duplicate catalog sample key: {key}")
        seen.add(key)
        image_path = Path(str(item["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Catalog image does not exist for {key}: {image_path}")
        record = "\t".join(
            (
                key,
                image_path.suffix.lower(),
                str(image_path.stat().st_size),
                _sha256_file(image_path),
            )
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = (
        REPO_ROOT / "configs" / "pc_bacs_config.py",
        REPO_ROOT / "utils" / "pc_bacs.py",
        REPO_ROOT / "utils" / "dataloader.py",
        Path(__file__).resolve(),
    )
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _repo_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _resolve_paths(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    if args.output_dir is None:
        args.output_dir = args.data_root / "splits" / "pc_bacs"
    else:
        args.output_dir = args.output_dir.resolve()
    if args.features_path is None:
        args.features_path = (
            args.data_root / "cache" / "pc_bacs_dino_vitb14_392.pt"
        )
    else:
        args.features_path = args.features_path.resolve()
    if args.seed_split is not None:
        args.seed_split = args.seed_split.resolve()
    if args.selector_seed_split is None and args.seed_split is not None:
        args.selector_seed_split = args.seed_split
    elif args.selector_seed_split is not None:
        args.selector_seed_split = args.selector_seed_split.resolve()
    if args.selector_checkpoint is not None:
        args.selector_checkpoint = args.selector_checkpoint.resolve()
    if args.scores_path is not None:
        args.scores_path = args.scores_path.resolve()


def _validate_args(args: argparse.Namespace, config: PCBACSConfig) -> None:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    for train_set in args.train_sets:
        image_root = args.data_root / train_set / "im"
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"Training image directory does not exist: {image_root}"
            )
    if args.feature_batch_size <= 0 or args.score_batch_size <= 0:
        raise ValueError("Feature and score batch sizes must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    if args.max_samples is not None:
        formal_output = (args.data_root / "splits" / "pc_bacs").resolve()
        formal_features = (
            args.data_root / "cache" / "pc_bacs_dino_vitb14_392.pt"
        ).resolve()
        if args.output_dir == formal_output:
            raise ValueError(
                "--max-samples is smoke-only and requires a non-formal --output-dir."
            )
        if args.features_path == formal_features:
            raise ValueError(
                "--max-samples requires a separate smoke --features-path."
            )
    if (
        not args.convert_split_only
        and args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    if not args.convert_split_only and not DINO_WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            f"The repository DINO weight is missing: {DINO_WEIGHT_PATH}"
        )
    if args.build_seed_only:
        if args.selector_checkpoint is not None:
            raise ValueError("--build-seed-only does not accept a selector checkpoint.")
    elif args.convert_split_only:
        if args.seed_split is None:
            raise ValueError("--convert-split-only requires --seed-split.")
    else:
        if args.seed_split is None or not args.seed_split.is_file():
            raise FileNotFoundError(
                f"Default selection requires an existing --seed-split: {args.seed_split}"
            )
        if args.selector_checkpoint is None or not args.selector_checkpoint.is_file():
            raise FileNotFoundError(
                "Default selection requires an existing --selector-checkpoint: "
                f"{args.selector_checkpoint}"
            )
        if (
            args.selector_seed_split is None
            or not args.selector_seed_split.is_file()
        ):
            raise FileNotFoundError(
                "Default selection requires an existing selector training seed via "
                f"--selector-seed-split: {args.selector_seed_split}"
            )
    config.validate(None)


def _image_roots(args: argparse.Namespace) -> list[str]:
    return [str(args.data_root / name / "im") for name in args.train_sets]


def _active_pool(
    dataset: SelectionPoolDataset,
    max_samples: int | None,
    seed: int,
) -> tuple[Dataset, list[dict[str, Any]], list[str]]:
    items = [dict(item) for item in dataset.items]
    if max_samples is None or max_samples >= len(items):
        return dataset, items, [str(item["key"]) for item in items]
    rng = random.Random(seed)
    indices = rng.sample(range(len(items)), max_samples)
    indices.sort(key=lambda index: str(items[index]["key"]))
    active_items = [items[index] for index in indices]
    return (
        Subset(dataset, indices),
        active_items,
        [str(item["key"]) for item in active_items],
    )


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


@torch.inference_mode()
def _extract_dino_features(
    model: BaseModel,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[list[str], torch.Tensor]:
    model.dino.eval()
    keys: list[str] = []
    features: list[torch.Tensor] = []
    for batch_keys, images in loader:
        images = images.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            batch_features = model.dino(images)
        if batch_features.ndim != 2 or batch_features.shape[1] != 768:
            raise RuntimeError(
                "Expected DINO global features [B,768], got "
                f"{tuple(batch_features.shape)}"
            )
        keys.extend(str(key) for key in batch_keys)
        features.append(batch_features.float().cpu())
    if not features:
        raise RuntimeError("DINO feature extraction produced no batches.")
    return keys, torch.cat(features, dim=0)


def _feature_spec(
    *,
    catalog_fingerprint: str,
    dino_fingerprint: str,
    use_amp: bool,
    device_type: str,
) -> dict[str, Any]:
    payload = {
        "format_version": FEATURE_CACHE_VERSION,
        "feature_type": "dinov2_vitb14_global",
        "feature_definition": "model.dino(normalized_392_rgb)",
        "feature_dim": 768,
        "catalog_fingerprint": catalog_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
        "feature_amp": bool(use_amp and device_type == "cuda"),
        "feature_device_type": device_type,
    }
    payload["feature_spec_fingerprint"] = _sha256_json(payload)
    return payload


def _validate_feature_cache(
    payload: Mapping[str, Any],
    *,
    sample_keys: Sequence[str],
    expected_spec: Mapping[str, Any],
) -> torch.Tensor:
    mismatches: list[str] = []
    for key, value in expected_spec.items():
        if payload.get(key) != value:
            mismatches.append(
                f"{key}: expected {value!r}, got {payload.get(key)!r}"
            )
    cached_keys = payload.get("sample_keys")
    if list(cached_keys or []) != list(sample_keys):
        mismatches.append("sample_keys differ from the active stable catalog")
    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        mismatches.append("features is not a tensor")
    elif tuple(features.shape) != (len(sample_keys), 768):
        mismatches.append(
            f"features shape must be {(len(sample_keys), 768)}, got {tuple(features.shape)}"
        )
    elif not torch.isfinite(features.float()).all():
        mismatches.append("features contains NaN or Inf")
    if mismatches:
        raise ValueError(
            "Feature cache mismatch; rerun with --rebuild-features:\n- "
            + "\n- ".join(mismatches)
        )
    return features.float().cpu()


def _load_or_extract_features(
    *,
    args: argparse.Namespace,
    model: BaseModel,
    loader: DataLoader,
    device: torch.device,
    sample_keys: Sequence[str],
    catalog_fingerprint: str,
    dino_fingerprint: str,
) -> tuple[torch.Tensor, bool]:
    expected_spec = _feature_spec(
        catalog_fingerprint=catalog_fingerprint,
        dino_fingerprint=dino_fingerprint,
        use_amp=args.amp,
        device_type=args.device,
    )
    if args.features_path.suffix.lower() == ".npy":
        raise ValueError(
            "Legacy .npy features have no stable key catalog and are not reusable. "
            "Use the keyed .pt feature cache."
        )
    if args.features_path.is_file() and not args.rebuild_features:
        payload = torch.load(args.features_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("Feature cache must contain a mapping payload.")
        return (
            _validate_feature_cache(
                payload, sample_keys=sample_keys, expected_spec=expected_spec
            ),
            True,
        )
    extracted_keys, features = _extract_dino_features(
        model, loader, device, use_amp=args.amp
    )
    if extracted_keys != list(sample_keys):
        raise RuntimeError("DINO feature key order diverged from the stable catalog.")
    payload = dict(expected_spec)
    payload.update(
        {
            "sample_keys": list(sample_keys),
            "features": features.float().cpu(),
            "normalized": False,
            "weight_path": DINO_WEIGHT_PATH.relative_to(REPO_ROOT).as_posix(),
        }
    )
    atomic_torch_save(
        payload,
        args.features_path,
        refuse_mismatch=not args.rebuild_features,
    )
    return features, False


def _score_spec(
    *,
    catalog_fingerprint: str,
    selector_fingerprint: str,
    dino_fingerprint: str,
    args: argparse.Namespace,
    config: PCBACSConfig,
) -> dict[str, Any]:
    payload = {
        "format_version": SCORE_CACHE_VERSION,
        "catalog_fingerprint": catalog_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
        "score_formula_version": config.score_formula_version,
        "output_index": 3,
        "output_size": config.output_size,
        "transform": "horizontal_flip",
        "sobel_padding": "replicate",
        "sobel_magnitude": "torch.hypot",
        "eps_location": "boundary_denominator_only",
        "eps": config.eps,
        "amp": bool(args.amp and args.device == "cuda"),
        "device_type": args.device,
    }
    payload["score_spec_fingerprint"] = _sha256_json(payload)
    return payload


def _load_selector_state(
    checkpoint_path: Path,
    *,
    selector_seed_keys: Sequence[str],
    dino_fingerprint: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Selector checkpoint must contain a mapping payload.")
    if int(payload.get("epoch", -1)) != SELECTOR_EPOCHS:
        raise ValueError(
            f"Selector checkpoint must be the completed epoch-5 artifact, got epoch={payload.get('epoch')!r}."
        )
    metadata = read_artifact_metadata(payload)
    if metadata is None:
        raise ValueError("Selector checkpoint is missing strict artifact metadata.")
    expected_seed_fingerprint = compute_labeled_split_fingerprint(selector_seed_keys)
    expected_metadata = {
        "training_design": "two_stage",
        "artifact_role": "teacher_enhancer",
        "labeled_split_fingerprint": expected_seed_fingerprint,
        "pc_frozen": True,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {metadata.get(key)!r}"
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError("Selector checkpoint identity mismatch:\n- " + "\n- ".join(mismatches))
    legacy_state = extract_non_pc_decoder_state(payload)
    non_pc_fingerprint = state_dict_fingerprint(legacy_state)
    selector_fingerprint = _sha256_json(
        {
            "architecture": "dinov2_vitb14_rsbl_legacy_decoder",
            "dino_fingerprint": dino_fingerprint,
            "non_pc_decoder_fingerprint": non_pc_fingerprint,
            "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
        }
    )
    return legacy_state, dict(metadata), non_pc_fingerprint, selector_fingerprint


def _resolve_score_cache_path(
    args: argparse.Namespace, expected_spec: Mapping[str, Any]
) -> Path:
    if args.scores_path is None:
        selector_short = str(expected_spec["selector_fingerprint"])[:12]
        catalog_short = str(expected_spec["catalog_fingerprint"])[:12]
        args.scores_path = (
            args.data_root
            / "cache"
            / f"pc_bacs_scores_{selector_short}_{catalog_short}.pt"
        )
    return args.scores_path


def _validate_score_cache(
    payload: Mapping[str, Any],
    *,
    sample_keys: Sequence[str],
    expected_spec: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mismatches: list[str] = []
    for key, value in expected_spec.items():
        if payload.get(key) != value:
            mismatches.append(
                f"{key}: expected {value!r}, got {payload.get(key)!r}"
            )
    if list(payload.get("sample_keys") or []) != list(sample_keys):
        mismatches.append("sample_keys differ from the active stable catalog")
    tensors: list[torch.Tensor] = []
    for name in ("boundary_disagreement", "global_disagreement", "scores"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            mismatches.append(f"{name} is not a tensor")
            continue
        value = value.float().cpu()
        if tuple(value.shape) != (len(sample_keys),):
            mismatches.append(
                f"{name} shape must be {(len(sample_keys),)}, got {tuple(value.shape)}"
            )
        elif not torch.isfinite(value).all():
            mismatches.append(f"{name} contains NaN or Inf")
        elif bool(((value < -1e-6) | (value > 1.0 + 1e-6)).any()):
            mismatches.append(f"{name} contains values outside [0, 1]")
        tensors.append(value)
    if len(tensors) == 3 and all(
        tuple(value.shape) == (len(sample_keys),) for value in tensors
    ):
        boundary, global_disagreement, scores = tensors
        expected_scores = boundary * (1.0 - global_disagreement)
        if not torch.allclose(scores, expected_scores, rtol=1e-6, atol=1e-7):
            mismatches.append(
                "scores do not satisfy score = boundary_disagreement * "
                "(1 - global_disagreement)"
            )
    if mismatches:
        raise ValueError("Score cache mismatch:\n- " + "\n- ".join(mismatches))
    return tensors[0], tensors[1], tensors[2]


@torch.inference_mode()
def _score_pool(
    model: BaseModel,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    eps: float,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    keys: list[str] = []
    boundary_values: list[torch.Tensor] = []
    global_values: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    for batch_keys, images in loader:
        images = images.to(device, non_blocking=True)
        two_views = torch.cat((images, torch.flip(images, dims=(-1,))), dim=0)
        with _autocast(device, use_amp):
            outputs = model(two_views, pc_mode="off")
            logits = outputs[3]
        if logits.ndim != 4 or logits.shape[1:] != (1, 98, 98):
            raise RuntimeError(
                f"Expected selector z_main [2B,1,98,98], got {tuple(logits.shape)}"
            )
        probability, transformed_probability = logits.sigmoid().chunk(2, dim=0)
        transformed_probability = torch.flip(transformed_probability, dims=(-1,))
        result = compute_pc_bacs_score(
            probability, transformed_probability, eps=eps
        )
        keys.extend(str(key) for key in batch_keys)
        boundary_values.append(result["boundary_disagreement"].float().cpu())
        global_values.append(result["global_disagreement"].float().cpu())
        scores.append(result["score"].float().cpu())
    return (
        keys,
        torch.cat(boundary_values),
        torch.cat(global_values),
        torch.cat(scores),
    )


def _load_or_score_pool(
    *,
    args: argparse.Namespace,
    model: BaseModel,
    loader: DataLoader,
    device: torch.device,
    sample_keys: Sequence[str],
    expected_spec: Mapping[str, Any],
    config: PCBACSConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    _resolve_score_cache_path(args, expected_spec)
    if args.reuse_scores:
        if not args.scores_path.is_file():
            raise FileNotFoundError(
                f"--reuse-scores requested but cache is missing: {args.scores_path}"
            )
        payload = torch.load(args.scores_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("Score cache must contain a mapping payload.")
        boundary, global_disagreement, scores = _validate_score_cache(
            payload, sample_keys=sample_keys, expected_spec=expected_spec
        )
        return boundary, global_disagreement, scores, True
    if args.scores_path.is_file() and not args.rebuild_scores:
        raise FileExistsError(
            "A score cache already exists. Use --reuse-scores for strict reuse or "
            f"--rebuild-scores for explicit replacement: {args.scores_path}"
        )
    scored_keys, boundary, global_disagreement, scores = _score_pool(
        model, loader, device, use_amp=args.amp, eps=config.eps
    )
    if scored_keys != list(sample_keys):
        raise RuntimeError("Selector score key order diverged from the stable catalog.")
    payload = dict(expected_spec)
    payload.update(
        {
            "sample_keys": list(sample_keys),
            "boundary_disagreement": boundary,
            "global_disagreement": global_disagreement,
            "scores": scores,
        }
    )
    atomic_torch_save(
        payload,
        args.scores_path,
        refuse_mismatch=not args.rebuild_scores,
    )
    return boundary, global_disagreement, scores, False


def _canonicalize_split_values(
    values: Iterable[Any], sample_keys: Sequence[str]
) -> list[str]:
    catalog = set(sample_keys)
    basename_map: dict[str, list[str]] = {}
    for key in sample_keys:
        basename_map.setdefault(key.rsplit("/", 1)[-1], []).append(key)
    normalized: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise TypeError("PC-BACS split entries must be strings, not integer indices.")
        value = raw.strip().replace("\\", "/")
        if not value:
            continue
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe split entry: {raw!r}")
        if value in catalog:
            normalized.append(value)
            continue
        value_without_extension = Path(value).with_suffix("").as_posix()
        if value_without_extension in catalog:
            normalized.append(value_without_extension)
            continue
        basename = Path(value).stem
        matches = basename_map.get(basename, [])
        if len(matches) != 1:
            raise ValueError(
                f"Cannot uniquely resolve legacy split entry {raw!r}; matches={matches}"
            )
        normalized.append(matches[0])
    if len(set(normalized)) != len(normalized):
        raise ValueError("Converted split contains duplicate sample keys.")
    unknown = sorted(set(normalized) - catalog)
    if unknown:
        raise ValueError(f"Converted split contains unknown keys: {unknown[:5]}")
    return sorted(normalized)


def _load_split_source(path: Path) -> list[Any]:
    if path.suffix.lower() == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, (list, tuple)):
            raise TypeError("A split .pt must contain a list/tuple of keys.")
        return list(payload)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _load_validated_seed_split(
    path: Path,
    *,
    sample_keys: Sequence[str],
    expected_count: int,
    label: str,
) -> list[str]:
    keys = _canonicalize_split_values(_load_split_source(path), sample_keys)
    if len(keys) != expected_count:
        raise ValueError(
            f"{label} must contain exactly {expected_count} stable keys, got {len(keys)}."
        )
    return keys


def _require_kmeans_center_seed(
    seed_keys: Sequence[str], center_seed_keys: Sequence[str], *, label: str
) -> None:
    actual = set(seed_keys)
    expected = set(center_seed_keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} is not the deterministic KMeans-center seed; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )


def _save_csv(
    path: Path,
    *,
    sample_keys: Sequence[str],
    cluster_ids: Sequence[int],
    center_distances: Sequence[float],
    boundary: torch.Tensor,
    global_disagreement: torch.Tensor,
    scores: torch.Tensor,
    selection_result: Any,
    target_counts: Sequence[int],
) -> None:
    fields = [
        "sample_key",
        "cluster_id",
        "boundary_disagreement",
        "global_disagreement",
        "pc_bacs_score",
        "dino_center_distance",
    ]
    for count in target_counts:
        fields.extend((f"selected_{count:04d}", f"selection_rank_{count:04d}"))
    fields.append("dedup_skipped_count")
    rows: list[dict[str, Any]] = []
    split_sets = {
        int(count): set(selection_result.splits[int(count)]) for count in target_counts
    }
    for index, key in enumerate(sample_keys):
        row: dict[str, Any] = {
            "sample_key": key,
            "cluster_id": int(cluster_ids[index]),
            "boundary_disagreement": f"{float(boundary[index]):.10f}",
            "global_disagreement": f"{float(global_disagreement[index]):.10f}",
            "pc_bacs_score": f"{float(scores[index]):.10f}",
            "dino_center_distance": f"{float(center_distances[index]):.10f}",
            "dedup_skipped_count": int(
                selection_result.dedup_skipped_count.get(key, 0)
            ),
        }
        for count in target_counts:
            row[f"selected_{count:04d}"] = int(key in split_sets[int(count)])
            row[f"selection_rank_{count:04d}"] = (
                selection_result.selection_rank.get(int(count), {}).get(key, "")
            )
        rows.append(row)
    rows.sort(key=lambda row: str(row["sample_key"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if path.exists() and path.read_bytes() != temporary.read_bytes():
        temporary.unlink()
        raise FileExistsError(
            f"Refusing to overwrite a different formal CSV artifact: {path}"
        )
    os.replace(temporary, path)


def _score_quantiles(scores: torch.Tensor) -> dict[str, float]:
    values = scores.float().cpu().numpy()
    names = ("min", "p10", "p25", "median", "p75", "p90", "max")
    quantiles = np.quantile(values, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    return {name: float(value) for name, value in zip(names, quantiles)}


def _environment_manifest(device: torch.device) -> dict[str, Any]:
    cuda_name = None
    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cuda_name": cuda_name,
    }


def _print_stage(index: int, title: str) -> float:
    print(f"[{index}/6] {title}", flush=True)
    return time.perf_counter()


def _print_elapsed(started: float) -> None:
    print(f"      completed in {time.perf_counter() - started:.2f}s", flush=True)


def _dry_run_report(
    *,
    args: argparse.Namespace,
    sample_count: int,
    catalog_fingerprint: str,
) -> None:
    print("PC-BACS dry run (no files will be written)")
    print(f"  samples: {sample_count}")
    print(f"  catalog fingerprint: {catalog_fingerprint}")
    print(f"  feature cache: {args.features_path}")
    print(f"  feature cache exists: {args.features_path.is_file()}")
    print(f"  seed split: {args.seed_split}")
    print(f"  selector checkpoint: {args.selector_checkpoint}")
    print(f"  output directory: {args.output_dir}")
    print(f"  target counts: {args.target_counts}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    config = PCBACSConfig(
        n_clusters=int(args.n_clusters),
        target_counts=tuple(int(count) for count in args.target_counts),
        random_seed=int(args.seed),
        feature_batch_size=int(args.feature_batch_size),
        score_batch_size=int(args.score_batch_size),
        num_workers=int(args.num_workers),
        use_amp=bool(args.amp),
        dedup_threshold=float(args.dedup_threshold),
        selector_seed_count=int(args.n_clusters),
    )
    _validate_args(args, config)
    _set_seed(config.random_seed)
    device = torch.device(args.device)
    # BaseModel currently resolves its local DINO checkout and weight relative to
    # the repository root. All user-provided paths were made absolute above.
    os.chdir(REPO_ROOT)

    started = _print_stage(1, "Build stable RGB catalog")
    full_dataset = SelectionPoolDataset(_image_roots(args), image_size=config.input_size)
    full_sample_keys = [str(item["key"]) for item in full_dataset.items]
    dataset, items, sample_keys = _active_pool(
        full_dataset, args.max_samples, config.random_seed
    )
    if args.convert_split_only:
        pass
    elif args.build_seed_only:
        if len(sample_keys) < config.n_clusters:
            raise ValueError(
                f"KMeans requires at least {config.n_clusters} samples, got "
                f"{len(sample_keys)}."
            )
    else:
        config.validate(len(sample_keys))
    catalog_fingerprint = _catalog_fingerprint(items)
    print(f"      samples={len(sample_keys)} catalog={catalog_fingerprint[:12]}")
    _print_elapsed(started)

    if args.convert_split_only:
        converted = _canonicalize_split_values(
            _load_split_source(args.seed_split), sample_keys
        )
        output = args.output_dir / f"kmeans_{len(converted):04d}_seed_keys.pt"
        if args.dry_run:
            print(f"Would write {len(converted)} canonical keys to {output}")
            return 0
        fingerprint = save_split_keys(output, converted)
        print(f"Saved {output} fingerprint={fingerprint}")
        return 0

    dino_fingerprint = _sha256_file(DINO_WEIGHT_PATH)
    if args.dry_run:
        _dry_run_report(
            args=args,
            sample_count=len(sample_keys),
            catalog_fingerprint=catalog_fingerprint,
        )
        expected_feature_spec = _feature_spec(
            catalog_fingerprint=catalog_fingerprint,
            dino_fingerprint=dino_fingerprint,
            use_amp=args.amp,
            device_type=args.device,
        )
        dry_features: torch.Tensor | None = None
        if args.features_path.suffix.lower() == ".npy":
            raise ValueError(
                "Legacy .npy features have no stable key sidecar and cannot be reused."
            )
        if args.features_path.is_file() and not args.rebuild_features:
            feature_payload = torch.load(
                args.features_path, map_location="cpu", weights_only=False
            )
            if not isinstance(feature_payload, Mapping):
                raise ValueError("Feature cache must contain a mapping payload.")
            dry_features = _validate_feature_cache(
                feature_payload,
                sample_keys=sample_keys,
                expected_spec=expected_feature_spec,
            )
            print("  feature cache validation: passed")
        elif args.rebuild_features:
            print("  feature cache validation: explicit rebuild requested")
        else:
            print("  feature cache validation: cache absent; extraction would be required")

        dry_kmeans = None
        if dry_features is not None:
            dry_kmeans = fit_dino_kmeans(
                sample_keys,
                normalize_dino_features(dry_features),
                n_clusters=config.n_clusters,
                random_seed=config.random_seed,
            )
            print(
                f"  KMeans validation: passed ({len(dry_kmeans.seed_keys)} centers)"
            )

        if not args.build_seed_only:
            selection_seed_keys = _load_validated_seed_split(
                args.seed_split,
                sample_keys=sample_keys,
                expected_count=config.selector_seed_count,
                label="Selection seed split",
            )
            selector_seed_keys = _load_validated_seed_split(
                args.selector_seed_split,
                sample_keys=full_sample_keys,
                expected_count=SELECTOR_TRAINING_SEED_COUNT,
                label="Selector training seed split",
            )
            legacy_state, metadata, non_pc_fingerprint, selector_fingerprint = (
                _load_selector_state(
                    args.selector_checkpoint,
                    selector_seed_keys=selector_seed_keys,
                    dino_fingerprint=dino_fingerprint,
                )
            )
            dry_selector = BaseModel(pc_cfg=None)
            dry_selector.decoder.load_state_dict(legacy_state, strict=True)
            del dry_selector
            if dry_kmeans is not None:
                _require_kmeans_center_seed(
                    selection_seed_keys,
                    dry_kmeans.seed_keys,
                    label="Selection seed split",
                )
                if args.max_samples is None:
                    _require_kmeans_center_seed(
                        selector_seed_keys,
                        dry_kmeans.seed_keys,
                        label="Selector training seed split",
                    )
            expected_score_spec = _score_spec(
                catalog_fingerprint=catalog_fingerprint,
                selector_fingerprint=selector_fingerprint,
                dino_fingerprint=dino_fingerprint,
                args=args,
                config=config,
            )
            score_path = _resolve_score_cache_path(args, expected_score_spec)
            if args.reuse_scores:
                if not score_path.is_file():
                    raise FileNotFoundError(
                        f"--reuse-scores requested but cache is missing: {score_path}"
                    )
                score_payload = torch.load(
                    score_path, map_location="cpu", weights_only=False
                )
                if not isinstance(score_payload, Mapping):
                    raise ValueError("Score cache must contain a mapping payload.")
                _validate_score_cache(
                    score_payload,
                    sample_keys=sample_keys,
                    expected_spec=expected_score_spec,
                )
                print("  score cache validation: passed")
            else:
                print(f"  score cache: {score_path} (full scoring would run)")
            print(
                "  non-PC selector fingerprint: "
                f"{non_pc_fingerprint} (strict load and metadata passed; "
                f"epoch={SELECTOR_EPOCHS}, design={metadata['training_design']})"
            )
        return 0

    loader = _make_loader(
        dataset,
        batch_size=config.feature_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    model = BaseModel(pc_cfg=None).to(device).eval()

    started = _print_stage(2, "Load or extract keyed DINO features")
    features, feature_cache_hit = _load_or_extract_features(
        args=args,
        model=model,
        loader=loader,
        device=device,
        sample_keys=sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    normalized_features = normalize_dino_features(features)
    print(f"      cache_hit={feature_cache_hit} shape={tuple(features.shape)}")
    _print_elapsed(started)

    started = _print_stage(3, "Fit deterministic DINO KMeans")
    kmeans_result = fit_dino_kmeans(
        sample_keys,
        normalized_features,
        n_clusters=config.n_clusters,
        random_seed=config.random_seed,
    )
    _print_elapsed(started)

    if args.build_seed_only:
        seed_output = args.output_dir / f"kmeans_{config.n_clusters:04d}_seed_keys.pt"
        seed_fingerprint = save_split_keys(seed_output, kmeans_result.seed_keys)
        seed_manifest = {
            "format_version": 1,
            "method": "DINO-KMeans-center PC-BACS seed",
            "sample_count": len(sample_keys),
            "seed_count": len(kmeans_result.seed_keys),
            "n_clusters": config.n_clusters,
            "random_seed": config.random_seed,
            "catalog_fingerprint": catalog_fingerprint,
            "dino_fingerprint": dino_fingerprint,
            "feature_spec_fingerprint": _feature_spec(
                catalog_fingerprint=catalog_fingerprint,
                dino_fingerprint=dino_fingerprint,
                use_amp=args.amp,
                device_type=args.device,
            )["feature_spec_fingerprint"],
            "seed_fingerprint": seed_fingerprint,
            "seed_path": seed_output.name,
            "smoke_test": args.max_samples is not None,
        }
        atomic_json_save(
            seed_manifest,
            args.output_dir / f"kmeans_{config.n_clusters:04d}_seed_manifest.json",
            refuse_mismatch=not args.rebuild_features,
        )
        print(f"Saved {seed_output} ({len(kmeans_result.seed_keys)} keys)")
        return 0

    selection_seed_keys = _load_validated_seed_split(
        args.seed_split,
        sample_keys=sample_keys,
        expected_count=config.selector_seed_count,
        label="Selection seed split",
    )
    selector_seed_keys = _load_validated_seed_split(
        args.selector_seed_split,
        sample_keys=full_sample_keys,
        expected_count=SELECTOR_TRAINING_SEED_COUNT,
        label="Selector training seed split",
    )
    _require_kmeans_center_seed(
        selection_seed_keys,
        kmeans_result.seed_keys,
        label="Selection seed split",
    )
    if args.max_samples is None:
        _require_kmeans_center_seed(
            selector_seed_keys,
            kmeans_result.seed_keys,
            label="Selector training seed split",
        )

    started = _print_stage(4, "Strict-load selector and score RGB pool")
    legacy_state, selector_metadata, non_pc_fingerprint, selector_fingerprint = (
        _load_selector_state(
            args.selector_checkpoint,
            selector_seed_keys=selector_seed_keys,
            dino_fingerprint=dino_fingerprint,
        )
    )
    model.decoder.load_state_dict(legacy_state, strict=True)
    score_loader = _make_loader(
        dataset,
        batch_size=config.score_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    expected_score_spec = _score_spec(
        catalog_fingerprint=catalog_fingerprint,
        selector_fingerprint=selector_fingerprint,
        dino_fingerprint=dino_fingerprint,
        args=args,
        config=config,
    )
    boundary, global_disagreement, scores, score_cache_hit = _load_or_score_pool(
        args=args,
        model=model,
        loader=score_loader,
        device=device,
        sample_keys=sample_keys,
        expected_spec=expected_score_spec,
        config=config,
    )
    quantiles = _score_quantiles(scores)
    print(f"      selector={selector_fingerprint[:12]} cache_hit={score_cache_hit}")
    print("      score quantiles=" + json.dumps(quantiles, sort_keys=True))
    _print_elapsed(started)

    started = _print_stage(5, "Allocate cluster budgets and build nested splits")
    selection_result = build_nested_splits(
        sample_keys,
        normalized_features,
        kmeans_result.cluster_ids,
        scores,
        selection_seed_keys,
        target_counts=config.target_counts,
        dedup_threshold=config.dedup_threshold,
    )
    for smaller, larger in zip(config.target_counts, config.target_counts[1:]):
        if not set(selection_result.splits[int(smaller)]) <= set(
            selection_result.splits[int(larger)]
        ):
            raise RuntimeError(f"Nested split invariant failed: {smaller} !<= {larger}")
    _print_elapsed(started)

    started = _print_stage(6, "Validate and atomically save artifacts")
    split_manifest: dict[str, Any] = {}
    for count in config.target_counts:
        keys = selection_result.splits[int(count)]
        if len(keys) != int(count):
            raise RuntimeError(
                f"Split {count} has {len(keys)} keys; refusing partial output."
            )
        output = args.output_dir / f"pc_bacs_{int(count):04d}_keys.pt"
        fingerprint = save_split_keys(output, keys)
        split_manifest[str(int(count))] = {
            "path": output.name,
            "count": len(keys),
            "fingerprint": fingerprint,
        }
    csv_path = args.output_dir / "pc_bacs_scores.csv"
    _save_csv(
        csv_path,
        sample_keys=sample_keys,
        cluster_ids=kmeans_result.cluster_ids,
        center_distances=kmeans_result.center_distances,
        boundary=boundary,
        global_disagreement=global_disagreement,
        scores=scores,
        selection_result=selection_result,
        target_counts=config.target_counts,
    )
    cluster_sizes = Counter(int(value) for value in kmeans_result.cluster_ids)
    manifest = {
        "format_version": 1,
        "method": "PC-BACS",
        "repo_commit": _repo_commit(),
        "implementation_fingerprint": _implementation_fingerprint(),
        "dataset": {
            "root": args.data_root.name,
            "train_sets": list(args.train_sets),
            "sample_count": len(sample_keys),
            "catalog_fingerprint": catalog_fingerprint,
        },
        "selector": {
            "checkpoint": args.selector_checkpoint.name,
            "selector_fingerprint": selector_fingerprint,
            "non_pc_decoder_fingerprint": non_pc_fingerprint,
            "training_seed_split": args.selector_seed_split.name,
            "training_seed_fingerprint": compute_labeled_split_fingerprint(
                selector_seed_keys
            ),
            "training_seed_count": len(selector_seed_keys),
            "training_design": selector_metadata["training_design"],
            "artifact_role": selector_metadata["artifact_role"],
            "pc_frozen": selector_metadata["pc_frozen"],
            "epochs": SELECTOR_EPOCHS,
            "forward_mode": "off",
        },
        "selection": {
            "seed_split": args.seed_split.name,
            "seed_fingerprint": compute_labeled_split_fingerprint(
                selection_seed_keys
            ),
            "clusters": config.n_clusters,
            "target_counts": list(config.target_counts),
            "dedup_threshold": config.dedup_threshold,
            "random_seed": config.random_seed,
            "score_formula": "D_bd * (1 - D_all)",
            "score_formula_version": config.score_formula_version,
            "cluster_sizes": {str(k): v for k, v in sorted(cluster_sizes.items())},
            "rounds": selection_result.rounds,
            "feature_batch_size": config.feature_batch_size,
            "score_batch_size": config.score_batch_size,
            "num_workers": config.num_workers,
            "amp": bool(args.amp and device.type == "cuda"),
            "deterministic_algorithms": True,
            "max_samples": args.max_samples,
        },
        "cache": {
            "feature_path": args.features_path.name,
            "score_path": args.scores_path.name,
            "feature_validated": True,
            "score_validated": True,
            "feature_spec_fingerprint": _feature_spec(
                catalog_fingerprint=catalog_fingerprint,
                dino_fingerprint=dino_fingerprint,
                use_amp=args.amp,
                device_type=args.device,
            )["feature_spec_fingerprint"],
            "score_spec_fingerprint": expected_score_spec[
                "score_spec_fingerprint"
            ],
        },
        "model": {
            "dino_weight": DINO_WEIGHT_PATH.relative_to(REPO_ROOT).as_posix(),
            "dino_fingerprint": dino_fingerprint,
        },
        "score_quantiles": quantiles,
        "outputs": split_manifest,
        "scores_csv": csv_path.name,
        "environment": _environment_manifest(device),
        "smoke_test": args.max_samples is not None,
    }
    atomic_json_save(manifest, args.output_dir / "pc_bacs_manifest.json")
    _print_elapsed(started)
    print(
        "PC-BACS completed: "
        + ", ".join(
            f"{count}={split_manifest[str(int(count))]['fingerprint'][:12]}"
            for count in config.target_counts
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
