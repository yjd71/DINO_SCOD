from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from configs.pc_bacs_config import SCORE_FORMULA_VERSION as SOURCE_SCORE_FORMULA_VERSION
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    extract_non_pc_decoder_state,
    normalize_sample_key,
    read_artifact_metadata,
    state_dict_fingerprint,
)
from utils.dataloader import SelectionPoolDataset
from utils.global_additive import (
    GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION,
    GLOBAL_ADDITIVE_FORMULA_VERSION,
    build_global_deduplicated_nested_splits,
    build_global_nested_splits,
    compute_global_additive_score,
    labeled_names_from_keys,
)


REPO_ROOT = Path(__file__).resolve().parent
DINO_WEIGHT_PATH = REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth"
SOURCE_SCORE_CACHE_VERSION = 1
DERIVED_SCORE_CACHE_VERSION = 1
MANIFEST_VERSION = 1
SELECTOR_EPOCHS = 5
FORMAL_TARGET_COUNTS = (41, 202, 404, 808)
PREPROCESS_SPEC = {
    "color": "opencv_bgr_to_rgb_to_pil",
    "input_size": 392,
    "resize": "bilinear",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
DINO_FEATURE_STATIC_SPEC = {
    "format_version": 1,
    "feature_type": "dinov2_vitb14_global",
    "feature_definition": "model.dino(normalized_392_rgb)",
    "feature_dim": 768,
}
DEDUP_COMPARISON_RULE = "cosine_similarity > threshold"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact nested labeled splits by the global additive score "
            "D_bd + (1 - D_all), without KMeans."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument(
        "--train-sets", nargs="+", default=["TR-CAMO", "TR-COD10K"]
    )
    parser.add_argument("--seed-split", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--source-score-cache", type=Path, required=True)
    parser.add_argument(
        "--dedup-mode",
        choices=("none", "dino-cosine"),
        default="none",
        help="Optional global DINO cosine deduplication; never runs KMeans.",
    )
    parser.add_argument(
        "--dino-feature-cache",
        type=Path,
        help="Existing keyed [N,768] DINO cache required by dino-cosine mode.",
    )
    parser.add_argument("--dedup-threshold", type=float, default=0.95)
    parser.add_argument(
        "--target-counts",
        nargs="+",
        type=int,
        default=list(FORMAL_TARGET_COUNTS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.seed_split = args.seed_split.resolve()
    args.selector_checkpoint = args.selector_checkpoint.resolve()
    args.source_score_cache = args.source_score_cache.resolve()
    if args.dino_feature_cache is not None:
        args.dino_feature_cache = args.dino_feature_cache.resolve()
    args.output_dir = args.output_dir.resolve()


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    for label, path in (
        ("data root", args.data_root),
        ("seed split", args.seed_split),
        ("selector checkpoint", args.selector_checkpoint),
        ("source score cache", args.source_score_cache),
    ):
        if label == "data root":
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {label}: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not DINO_WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"Missing repository DINO weight: {DINO_WEIGHT_PATH}")
    if not args.train_sets or len(set(args.train_sets)) != len(args.train_sets):
        raise ValueError("--train-sets must contain unique dataset names.")
    targets = tuple(args.target_counts)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in targets):
        raise TypeError("--target-counts must contain integers.")
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("--target-counts must contain positive integers.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("--target-counts must be strictly increasing.")
    if not math.isfinite(args.dedup_threshold) or not (
        0.0 < args.dedup_threshold <= 1.0
    ):
        raise ValueError("--dedup-threshold must be finite and in (0, 1].")
    if args.dedup_mode == "dino-cosine":
        if args.dino_feature_cache is None:
            raise ValueError(
                "--dedup-mode dino-cosine requires --dino-feature-cache."
            )
        if not args.dino_feature_cache.is_file():
            raise FileNotFoundError(
                f"Missing DINO feature cache: {args.dino_feature_cache}"
            )
    elif args.dino_feature_cache is not None:
        raise ValueError(
            "--dino-feature-cache is only valid with --dedup-mode dino-cosine."
        )
    return targets


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        key = normalize_sample_key(str(item["key"]))
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
        REPO_ROOT / "select_global_additive.py",
        REPO_ROOT / "utils" / "global_additive.py",
        REPO_ROOT / "utils" / "dataloader.py",
        REPO_ROOT / "utils" / "checkpoint_pc_hbm.py",
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


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_split_source(path: Path) -> list[Any]:
    if path.suffix.lower() == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, (list, tuple)):
            raise TypeError("A seed .pt must contain a list/tuple of sample keys.")
        return list(payload)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


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
            raise TypeError("Global-additive seed entries must be strings.")
        value = raw.strip().replace("\\", "/")
        if not value:
            continue
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe seed entry: {raw!r}")
        if value in catalog:
            normalized.append(value)
            continue
        without_extension = Path(value).with_suffix("").as_posix()
        if without_extension in catalog:
            normalized.append(without_extension)
            continue
        basename = Path(value).stem
        matches = basename_map.get(basename, [])
        if len(matches) != 1:
            raise ValueError(
                f"Cannot uniquely resolve seed entry {raw!r}; matches={matches}"
            )
        normalized.append(matches[0])
    if len(normalized) != len(set(normalized)):
        raise ValueError("Seed split contains duplicate sample keys.")
    unknown = sorted(set(normalized) - catalog)
    if unknown:
        raise ValueError(f"Seed split contains unknown keys: {unknown[:5]}")
    if not normalized:
        raise ValueError("Seed split must not be empty.")
    return sorted(normalized)


def _load_selector_identity(
    checkpoint_path: Path,
    *,
    selector_seed_keys: Sequence[str],
    dino_fingerprint: str,
) -> tuple[dict[str, Any], str, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Selector checkpoint must contain a mapping payload.")
    if int(payload.get("epoch", -1)) != SELECTOR_EPOCHS:
        raise ValueError(
            "Selector checkpoint must be the completed epoch-5 artifact, "
            f"got epoch={payload.get('epoch')!r}."
        )
    try:
        metadata = read_artifact_metadata(payload)
    except RuntimeError as error:
        # The formal split0.01 selector predates the architecture/schema fields
        # now required by the repository-wide checkpoint reader.  Accept only
        # its explicit metadata-v1 envelope and still enforce every selector
        # identity field used by this protocol below.
        legacy_metadata = payload.get("artifact_meta")
        if not isinstance(legacy_metadata, Mapping) or int(
            legacy_metadata.get("artifact_metadata_version", -1)
        ) != 1:
            raise error
        metadata = dict(legacy_metadata)
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
        f"{key}: expected {expected!r}, got {metadata.get(key)!r}"
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
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
    return dict(metadata), non_pc_fingerprint, selector_fingerprint


def _expected_source_spec(
    *,
    catalog_fingerprint: str,
    selector_fingerprint: str,
    dino_fingerprint: str,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "format_version": SOURCE_SCORE_CACHE_VERSION,
        "catalog_fingerprint": catalog_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
        "score_formula_version": SOURCE_SCORE_FORMULA_VERSION,
        "output_index": 3,
        "output_size": 98,
        "transform": "horizontal_flip",
        "sobel_padding": "replicate",
        "sobel_magnitude": "torch.hypot",
        "eps_location": "boundary_denominator_only",
        "eps": 1e-6,
        "amp": True,
        "device_type": "cuda",
    }
    spec["score_spec_fingerprint"] = _sha256_json(spec)
    return spec


def _validate_source_score_cache(
    payload: Mapping[str, Any],
    *,
    sample_keys: Sequence[str],
    expected_spec: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mismatches: list[str] = []
    for key, expected in expected_spec.items():
        if payload.get(key) != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, got {payload.get(key)!r}"
            )
    if list(payload.get("sample_keys") or []) != list(sample_keys):
        mismatches.append("sample_keys differ from the active stable catalog")

    tensors: dict[str, torch.Tensor] = {}
    for name in ("boundary_disagreement", "global_disagreement", "scores"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            mismatches.append(f"{name} is not a tensor")
            continue
        if value.dtype != torch.float32:
            mismatches.append(
                f"{name} dtype must be torch.float32, got {value.dtype}"
            )
            continue
        value = value.detach().cpu().contiguous()
        if tuple(value.shape) != (len(sample_keys),):
            mismatches.append(
                f"{name} shape must be {(len(sample_keys),)}, got {tuple(value.shape)}"
            )
        elif not torch.isfinite(value).all():
            mismatches.append(f"{name} contains NaN or Inf")
        elif bool(((value < -1e-6) | (value > 1.0 + 1e-6)).any()):
            mismatches.append(f"{name} contains values outside [0, 1]")
        tensors[name] = value

    if set(tensors) == {"boundary_disagreement", "global_disagreement", "scores"}:
        boundary = tensors["boundary_disagreement"]
        global_disagreement = tensors["global_disagreement"]
        scores = tensors["scores"]
        if all(value.shape == boundary.shape for value in tensors.values()):
            expected_scores = boundary * (1.0 - global_disagreement)
            if not torch.allclose(scores, expected_scores, rtol=1e-6, atol=1e-7):
                mismatches.append(
                    "source scores do not satisfy D_bd * (1 - D_all)"
                )
    if mismatches:
        raise ValueError("Source score cache mismatch:\n- " + "\n- ".join(mismatches))
    return tensors["boundary_disagreement"], tensors["global_disagreement"]


def _expected_dino_feature_static_spec(
    *, catalog_fingerprint: str, dino_fingerprint: str
) -> dict[str, Any]:
    return {
        **DINO_FEATURE_STATIC_SPEC,
        "catalog_fingerprint": catalog_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
    }


def _sample_key_order_fingerprint(sample_keys: Sequence[str]) -> str:
    return _sha256_json({"sample_keys": list(sample_keys)})


def _validate_dino_feature_cache(
    payload: Mapping[str, Any],
    *,
    sample_keys: Sequence[str],
    expected_static_spec: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Strictly validate an existing keyed DINO cache without model execution."""

    mismatches: list[str] = []
    for key, expected in expected_static_spec.items():
        if payload.get(key) != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, got {payload.get(key)!r}"
            )

    feature_amp = payload.get("feature_amp")
    if not isinstance(feature_amp, bool):
        mismatches.append("feature_amp must be boolean")
    feature_device_type = payload.get("feature_device_type")
    if feature_device_type not in ("cpu", "cuda"):
        mismatches.append("feature_device_type must be 'cpu' or 'cuda'")

    dynamic_spec = dict(expected_static_spec)
    dynamic_spec.update(
        {
            "feature_amp": feature_amp,
            "feature_device_type": feature_device_type,
        }
    )
    expected_spec_fingerprint = _sha256_json(dynamic_spec)
    if payload.get("feature_spec_fingerprint") != expected_spec_fingerprint:
        mismatches.append(
            "feature_spec_fingerprint does not match the cache specification"
        )

    if list(payload.get("sample_keys") or []) != list(sample_keys):
        mismatches.append("sample_keys differ from the active stable catalog order")
    normalized = payload.get("normalized")
    if not isinstance(normalized, bool):
        mismatches.append("normalized must be boolean")

    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        mismatches.append("features is not a tensor")
    else:
        if features.dtype != torch.float32:
            mismatches.append(
                f"features dtype must be torch.float32, got {features.dtype}"
            )
        if tuple(features.shape) != (len(sample_keys), 768):
            mismatches.append(
                "features shape must be "
                f"{(len(sample_keys), 768)}, got {tuple(features.shape)}"
            )
        elif not torch.isfinite(features).all():
            mismatches.append("features contains NaN or Inf")
        elif bool((torch.linalg.vector_norm(features, dim=1) <= 0.0).any()):
            mismatches.append("features contains a zero-norm row")

    if mismatches:
        raise ValueError(
            "DINO feature cache mismatch:\n- " + "\n- ".join(mismatches)
        )
    assert isinstance(features, torch.Tensor)
    return features.detach().cpu().contiguous(), {
        **dynamic_spec,
        "feature_spec_fingerprint": expected_spec_fingerprint,
        "key_order_fingerprint": _sample_key_order_fingerprint(sample_keys),
        "normalized": normalized,
    }


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _payload_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _encode_csv(
    *,
    sample_keys: Sequence[str],
    boundary: torch.Tensor,
    global_disagreement: torch.Tensor,
    scores: torch.Tensor,
    seed_keys: set[str],
    selection_result: Any,
    target_counts: Sequence[int],
    dedup_mode: str = "none",
) -> bytes:
    fieldnames = [
        "sample_key",
        "labeled_name",
        "boundary_disagreement",
        "global_disagreement",
        "global_additive_score",
        "is_seed",
        "global_candidate_rank",
    ]
    if dedup_mode == "dino-cosine":
        fieldnames.extend(
            (
                "dedup_decision",
                "dedup_max_cosine_similarity",
                "dedup_reference_key",
                "dedup_relaxed",
                "dedup_evaluated_target_count",
                "dedup_selected_target_count",
            )
        )
    for target in target_counts:
        fieldnames.extend((f"selected_{target:04d}", f"selection_rank_{target:04d}"))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for index, key in enumerate(sample_keys):
        row: dict[str, Any] = {
            "sample_key": key,
            "labeled_name": key.rsplit("/", 1)[-1],
            "boundary_disagreement": f"{float(boundary[index]):.10f}",
            "global_disagreement": f"{float(global_disagreement[index]):.10f}",
            "global_additive_score": f"{float(scores[index]):.10f}",
            "is_seed": int(key in seed_keys),
            "global_candidate_rank": selection_result.global_rank.get(key, ""),
        }
        if dedup_mode == "dino-cosine":
            audit = selection_result.audit[key]
            row.update(
                {
                    "dedup_decision": audit.decision,
                    "dedup_max_cosine_similarity": (
                        ""
                        if audit.max_cosine_similarity is None
                        else f"{audit.max_cosine_similarity:.10f}"
                    ),
                    "dedup_reference_key": audit.reference_key or "",
                    "dedup_relaxed": int(audit.relaxed),
                    "dedup_evaluated_target_count": (
                        ""
                        if audit.evaluated_target_count is None
                        else audit.evaluated_target_count
                    ),
                    "dedup_selected_target_count": (
                        ""
                        if audit.selected_target_count is None
                        else audit.selected_target_count
                    ),
                }
            )
        for target in target_counts:
            selected = key in selection_result.selection_rank[int(target)]
            row[f"selected_{target:04d}"] = int(selected)
            row[f"selection_rank_{target:04d}"] = (
                selection_result.selection_rank[int(target)].get(key, "")
            )
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _stage_and_publish(
    torch_payloads: Mapping[Path, Any], text_payloads: Mapping[Path, bytes]
) -> None:
    for path, expected in torch_payloads.items():
        if path.is_file():
            actual = torch.load(path, map_location="cpu", weights_only=False)
            if not _payload_equal(actual, expected):
                raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
    for path, expected in text_payloads.items():
        if path.is_file() and path.read_bytes() != expected:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")

    staged: list[tuple[Path, Path]] = []
    try:
        for path, payload in torch_payloads.items():
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
            )
            temporary = Path(handle.name)
            handle.close()
            torch.save(payload, temporary)
            staged.append((temporary, path))
        for path, payload in text_payloads.items():
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
            )
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            staged.append((temporary, path))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _score_quantiles(scores: torch.Tensor) -> dict[str, float]:
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    values = torch.quantile(scores.float().cpu(), quantiles)
    return {
        label: float(value)
        for label, value in zip(("min", "q25", "median", "q75", "max"), values)
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_args(args)
    targets = _validate_args(args)

    dataset = SelectionPoolDataset(
        [str(args.data_root / name / "im") for name in args.train_sets],
        image_size=392,
    )
    items = [dict(item) for item in dataset.items]
    sample_keys = list(dataset.sample_keys)
    if sample_keys != sorted(sample_keys):
        raise RuntimeError("SelectionPoolDataset catalog is not in stable key order.")
    catalog_fingerprint = _catalog_fingerprint(items)
    seed_keys = _canonicalize_split_values(
        _load_split_source(args.seed_split), sample_keys
    )
    if len(seed_keys) > targets[0]:
        raise ValueError("Seed count exceeds the smallest target count.")
    if targets[-1] > len(sample_keys):
        raise ValueError("Largest target exceeds the active catalog size.")

    dino_fingerprint = _sha256_file(DINO_WEIGHT_PATH)
    selector_metadata, non_pc_fingerprint, selector_fingerprint = (
        _load_selector_identity(
            args.selector_checkpoint,
            selector_seed_keys=seed_keys,
            dino_fingerprint=dino_fingerprint,
        )
    )
    expected_source_spec = _expected_source_spec(
        catalog_fingerprint=catalog_fingerprint,
        selector_fingerprint=selector_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    source_payload = torch.load(
        args.source_score_cache, map_location="cpu", weights_only=False
    )
    if not isinstance(source_payload, Mapping):
        raise TypeError("Source score cache must contain a mapping payload.")
    boundary, global_disagreement = _validate_source_score_cache(
        source_payload,
        sample_keys=sample_keys,
        expected_spec=expected_source_spec,
    )
    scores = compute_global_additive_score(boundary, global_disagreement)
    dedup_enabled = args.dedup_mode == "dino-cosine"
    feature_cache_sha256: str | None = None
    feature_cache_metadata: dict[str, Any] | None = None
    if dedup_enabled:
        feature_payload = torch.load(
            args.dino_feature_cache, map_location="cpu", weights_only=False
        )
        if not isinstance(feature_payload, Mapping):
            raise TypeError("DINO feature cache must contain a mapping payload.")
        features, feature_cache_metadata = _validate_dino_feature_cache(
            feature_payload,
            sample_keys=sample_keys,
            expected_static_spec=_expected_dino_feature_static_spec(
                catalog_fingerprint=catalog_fingerprint,
                dino_fingerprint=dino_fingerprint,
            ),
        )
        feature_cache_sha256 = _sha256_file(args.dino_feature_cache)
        selection_result = build_global_deduplicated_nested_splits(
            sample_keys,
            scores,
            seed_keys,
            features,
            target_counts=targets,
            dedup_threshold=args.dedup_threshold,
        )
    else:
        selection_result = build_global_nested_splits(
            sample_keys,
            scores,
            seed_keys,
            target_counts=targets,
        )

    split_fingerprints = {
        int(target): compute_labeled_split_fingerprint(
            selection_result.splits[int(target)]
        )
        for target in targets
    }
    first_added = selection_result.selection_order[len(seed_keys)]
    dry_report = {
        "catalog_count": len(sample_keys),
        "catalog_fingerprint": catalog_fingerprint,
        "seed_count": len(seed_keys),
        "selector_fingerprint": selector_fingerprint,
        "source_score_cache": _repo_relative(args.source_score_cache),
        "source_score_cache_sha256": _sha256_file(args.source_score_cache),
        "score_formula": "D_bd + (1 - D_all)",
        "score_formula_version": GLOBAL_ADDITIVE_FORMULA_VERSION,
        "uses_kmeans": False,
        "dedup": dedup_enabled,
        "dedup_mode": args.dedup_mode,
        "first_added": first_added,
        "targets": {str(key): value for key, value in split_fingerprints.items()},
    }
    if dedup_enabled:
        dry_report.update(
            {
                "dedup_threshold": args.dedup_threshold,
                "dedup_comparison_rule": DEDUP_COMPARISON_RULE,
                "dino_feature_cache": _repo_relative(args.dino_feature_cache),
                "dino_feature_cache_sha256": feature_cache_sha256,
                "dedup_rounds": [
                    asdict(round_record) for round_record in selection_result.rounds
                ],
            }
        )
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0

    source_cache_sha256 = dry_report["source_score_cache_sha256"]
    artifact_prefix = (
        "global_additive_dedup" if dedup_enabled else "global_additive"
    )
    derived_cache_path = (
        args.data_root / "cache" / f"{artifact_prefix}_scores_split0.01.pt"
    )
    derived_cache = {
        "format_version": DERIVED_SCORE_CACHE_VERSION,
        "catalog_fingerprint": catalog_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
        "source_score_cache": _repo_relative(args.source_score_cache),
        "source_score_cache_sha256": source_cache_sha256,
        "source_score_spec_fingerprint": expected_source_spec[
            "score_spec_fingerprint"
        ],
        "score_formula": "D_bd + (1 - D_all)",
        "score_formula_version": GLOBAL_ADDITIVE_FORMULA_VERSION,
        "score_range": [0.0, 2.0],
        "ranking_tie_break": "(-score, sample_key)",
        "uses_kmeans": False,
        "dedup": dedup_enabled,
        "sample_keys": list(sample_keys),
        "boundary_disagreement": boundary,
        "global_disagreement": global_disagreement,
        "scores": scores,
    }
    if dedup_enabled:
        assert feature_cache_metadata is not None
        derived_cache.update(
            {
                "dedup_mode": args.dedup_mode,
                "dedup_protocol_version": GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION,
                "dedup_threshold": args.dedup_threshold,
                "dedup_comparison_rule": DEDUP_COMPARISON_RULE,
                "dino_feature_cache": _repo_relative(args.dino_feature_cache),
                "dino_feature_cache_sha256": feature_cache_sha256,
                "dino_feature_spec_fingerprint": feature_cache_metadata[
                    "feature_spec_fingerprint"
                ],
                "dino_feature_key_order_fingerprint": feature_cache_metadata[
                    "key_order_fingerprint"
                ],
                # The catalog digest hashes every image byte stream and is thus
                # also the image-content identity for this offline protocol.
                "image_fingerprint": catalog_fingerprint,
                "image_fingerprint_scheme": "pc_bacs_catalog_v1_per_image_sha256",
            }
        )

    torch_payloads: dict[Path, Any] = {derived_cache_path: derived_cache}
    text_payloads: dict[Path, bytes] = {}
    outputs: dict[str, Any] = {}
    for target in targets:
        keys = selection_result.splits[int(target)]
        names = labeled_names_from_keys(keys)
        pt_path = args.output_dir / f"{artifact_prefix}_{target:04d}_keys.pt"
        txt_path = (
            args.output_dir / f"{artifact_prefix}_{target:04d}_labeled_names.txt"
        )
        torch_payloads[pt_path] = keys
        text_payloads[txt_path] = ("\n".join(names) + "\n").encode("utf-8")
        outputs[str(target)] = {
            "count": len(keys),
            "split_fingerprint": split_fingerprints[int(target)],
            "pt_path": pt_path.name,
            "txt_path": txt_path.name,
            "txt_sha256": hashlib.sha256(text_payloads[txt_path]).hexdigest(),
        }

    csv_path = args.output_dir / f"{artifact_prefix}_scores.csv"
    csv_payload = _encode_csv(
        sample_keys=sample_keys,
        boundary=boundary,
        global_disagreement=global_disagreement,
        scores=scores,
        seed_keys=set(seed_keys),
        selection_result=selection_result,
        target_counts=targets,
        dedup_mode=args.dedup_mode,
    )
    text_payloads[csv_path] = csv_payload

    selection_manifest: dict[str, Any] = {
        "strategy": "global_topk_dino_cosine_dedup" if dedup_enabled else "global_topk",
        "score_formula": "D_bd + (1 - D_all)",
        "score_formula_version": GLOBAL_ADDITIVE_FORMULA_VERSION,
        "score_range": [0.0, 2.0],
        "score_quantiles": _score_quantiles(scores),
        "ranking_tie_break": "(-score, sample_key)",
        "uses_kmeans": False,
        "dedup": dedup_enabled,
        "target_counts": list(targets),
        "first_added": first_added,
    }
    if dedup_enabled:
        decision_counts: dict[str, int] = {}
        relaxed_count = 0
        for audit in selection_result.audit.values():
            decision_counts[audit.decision] = (
                decision_counts.get(audit.decision, 0) + 1
            )
            relaxed_count += int(audit.relaxed)
        selection_manifest.update(
            {
                "dedup_mode": args.dedup_mode,
                "dedup_protocol_version": GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION,
                "dedup_threshold": args.dedup_threshold,
                "dedup_comparison_rule": DEDUP_COMPARISON_RULE,
                "dedup_scope": "all_seeds_and_previously_selected_samples",
                "dedup_rounds": [
                    asdict(round_record) for round_record in selection_result.rounds
                ],
                "dedup_audit": {
                    "csv_path": csv_path.name,
                    "decision_field": "dedup_decision",
                    "max_cosine_similarity_field": (
                        "dedup_max_cosine_similarity"
                    ),
                    "reference_key_field": "dedup_reference_key",
                    "relaxed_field": "dedup_relaxed",
                    "decision_counts": dict(sorted(decision_counts.items())),
                    "relaxed_count": relaxed_count,
                },
            }
        )

    manifest = {
        "format_version": MANIFEST_VERSION,
        "method": "Global-Additive-BACS",
        "protocol": (
            GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION
            if dedup_enabled
            else "global_topk_no_kmeans_v1"
        ),
        "repo_commit": _repo_commit(),
        "implementation_fingerprint": _implementation_fingerprint(),
        "dataset": {
            "root": _repo_relative(args.data_root),
            "train_sets": list(args.train_sets),
            "sample_count": len(sample_keys),
            "catalog_fingerprint": catalog_fingerprint,
        },
        "selector": {
            "checkpoint": _repo_relative(args.selector_checkpoint),
            "epoch": SELECTOR_EPOCHS,
            "training_design": selector_metadata["training_design"],
            "artifact_role": selector_metadata["artifact_role"],
            "pc_frozen": selector_metadata["pc_frozen"],
            "training_seed_split": _repo_relative(args.seed_split),
            "training_seed_count": len(seed_keys),
            "training_seed_fingerprint": compute_labeled_split_fingerprint(
                seed_keys
            ),
            "non_pc_decoder_fingerprint": non_pc_fingerprint,
            "selector_fingerprint": selector_fingerprint,
            "dino_weight": _repo_relative(DINO_WEIGHT_PATH),
            "dino_fingerprint": dino_fingerprint,
        },
        "source_cache": {
            "path": _repo_relative(args.source_score_cache),
            "sha256": source_cache_sha256,
            "score_spec_fingerprint": expected_source_spec[
                "score_spec_fingerprint"
            ],
            "source_formula_version": SOURCE_SCORE_FORMULA_VERSION,
            "derived_cache": _repo_relative(derived_cache_path),
        },
        "selection": selection_manifest,
        "outputs": outputs,
        "scores_csv": {
            "path": csv_path.name,
            "sha256": hashlib.sha256(csv_payload).hexdigest(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "numpy": _package_version("numpy"),
        },
    }
    if dedup_enabled:
        assert feature_cache_metadata is not None
        manifest["dino_feature_cache"] = {
            "path": _repo_relative(args.dino_feature_cache),
            "sha256": feature_cache_sha256,
            "feature_spec_fingerprint": feature_cache_metadata[
                "feature_spec_fingerprint"
            ],
            "key_order_fingerprint": feature_cache_metadata[
                "key_order_fingerprint"
            ],
            "catalog_fingerprint": catalog_fingerprint,
            "image_fingerprint": catalog_fingerprint,
            "image_fingerprint_scheme": "pc_bacs_catalog_v1_per_image_sha256",
            "dino_fingerprint": dino_fingerprint,
            "preprocessing_fingerprint": feature_cache_metadata[
                "preprocessing_fingerprint"
            ],
            "feature_amp": feature_cache_metadata["feature_amp"],
            "feature_device_type": feature_cache_metadata[
                "feature_device_type"
            ],
            "shape": [len(sample_keys), 768],
            "dtype": "float32",
        }
    manifest_path = args.output_dir / f"{artifact_prefix}_manifest.json"
    text_payloads[manifest_path] = _encode_json(manifest)

    _stage_and_publish(torch_payloads, text_payloads)
    print(json.dumps(dry_report, indent=2, sort_keys=True))
    print(f"Saved global-additive artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
