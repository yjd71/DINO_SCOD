from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    normalize_sample_key,
)
from utils.dataloader import SelectionPoolDataset
from utils.kmeans_only import (
    KMEANS_ONLY_PROTOCOL_VERSION,
    build_kmeans_only_nested_splits,
    fit_kmeans_only,
    labeled_names_from_keys,
)


REPO_ROOT = Path(__file__).resolve().parent
DINO_WEIGHT_PATH = REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth"
FORMAL_TARGET_COUNTS = (41, 202, 404, 808)
FORMAL_N_CLUSTERS = 40
FORMAL_RANDOM_SEED = 2025
MANIFEST_VERSION = 1
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact nested labeled splits using only deterministic DINO "
            "KMeans clusters, sqrt quotas, and center distance."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument(
        "--train-sets",
        nargs="+",
        default=["TR-CAMO", "TR-COD10K"],
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt"),
    )
    parser.add_argument("--n-clusters", type=int, default=FORMAL_N_CLUSTERS)
    parser.add_argument("--seed", type=int, default=FORMAL_RANDOM_SEED)
    parser.add_argument(
        "--target-counts",
        nargs="+",
        type=int,
        default=list(FORMAL_TARGET_COUNTS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./Dataset/COD/splits/kmeans_only"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.features_path = args.features_path.resolve()
    args.output_dir = args.output_dir.resolve()


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {args.data_root}")
    if not args.features_path.is_file():
        raise FileNotFoundError(f"Missing DINO feature cache: {args.features_path}")
    if not DINO_WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"Missing repository DINO weight: {DINO_WEIGHT_PATH}")
    if not args.train_sets or len(set(args.train_sets)) != len(args.train_sets):
        raise ValueError("--train-sets must contain unique dataset names.")
    if isinstance(args.n_clusters, bool) or args.n_clusters <= 0:
        raise ValueError("--n-clusters must be positive.")
    if isinstance(args.seed, bool):
        raise TypeError("--seed must be an integer.")
    targets = tuple(args.target_counts)
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("--target-counts must contain positive integers.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("--target-counts must be strictly increasing.")
    if targets[0] <= args.n_clusters:
        raise ValueError("The smallest target must be greater than --n-clusters.")
    return targets


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
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


def _expected_feature_static_spec(
    *,
    catalog_fingerprint: str,
    dino_fingerprint: str,
) -> dict[str, Any]:
    return {
        **DINO_FEATURE_STATIC_SPEC,
        "catalog_fingerprint": catalog_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "preprocessing_fingerprint": _sha256_json(PREPROCESS_SPEC),
    }


def _sample_key_order_fingerprint(sample_keys: Sequence[str]) -> str:
    return _sha256_json({"sample_keys": list(sample_keys)})


def _validate_feature_cache(
    payload: Mapping[str, Any],
    *,
    sample_keys: Sequence[str],
    expected_static_spec: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
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
    dynamic_spec = {
        **expected_static_spec,
        "feature_amp": feature_amp,
        "feature_device_type": feature_device_type,
    }
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
                f"features shape must be {(len(sample_keys), 768)}, "
                f"got {tuple(features.shape)}"
            )
        elif not bool(torch.isfinite(features).all()):
            mismatches.append("features contains NaN or Inf")
        elif bool((torch.linalg.vector_norm(features, dim=1) <= 0.0).any()):
            mismatches.append("features contains a zero-norm row")
    if mismatches:
        raise ValueError("DINO feature cache mismatch:\n- " + "\n- ".join(mismatches))
    assert isinstance(features, torch.Tensor)
    return features.detach().cpu().contiguous(), {
        **dynamic_spec,
        "feature_spec_fingerprint": expected_spec_fingerprint,
        "key_order_fingerprint": _sample_key_order_fingerprint(sample_keys),
        "normalized": normalized,
    }


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _payload_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _stage_and_publish(
    torch_payloads: Mapping[Path, Any],
    text_payloads: Mapping[Path, bytes],
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
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
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
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
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


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _encode_csv(
    *,
    sample_keys: Sequence[str],
    cluster_ids: torch.Tensor,
    center_distances: torch.Tensor,
    seed_keys: Sequence[str],
    selection_result: Any,
    target_counts: Sequence[int],
) -> bytes:
    fieldnames = [
        "sample_key",
        "labeled_name",
        "cluster_id",
        "cluster_size",
        "center_distance",
        "center_rank_in_cluster",
        "is_center_seed",
        "first_selected_target",
        "selection_order",
        "selected_0040",
        "selection_rank_0040",
    ]
    for target in target_counts:
        fieldnames.extend((f"selected_{target:04d}", f"selection_rank_{target:04d}"))

    key_to_index = {key: index for index, key in enumerate(sample_keys)}
    cluster_sizes = Counter(int(value) for value in cluster_ids.tolist())
    center_ranks: dict[str, int] = {}
    for cluster_id in sorted(cluster_sizes):
        members = sorted(
            (
                key
                for key in sample_keys
                if int(cluster_ids[key_to_index[key]]) == cluster_id
            ),
            key=lambda key: (
                float(center_distances[key_to_index[key]]),
                key,
            ),
        )
        center_ranks.update(
            {key: rank for rank, key in enumerate(members, start=1)}
        )
    overall_rank = {
        key: rank
        for rank, key in enumerate(selection_result.selection_order, start=1)
    }
    seed_set = set(seed_keys)
    seed_rank = {
        key: rank for rank, key in enumerate(selection_result.seed_keys, start=1)
    }

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for key in sample_keys:
        index = key_to_index[key]
        cluster_id = int(cluster_ids[index])
        row: dict[str, Any] = {
            "sample_key": key,
            "labeled_name": key.rsplit("/", 1)[-1],
            "cluster_id": cluster_id,
            "cluster_size": cluster_sizes[cluster_id],
            "center_distance": f"{float(center_distances[index]):.10f}",
            "center_rank_in_cluster": center_ranks[key],
            "is_center_seed": int(key in seed_set),
            "first_selected_target": selection_result.first_selected_target.get(key, ""),
            "selection_order": overall_rank.get(key, ""),
            "selected_0040": int(key in seed_set),
            "selection_rank_0040": seed_rank.get(key, ""),
        }
        for target in target_counts:
            ranks = selection_result.selection_rank[int(target)]
            row[f"selected_{target:04d}"] = int(key in ranks)
            row[f"selection_rank_{target:04d}"] = ranks.get(key, "")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _repo_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        REPO_ROOT / "select_kmeans_only.py",
        REPO_ROOT / "utils" / "kmeans_only.py",
        REPO_ROOT / "utils" / "dataloader.py",
        REPO_ROOT / "utils" / "checkpoint_pc_hbm.py",
    ):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


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
    if args.n_clusters > len(sample_keys):
        raise ValueError("--n-clusters exceeds the active catalog size.")
    if targets[-1] > len(sample_keys):
        raise ValueError("Largest target exceeds the active catalog size.")

    catalog_fingerprint = _catalog_fingerprint(items)
    dino_fingerprint = _sha256_file(DINO_WEIGHT_PATH)
    feature_payload = torch.load(
        args.features_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(feature_payload, Mapping):
        raise TypeError("DINO feature cache must contain a mapping payload.")
    features, feature_metadata = _validate_feature_cache(
        feature_payload,
        sample_keys=sample_keys,
        expected_static_spec=_expected_feature_static_spec(
            catalog_fingerprint=catalog_fingerprint,
            dino_fingerprint=dino_fingerprint,
        ),
    )

    kmeans_fit = fit_kmeans_only(
        sample_keys,
        features,
        n_clusters=args.n_clusters,
        random_seed=args.seed,
    )
    selection_result = build_kmeans_only_nested_splits(
        sample_keys,
        kmeans_fit.cluster_ids,
        kmeans_fit.center_distances,
        kmeans_fit.seed_keys,
        target_counts=targets,
    )
    seed_keys = sorted(selection_result.seed_keys)
    seed_fingerprint = compute_labeled_split_fingerprint(seed_keys)
    split_fingerprints = {
        int(target): compute_labeled_split_fingerprint(
            selection_result.splits[int(target)]
        )
        for target in targets
    }
    first_added = selection_result.selection_order[len(seed_keys)]
    feature_cache_sha256 = _sha256_file(args.features_path)
    dry_report = {
        "catalog_count": len(sample_keys),
        "catalog_fingerprint": catalog_fingerprint,
        "feature_cache": _repo_relative(args.features_path),
        "feature_cache_sha256": feature_cache_sha256,
        "protocol": KMEANS_ONLY_PROTOCOL_VERSION,
        "n_clusters": args.n_clusters,
        "random_seed": args.seed,
        "quota_policy": "sqrt_remaining_capacity_largest_remainder",
        "within_cluster_order": "(center_distance, sample_key)",
        "seed_fingerprint": seed_fingerprint,
        "first_added": first_added,
        "targets": {
            str(target): split_fingerprints[int(target)] for target in targets
        },
        "uses_selector": False,
        "uses_score_cache": False,
        "uses_pc_bacs_score": False,
        "uses_gt": False,
        "dedup": False,
        "model_forward": False,
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0

    torch_payloads: dict[Path, Any] = {}
    text_payloads: dict[Path, bytes] = {}
    outputs: dict[str, Any] = {}

    def add_split_artifacts(
        label: int,
        keys: list[str],
        *,
        seed_artifact: bool = False,
    ) -> None:
        names = labeled_names_from_keys(keys)
        artifact_stem = f"kmeans_only_{label:04d}"
        if seed_artifact:
            artifact_stem += "_seed"
        pt_path = args.output_dir / f"{artifact_stem}_keys.pt"
        txt_path = args.output_dir / f"{artifact_stem}_labeled_names.txt"
        torch_payloads[pt_path] = keys
        text_payloads[txt_path] = ("\n".join(names) + "\n").encode("utf-8")
        outputs[str(label)] = {
            "count": len(keys),
            "split_fingerprint": compute_labeled_split_fingerprint(keys),
            "pt_path": pt_path.name,
            "txt_path": txt_path.name,
            "txt_sha256": hashlib.sha256(text_payloads[txt_path]).hexdigest(),
        }

    add_split_artifacts(args.n_clusters, seed_keys, seed_artifact=True)
    for target in targets:
        add_split_artifacts(int(target), selection_result.splits[int(target)])

    csv_path = args.output_dir / "kmeans_only_assignments.csv"
    csv_payload = _encode_csv(
        sample_keys=sample_keys,
        cluster_ids=kmeans_fit.cluster_ids,
        center_distances=kmeans_fit.center_distances,
        seed_keys=seed_keys,
        selection_result=selection_result,
        target_counts=targets,
    )
    text_payloads[csv_path] = csv_payload

    cluster_sizes = Counter(int(value) for value in kmeans_fit.cluster_ids.tolist())
    manifest = {
        "format_version": MANIFEST_VERSION,
        "method": "DINO-KMeans-only",
        "protocol": KMEANS_ONLY_PROTOCOL_VERSION,
        "repo_commit": _repo_commit(),
        "implementation_fingerprint": _implementation_fingerprint(),
        "dataset": {
            "root": _repo_relative(args.data_root),
            "train_sets": list(args.train_sets),
            "sample_count": len(sample_keys),
            "catalog_fingerprint": catalog_fingerprint,
            "image_fingerprint": catalog_fingerprint,
            "image_fingerprint_scheme": "pc_bacs_catalog_v1_per_image_sha256",
        },
        "dino_feature_cache": {
            "path": _repo_relative(args.features_path),
            "sha256": feature_cache_sha256,
            "feature_spec_fingerprint": feature_metadata[
                "feature_spec_fingerprint"
            ],
            "key_order_fingerprint": feature_metadata["key_order_fingerprint"],
            "dino_fingerprint": dino_fingerprint,
            "preprocessing_fingerprint": feature_metadata[
                "preprocessing_fingerprint"
            ],
            "feature_amp": feature_metadata["feature_amp"],
            "feature_device_type": feature_metadata["feature_device_type"],
            "shape": [len(sample_keys), 768],
            "dtype": "float32",
            "l2_normalized_before_kmeans": True,
        },
        "selection": {
            "n_clusters": args.n_clusters,
            "random_state": args.seed,
            "n_init": 10,
            "algorithm": "lloyd",
            "center_definition": "catalog_ordered_float64_mean_of_final_labels",
            "center_distance": "euclidean_on_l2_normalized_dino_features",
            "seed_rule": "one center-nearest sample per cluster",
            "seed_tie_break": "sample_key",
            "seed_count": len(seed_keys),
            "seed_fingerprint": seed_fingerprint,
            "quota_policy": "sqrt_remaining_capacity_largest_remainder",
            "quota_tie_break": "cluster_id",
            "within_cluster_order": "(center_distance, sample_key)",
            "target_counts": list(targets),
            "first_added": first_added,
            "cluster_sizes": {
                str(cluster_id): cluster_sizes[cluster_id]
                for cluster_id in sorted(cluster_sizes)
            },
            "rounds": [
                {
                    **asdict(round_record),
                    "quotas": {
                        str(cluster_id): quota
                        for cluster_id, quota in round_record.quotas.items()
                    },
                }
                for round_record in selection_result.rounds
            ],
            "uses_selector": False,
            "uses_score_cache": False,
            "uses_pc_bacs_score": False,
            "uses_gt": False,
            "dedup": False,
            "model_forward": False,
        },
        "outputs": outputs,
        "assignments_csv": {
            "path": csv_path.name,
            "sha256": hashlib.sha256(csv_payload).hexdigest(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
        },
    }
    manifest_path = args.output_dir / "kmeans_only_manifest.json"
    text_payloads[manifest_path] = _encode_json(manifest)

    _stage_and_publish(torch_payloads, text_payloads)
    print(json.dumps(dry_report, indent=2, sort_keys=True))
    print(f"Saved KMeans-only artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
