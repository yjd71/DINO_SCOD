"""Deterministic RDSM baselines over stable DINOv2 feature catalogs.

``original`` reproduces centroid sampling independently for every requested
budget.  ``seeded`` keeps a shared bootstrap split and expands it from one
KMeans fit, which makes the three saved splits strictly nested.
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchvision
from PIL import Image, __version__ as pillow_version
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from torchvision import transforms
from tqdm import tqdm

from selection.artifacts import (
    atomic_json_save,
    atomic_torch_save,
    compute_catalog_fingerprint,
    compute_image_fingerprint,
    compute_key_order_fingerprint,
    file_sha256,
    load_split_keys,
    save_split_keys,
    stable_fingerprint,
)
from selection.protocol import (
    SamplingProtocol,
    add_target_counts_argument,
    protocol_from_args,
)


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
FEATURE_CACHE_SCHEMA = "rdsm-dinov2-cls-v1"
SELECTION_SCHEMA = "rdsm-selection-v1"
KMEANS_N_INIT = 10
KMEANS_ALGORITHM = "lloyd"
INPUT_SIZE = 392
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class CatalogItem:
    key: str
    path: Path
    subset: str


@dataclass(frozen=True, slots=True)
class RDSMSelectionResult:
    """Pure selection result; every split is sorted by stable sample key."""

    mode: str
    splits: Mapping[int, tuple[str, ...]]
    acquisition_order: tuple[str, ...]
    fitted_cluster_counts: tuple[int, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic RDSM-original or RDSM-seeded splits."
    )
    parser.add_argument("--mode", choices=("original", "seeded"), required=True)
    add_target_counts_argument(parser, required=True)
    parser.add_argument(
        "--debug-custom-counts",
        "--allow-custom-counts",
        dest="allow_custom_counts",
        action="store_true",
        help="Allow non-formal counts for isolated debug runs only.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument(
        "--train-sets", nargs="+", default=("TR-CAMO", "TR-COD10K")
    )
    parser.add_argument(
        "--bootstrap-split",
        type=Path,
        help="Required for seeded mode; must contain exactly the first budget.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--features-path",
        type=Path,
        help="Stable-key .pt DINO feature cache (default: shared cache under data root).",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Explicitly rebuild and atomically replace a mismatched feature cache.",
    )
    parser.add_argument("--dino-repo", type=Path, default=Path("./dinov2"))
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=Path("./dinov2_vitb14_pretrain.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args(argv)


def resolve_protocol(args: argparse.Namespace) -> SamplingProtocol:
    """Resolve the common protocol and keep custom budgets debug-only."""

    return protocol_from_args(args, allow_custom=bool(args.allow_custom_counts))


def rgb_loader(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise FileNotFoundError(f"Failed to read RGB image: {path}") from error


def build_catalog(data_root: Path, train_sets: Sequence[str]) -> list[CatalogItem]:
    if not train_sets:
        raise ValueError("--train-sets must contain at least one subset")
    if len(set(train_sets)) != len(train_sets):
        raise ValueError("--train-sets contains duplicate subset names")

    items: list[CatalogItem] = []
    for subset in train_sets:
        image_root = data_root / subset / "im"
        if not image_root.is_dir():
            raise FileNotFoundError(f"training image directory does not exist: {image_root}")
        for path in sorted(image_root.iterdir(), key=lambda value: value.name):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                items.append(
                    CatalogItem(
                        key=f"{subset}/{path.stem}",
                        path=path.resolve(),
                        subset=str(subset),
                    )
                )

    items.sort(key=lambda item: item.key)
    if not items:
        raise ValueError(f"No training images found under: {data_root}")
    keys = [item.key for item in items]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise ValueError(f"catalog contains duplicate sample keys: {duplicates[:3]!r}")
    return items


def _canonicalize_feature_inputs(
    sample_keys: Sequence[str], features: np.ndarray | torch.Tensor
) -> tuple[list[str], np.ndarray]:
    if isinstance(sample_keys, (str, bytes)):
        raise TypeError("sample_keys must be a sequence of strings")
    keys = list(sample_keys)
    if not keys or not all(isinstance(key, str) and key for key in keys):
        raise ValueError("sample_keys must contain non-empty strings")
    if len(keys) != len(set(keys)):
        raise ValueError("sample_keys contains duplicates")

    if isinstance(features, torch.Tensor):
        array = features.detach().cpu().numpy()
    else:
        array = np.asarray(features)
    if array.ndim != 2 or array.shape[0] != len(keys) or array.shape[1] <= 0:
        raise ValueError(
            "features must have shape [len(sample_keys), feature_dim], got "
            f"{tuple(array.shape)}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("features must be numeric")
    if not np.isfinite(array).all():
        raise ValueError("features contains NaN or infinity")

    order = sorted(range(len(keys)), key=keys.__getitem__)
    ordered_keys = [keys[index] for index in order]
    ordered_features = np.ascontiguousarray(array[order], dtype=np.float32)
    return ordered_keys, ordered_features


def _validate_target_counts(target_counts: Sequence[int], pool_size: int) -> tuple[int, ...]:
    counts = tuple(target_counts)
    if not counts or any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("target_counts must contain integers")
    if tuple(sorted(set(counts))) != counts or counts[0] <= 0:
        raise ValueError("target_counts must be positive, strictly increasing, and unique")
    if counts[-1] > pool_size:
        raise ValueError(
            f"largest target count {counts[-1]} exceeds catalog size {pool_size}"
        )
    return counts


def _fit_kmeans(features: np.ndarray, n_clusters: int, seed: int) -> KMeans:
    return KMeans(
        n_clusters=int(n_clusters),
        random_state=int(seed),
        n_init=KMEANS_N_INIT,
        algorithm=KMEANS_ALGORITHM,
    ).fit(features)


def _nearest_unique_center_samples(
    sample_keys: Sequence[str],
    features: np.ndarray,
    centers: np.ndarray,
) -> list[int]:
    """Choose one unique representative per center, including degenerate fits."""

    center_order = sorted(
        range(len(centers)),
        key=lambda index: (np.asarray(centers[index], dtype=np.float64).tobytes(), index),
    )
    selected: set[int] = set()
    for center_index in center_order:
        distances = np.linalg.norm(features - centers[center_index], axis=1)
        candidate_order = sorted(
            range(len(sample_keys)),
            key=lambda index: (float(distances[index]), sample_keys[index]),
        )
        representative = next(
            (index for index in candidate_order if index not in selected), None
        )
        if representative is None:
            raise RuntimeError("KMeans centers cannot be mapped to unique samples")
        selected.add(representative)
    if len(selected) != len(centers):
        raise RuntimeError("RDSM-original failed to produce one sample per center")
    return sorted(selected, key=sample_keys.__getitem__)


def select_rdsm_original(
    sample_keys: Sequence[str],
    features: np.ndarray | torch.Tensor,
    target_counts: Sequence[int],
    *,
    seed: int,
) -> RDSMSelectionResult:
    """Fit an independent KMeans for each budget and select its representatives."""

    keys, array = _canonicalize_feature_inputs(sample_keys, features)
    counts = _validate_target_counts(target_counts, len(keys))
    splits: dict[int, tuple[str, ...]] = {}
    for count in counts:
        fitted = _fit_kmeans(array, count, seed)
        selected_indices = _nearest_unique_center_samples(
            keys, array, np.asarray(fitted.cluster_centers_)
        )
        split = tuple(sorted(keys[index] for index in selected_indices))
        if len(split) != count or len(set(split)) != count:
            raise RuntimeError(
                f"RDSM-original produced {len(split)} samples for target {count}"
            )
        splits[count] = split
    return RDSMSelectionResult(
        mode="original",
        splits=splits,
        acquisition_order=(),
        fitted_cluster_counts=counts,
    )


def _seeded_acquisition_order(
    sample_keys: Sequence[str],
    labels: np.ndarray,
    center_distances: np.ndarray,
    bootstrap_keys: Sequence[str],
) -> list[str]:
    """Round-robin deterministic per-cluster queues ordered by distance/key."""

    bootstrap = set(bootstrap_keys)
    cluster_ids = sorted(set(int(label) for label in labels.tolist()))
    queues: dict[int, list[int]] = {}
    cluster_identity: dict[int, str] = {}
    for cluster_id in cluster_ids:
        members = [
            index for index, label in enumerate(labels) if int(label) == cluster_id
        ]
        members.sort(
            key=lambda index: (float(center_distances[index]), sample_keys[index])
        )
        cluster_identity[cluster_id] = sample_keys[members[0]]
        queues[cluster_id] = [
            index for index in members if sample_keys[index] not in bootstrap
        ]

    # Cluster labels are implementation details.  Ordering clusters by their
    # nearest stable sample key makes label permutations selection-equivalent.
    ordered_clusters = sorted(
        cluster_ids, key=lambda cluster_id: cluster_identity[cluster_id]
    )
    positions = {cluster_id: 0 for cluster_id in ordered_clusters}
    acquisition: list[str] = []
    while True:
        progressed = False
        for cluster_id in ordered_clusters:
            position = positions[cluster_id]
            queue = queues[cluster_id]
            if position >= len(queue):
                continue
            acquisition.append(sample_keys[queue[position]])
            positions[cluster_id] = position + 1
            progressed = True
        if not progressed:
            break

    expected = len(sample_keys) - len(bootstrap)
    if len(acquisition) != expected or len(set(acquisition)) != expected:
        raise RuntimeError("RDSM-seeded acquisition order does not cover the candidate pool")
    return acquisition


def select_rdsm_seeded(
    sample_keys: Sequence[str],
    features: np.ndarray | torch.Tensor,
    target_counts: Sequence[int],
    bootstrap_keys: Sequence[str],
    *,
    seed: int,
) -> RDSMSelectionResult:
    """Expand one common bootstrap with a single deterministic KMeans fit."""

    keys, array = _canonicalize_feature_inputs(sample_keys, features)
    counts = _validate_target_counts(target_counts, len(keys))
    bootstrap = list(bootstrap_keys)
    if not all(isinstance(key, str) for key in bootstrap):
        raise TypeError("bootstrap_keys must contain only strings")
    if len(bootstrap) != counts[0] or len(set(bootstrap)) != len(bootstrap):
        raise ValueError(
            f"bootstrap must contain exactly {counts[0]} unique sample keys"
        )
    catalog = set(keys)
    missing = sorted(set(bootstrap) - catalog)
    if missing:
        raise ValueError(f"bootstrap keys are absent from the catalog: {missing[:3]!r}")
    bootstrap = sorted(bootstrap)

    fitted = _fit_kmeans(array, counts[0], seed)
    labels = np.asarray(fitted.labels_, dtype=np.int64)
    centers = np.asarray(fitted.cluster_centers_)
    if labels.shape != (len(keys),):
        raise RuntimeError("KMeans returned an invalid label vector")
    center_distances = np.linalg.norm(array - centers[labels], axis=1)
    acquisition = _seeded_acquisition_order(
        keys, labels, center_distances, bootstrap
    )

    splits: dict[int, tuple[str, ...]] = {}
    bootstrap_set = set(bootstrap)
    for count in counts:
        needed = count - counts[0]
        selected = bootstrap_set | set(acquisition[:needed])
        if len(selected) != count:
            raise RuntimeError(
                f"RDSM-seeded produced {len(selected)} samples for target {count}"
            )
        splits[count] = tuple(sorted(selected))
    for smaller, larger in zip(counts, counts[1:]):
        if not set(splits[smaller]) < set(splits[larger]):
            raise RuntimeError("RDSM-seeded splits are not strictly nested")

    return RDSMSelectionResult(
        mode="seeded",
        splits=splits,
        acquisition_order=tuple(acquisition[: counts[-1] - counts[0]]),
        fitted_cluster_counts=(counts[0],),
    )


# Short aliases keep the pure functions convenient for experiment notebooks.
select_original = select_rdsm_original
select_seeded = select_rdsm_seeded


def _feature_cache_spec(
    *,
    sample_keys: Sequence[str],
    image_fingerprint: str,
    dino_repo: Path,
    dino_checkpoint: Path,
) -> dict[str, Any]:
    source_files = sorted(dino_repo.rglob("*.py"))
    if not source_files or not (dino_repo / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"DINO repository Python sources do not exist under: {dino_repo}"
        )
    if not dino_checkpoint.is_file():
        raise FileNotFoundError(f"DINO checkpoint does not exist: {dino_checkpoint}")
    return {
        "schema": FEATURE_CACHE_SCHEMA,
        "catalog_fingerprint": compute_catalog_fingerprint(sample_keys),
        "key_order_fingerprint": compute_key_order_fingerprint(sample_keys),
        "image_fingerprint": image_fingerprint,
        "model": "dinov2_vitb14",
        "dino_checkpoint_sha256": file_sha256(dino_checkpoint),
        "dino_source_fingerprint": stable_fingerprint(
            {
                path.relative_to(dino_repo).as_posix(): file_sha256(path)
                for path in source_files
            }
        ),
        "preprocessing": {
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "mean": list(NORMALIZE_MEAN),
            "std": list(NORMALIZE_STD),
            "interpolation": "bilinear",
            "antialias": True,
            "color_mode": "RGB",
        },
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": pillow_version,
        },
        "dtype": "torch.float32",
    }


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_feature_cache(
    payload: Any,
    *,
    expected_spec: Mapping[str, Any],
    sample_keys: Sequence[str],
) -> torch.Tensor:
    if not isinstance(payload, Mapping):
        raise ValueError("DINO feature cache must contain a mapping payload")
    if payload.get("spec") != dict(expected_spec):
        raise ValueError(
            "DINO feature cache identity mismatch; pass --rebuild-feature-cache "
            "to replace it explicitly"
        )
    if payload.get("sample_keys") != list(sample_keys):
        raise ValueError("DINO feature cache sample-key order mismatch")
    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        raise ValueError("DINO feature cache is missing its feature tensor")
    if features.dtype != torch.float32 or features.ndim != 2:
        raise ValueError("cached DINO features must be a 2-D float32 tensor")
    if features.shape[0] != len(sample_keys) or features.shape[1] <= 0:
        raise ValueError("cached DINO feature shape does not match the catalog")
    if not torch.isfinite(features).all().item():
        raise ValueError("cached DINO features contain NaN or infinity")
    return features.contiguous()


def _extract_dino_features(
    items: Sequence[CatalogItem],
    *,
    dino_repo: Path,
    dino_checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    dino = torch.hub.load(
        str(dino_repo),
        "dinov2_vitb14",
        source="local",
        pretrained=False,
    )
    state = _torch_load(dino_checkpoint)
    dino.load_state_dict(state)
    dino = dino.to(device)
    dino.eval()

    features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(items), batch_size), desc="DINOv2 features"):
        batch_items = items[start : start + batch_size]
        images = torch.stack(
            [transform(rgb_loader(item.path)) for item in batch_items], dim=0
        ).to(device)
        with torch.inference_mode():
            batch_features = dino(images)
        if not isinstance(batch_features, torch.Tensor) or batch_features.ndim != 2:
            raise RuntimeError("DINOv2 must return a [batch, feature_dim] tensor")
        features.append(batch_features.detach().float().cpu())
    result = torch.cat(features, dim=0).contiguous()
    if result.shape[0] != len(items) or not torch.isfinite(result).all().item():
        raise RuntimeError("DINOv2 feature extraction produced invalid output")
    return result


def _load_or_extract_features(
    *,
    items: Sequence[CatalogItem],
    cache_path: Path,
    expected_spec: Mapping[str, Any],
    args: argparse.Namespace,
) -> torch.Tensor:
    sample_keys = [item.key for item in items]
    if cache_path.exists() and not args.rebuild_feature_cache:
        return _validate_feature_cache(
            _torch_load(cache_path),
            expected_spec=expected_spec,
            sample_keys=sample_keys,
        )

    features = _extract_dino_features(
        items,
        dino_repo=args.dino_repo,
        dino_checkpoint=args.dino_checkpoint,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )
    payload = {
        "spec": dict(expected_spec),
        "sample_keys": sample_keys,
        "features": features,
    }
    atomic_torch_save(
        payload,
        cache_path,
        refuse_mismatch=not args.rebuild_feature_cache,
    )
    return _validate_feature_cache(
        _torch_load(cache_path), expected_spec=expected_spec, sample_keys=sample_keys
    )


def _resolve_paths(
    args: argparse.Namespace, protocol: SamplingProtocol
) -> tuple[Path, Path]:
    data_root = args.data_root.resolve()
    if args.output_dir is None:
        family_root = data_root / "splits" / "rdsm"
        if not protocol.is_formal:
            family_root = family_root / "debug"
        output_dir = (
            family_root
            / args.mode
            / protocol.name
            / f"seed{int(args.seed)}"
        )
    else:
        output_dir = args.output_dir.resolve()

    if not protocol.is_formal:
        debug_root = (data_root / "splits" / "rdsm" / "debug").resolve()
        try:
            output_dir.relative_to(debug_root)
        except ValueError as error:
            raise ValueError(
                "custom target counts must write below the isolated debug root: "
                f"{debug_root}"
            ) from error

    cache_path = (
        args.features_path.resolve()
        if args.features_path is not None
        else (data_root / "splits" / "cache" / "rdsm_dinov2_vitb14_cls.pt")
    )
    if cache_path.suffix.lower() != ".pt":
        raise ValueError("--features-path must be a metadata-bearing .pt cache")
    return output_dir, cache_path


def _validate_cli_args(args: argparse.Namespace, protocol: SamplingProtocol) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.mode == "seeded" and args.bootstrap_split is None:
        raise ValueError("--bootstrap-split is required for seeded mode")
    if args.mode == "original" and args.bootstrap_split is not None:
        raise ValueError("--bootstrap-split is only valid for seeded mode")
    if not protocol.is_formal and not args.allow_custom_counts:
        raise ValueError("custom target counts require --debug-custom-counts")


def _save_selection_artifacts(
    *,
    result: RDSMSelectionResult,
    output_dir: Path,
    protocol: SamplingProtocol,
    seed: int,
    sample_keys: Sequence[str],
    catalog_fingerprint: str,
    image_fingerprint: str,
    cache_path: Path,
    feature_spec: Mapping[str, Any],
    bootstrap_path: Path | None,
    bootstrap_keys: Sequence[str] | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_outputs: dict[str, Any] = {}
    reloaded_splits: dict[int, list[str]] = {}
    for count in protocol.target_counts:
        path = output_dir / f"rdsm_{result.mode}_{count:04d}_seed{seed}.pt"
        split = list(result.splits[count])
        if len(split) != count or split != sorted(split):
            raise RuntimeError(f"invalid in-memory split for target {count}")
        fingerprint = save_split_keys(path, split)
        reloaded = load_split_keys(
            path, catalog_keys=sample_keys, expected_count=count
        )
        if reloaded != split:
            raise RuntimeError(f"saved split failed round-trip validation: {path}")
        reloaded_splits[count] = reloaded
        split_outputs[str(count)] = {
            "path": str(path.resolve()),
            "count": count,
            "split_fingerprint": fingerprint,
            "file_sha256": file_sha256(path),
        }

    nested = all(
        set(reloaded_splits[smaller]) < set(reloaded_splits[larger])
        for smaller, larger in zip(protocol.target_counts, protocol.target_counts[1:])
    )
    if result.mode == "seeded" and not nested:
        raise RuntimeError("saved RDSM-seeded splits are not strictly nested")
    if result.mode == "seeded" and reloaded_splits[protocol.bootstrap_count] != list(
        bootstrap_keys or ()
    ):
        raise RuntimeError("seeded first split differs from the common bootstrap")

    manifest: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "method": f"RDSM-{result.mode}",
        "mode": result.mode,
        "seed": int(seed),
        "protocol": {
            "name": protocol.name,
            "target_counts": list(protocol.target_counts),
            "bootstrap_count": protocol.bootstrap_count,
            "is_formal": protocol.is_formal,
        },
        "catalog": {
            "sample_count": len(sample_keys),
            "catalog_fingerprint": catalog_fingerprint,
            "key_order_fingerprint": compute_key_order_fingerprint(sample_keys),
            "image_fingerprint": image_fingerprint,
        },
        "feature_cache": {
            "path": str(cache_path.resolve()),
            "spec_fingerprint": stable_fingerprint(feature_spec),
            "file_sha256": file_sha256(cache_path),
        },
        "kmeans": {
            "fitted_cluster_counts": list(result.fitted_cluster_counts),
            "random_state": int(seed),
            "n_init": KMEANS_N_INIT,
            "algorithm": KMEANS_ALGORITHM,
            "sklearn_version": sklearn_version,
            "original_budgets_are_independent": result.mode == "original",
            "seeded_cluster_queue_order": (
                "round_robin(distance_to_assigned_center,sample_key)"
                if result.mode == "seeded"
                else None
            ),
        },
        "bootstrap": (
            {
                "path": str(bootstrap_path.resolve()),
                "count": len(bootstrap_keys or ()),
                "key_fingerprint": stable_fingerprint(list(bootstrap_keys or ())),
                "file_sha256": file_sha256(bootstrap_path),
            }
            if bootstrap_path is not None
            else None
        ),
        "selection": {
            "strictly_nested": nested,
            "strict_nesting_enforced": result.mode == "seeded",
            "acquisition_count": len(result.acquisition_order),
            "acquisition_fingerprint": (
                stable_fingerprint(list(result.acquisition_order))
                if result.acquisition_order
                else None
            ),
        },
        "outputs": split_outputs,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json_save(manifest, output_dir / "selection_manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = resolve_protocol(args)
    _validate_cli_args(args, protocol)
    output_dir, cache_path = _resolve_paths(args, protocol)

    items = build_catalog(args.data_root.resolve(), args.train_sets)
    sample_keys = [item.key for item in items]
    _validate_target_counts(protocol.target_counts, len(sample_keys))
    catalog_fingerprint = compute_catalog_fingerprint(sample_keys)
    image_fingerprint = compute_image_fingerprint(
        sample_keys, {item.key: item.path for item in items}
    )
    feature_spec = _feature_cache_spec(
        sample_keys=sample_keys,
        image_fingerprint=image_fingerprint,
        dino_repo=args.dino_repo.resolve(),
        dino_checkpoint=args.dino_checkpoint.resolve(),
    )
    features = _load_or_extract_features(
        items=items,
        cache_path=cache_path,
        expected_spec=feature_spec,
        args=argparse.Namespace(
            **{
                **vars(args),
                "dino_repo": args.dino_repo.resolve(),
                "dino_checkpoint": args.dino_checkpoint.resolve(),
            }
        ),
    )

    bootstrap_keys: list[str] | None = None
    if args.mode == "seeded":
        bootstrap_keys = load_split_keys(
            args.bootstrap_split,
            catalog_keys=sample_keys,
            expected_count=protocol.bootstrap_count,
        )
        result = select_rdsm_seeded(
            sample_keys,
            features,
            protocol.target_counts,
            bootstrap_keys,
            seed=args.seed,
        )
    else:
        result = select_rdsm_original(
            sample_keys, features, protocol.target_counts, seed=args.seed
        )

    _save_selection_artifacts(
        result=result,
        output_dir=output_dir,
        protocol=protocol,
        seed=args.seed,
        sample_keys=sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        cache_path=cache_path,
        feature_spec=feature_spec,
        bootstrap_path=args.bootstrap_split,
        bootstrap_keys=bootstrap_keys,
    )
    print(f"Saved {result.mode} RDSM splits to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
