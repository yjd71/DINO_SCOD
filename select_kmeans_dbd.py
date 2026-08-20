from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from select_global_additive import (
    _expected_source_spec,
    _load_selector_identity,
    _validate_source_score_cache,
)
from select_kmeans_only import (
    DINO_WEIGHT_PATH,
    _catalog_fingerprint,
    _encode_json,
    _expected_feature_static_spec,
    _package_version,
    _repo_commit,
    _repo_relative,
    _sha256_file,
    _sha256_json,
    _stage_and_publish,
    _validate_feature_cache,
)
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.dataloader import SelectionPoolDataset
from utils.kmeans_dbd import (
    KMEANS_DBD_DEDUP_PROTOCOL_VERSION,
    build_kmeans_dbd_nested_splits,
)
from utils.kmeans_only import fit_kmeans_only, labeled_names_from_keys


REPO_ROOT = Path(__file__).resolve().parent
FORMAL_TARGET_COUNTS = (41, 202, 404, 808)
FORMAL_N_CLUSTERS = 40
FORMAL_RANDOM_SEED = 2025
FORMAL_DEDUP_THRESHOLD = 0.95
MANIFEST_VERSION = 1
ARTIFACT_PREFIX = "kmeans_dbd_dedup"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact nested labeled splits with deterministic DINO KMeans, "
            "boundary disagreement ranking, and same-cluster DINO deduplication."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument(
        "--train-sets",
        nargs="+",
        default=["TR-CAMO", "TR-COD10K"],
    )
    parser.add_argument(
        "--dino-feature-cache",
        type=Path,
        default=Path("./Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt"),
    )
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--source-score-cache", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=FORMAL_N_CLUSTERS)
    parser.add_argument("--seed", type=int, default=FORMAL_RANDOM_SEED)
    parser.add_argument(
        "--target-counts",
        nargs="+",
        type=int,
        default=list(FORMAL_TARGET_COUNTS),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=FORMAL_DEDUP_THRESHOLD,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./Dataset/COD/splits/kmeans_dbd_dedup"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.dino_feature_cache = args.dino_feature_cache.resolve()
    args.selector_checkpoint = args.selector_checkpoint.resolve()
    args.source_score_cache = args.source_score_cache.resolve()
    args.output_dir = args.output_dir.resolve()


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {args.data_root}")
    for label, path in (
        ("DINO feature cache", args.dino_feature_cache),
        ("selector checkpoint", args.selector_checkpoint),
        ("source score cache", args.source_score_cache),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
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
    if not 0.0 < args.dedup_threshold <= 1.0:
        raise ValueError("--dedup-threshold must be in (0, 1].")
    return targets


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        REPO_ROOT / "select_kmeans_dbd.py",
        REPO_ROOT / "utils" / "kmeans_dbd.py",
        REPO_ROOT / "utils" / "kmeans_only.py",
        REPO_ROOT / "select_kmeans_only.py",
        REPO_ROOT / "select_global_additive.py",
        REPO_ROOT / "utils" / "dataloader.py",
        REPO_ROOT / "utils" / "checkpoint_pc_hbm.py",
    ):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _encode_csv(
    *,
    sample_keys: Sequence[str],
    cluster_ids: torch.Tensor,
    center_distances: torch.Tensor,
    boundary_disagreement: torch.Tensor,
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
        "boundary_disagreement",
        "boundary_rank_in_cluster",
        "is_center_seed",
        "first_selected_target",
        "selection_order",
        "decision",
        "dedup_skip_count",
        "max_same_cluster_cosine",
        "reference_key",
        "relaxed_backfill",
        "selected_0040",
        "selection_rank_0040",
    ]
    for target in target_counts:
        fieldnames.extend((f"selected_{target:04d}", f"selection_rank_{target:04d}"))

    key_to_index = {key: index for index, key in enumerate(sample_keys)}
    cluster_sizes = Counter(int(value) for value in cluster_ids.tolist())
    center_ranks: dict[str, int] = {}
    boundary_ranks: dict[str, int] = {}
    for cluster_id in sorted(cluster_sizes):
        members = [
            key
            for key in sample_keys
            if int(cluster_ids[key_to_index[key]]) == cluster_id
        ]
        center_ranks.update(
            {
                key: rank
                for rank, key in enumerate(
                    sorted(
                        members,
                        key=lambda value: (
                            float(center_distances[key_to_index[value]]),
                            value,
                        ),
                    ),
                    start=1,
                )
            }
        )
        boundary_ranks.update(
            {
                key: rank
                for rank, key in enumerate(
                    sorted(
                        members,
                        key=lambda value: (
                            -float(boundary_disagreement[key_to_index[value]]),
                            value,
                        ),
                    ),
                    start=1,
                )
            }
        )

    overall_rank = {
        key: rank
        for rank, key in enumerate(selection_result.selection_order, start=1)
    }
    seed_rank = {
        key: rank for rank, key in enumerate(selection_result.seed_keys, start=1)
    }
    seed_set = set(selection_result.seed_keys)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for key in sample_keys:
        index = key_to_index[key]
        cluster_id = int(cluster_ids[index])
        decision = selection_result.decisions[key]
        maximum = decision.max_cosine_similarity
        row: dict[str, Any] = {
            "sample_key": key,
            "labeled_name": key.rsplit("/", 1)[-1],
            "cluster_id": cluster_id,
            "cluster_size": cluster_sizes[cluster_id],
            "center_distance": f"{float(center_distances[index]):.10f}",
            "center_rank_in_cluster": center_ranks[key],
            "boundary_disagreement": f"{float(boundary_disagreement[index]):.10f}",
            "boundary_rank_in_cluster": boundary_ranks[key],
            "is_center_seed": int(key in seed_set),
            "first_selected_target": decision.first_selected_target or "",
            "selection_order": overall_rank.get(key, ""),
            "decision": decision.decision,
            "dedup_skip_count": decision.skip_count,
            "max_same_cluster_cosine": (
                "" if maximum is None else f"{maximum:.10f}"
            ),
            "reference_key": decision.reference_key or "",
            "relaxed_backfill": int(decision.relaxed_backfill),
            "selected_0040": int(key in seed_set),
            "selection_rank_0040": seed_rank.get(key, ""),
        }
        for target in target_counts:
            ranks = selection_result.selection_rank[int(target)]
            row[f"selected_{target:04d}"] = int(key in ranks)
            row[f"selection_rank_{target:04d}"] = ranks.get(key, "")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


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
        args.dino_feature_cache,
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
    seed_keys = sorted(kmeans_fit.seed_keys)
    selector_metadata, non_pc_fingerprint, selector_fingerprint = (
        _load_selector_identity(
            args.selector_checkpoint,
            selector_seed_keys=seed_keys,
            dino_fingerprint=dino_fingerprint,
        )
    )
    score_payload = torch.load(
        args.source_score_cache,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(score_payload, Mapping):
        raise TypeError("Source score cache must contain a mapping payload.")
    expected_score_spec = _expected_source_spec(
        catalog_fingerprint=catalog_fingerprint,
        selector_fingerprint=selector_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    boundary_disagreement, _unused_global_disagreement = (
        _validate_source_score_cache(
            score_payload,
            sample_keys=sample_keys,
            expected_spec=expected_score_spec,
        )
    )

    selection_result = build_kmeans_dbd_nested_splits(
        sample_keys,
        features,
        kmeans_fit.cluster_ids,
        kmeans_fit.center_distances,
        boundary_disagreement,
        seed_keys,
        target_counts=targets,
        dedup_threshold=args.dedup_threshold,
    )
    seed_fingerprint = compute_labeled_split_fingerprint(seed_keys)
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
        "protocol": KMEANS_DBD_DEDUP_PROTOCOL_VERSION,
        "n_clusters": args.n_clusters,
        "random_seed": args.seed,
        "dedup_threshold": args.dedup_threshold,
        "seed_fingerprint": seed_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "first_added": first_added,
        "targets": {
            str(target): split_fingerprints[int(target)] for target in targets
        },
        "round_dedup_skips": [record.dedup_skips for record in selection_result.rounds],
        "round_dedup_backfill": [
            record.dedup_backfill_count for record in selection_result.rounds
        ],
        "round_relaxed_backfill": [
            record.relaxed_backfill_count for record in selection_result.rounds
        ],
        "ranking_uses_boundary_disagreement": True,
        "ranking_uses_global_disagreement": False,
        "ranking_uses_pc_bacs_value": False,
        "uses_gt": False,
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
        artifact_stem = f"{ARTIFACT_PREFIX}_{label:04d}"
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

    csv_path = args.output_dir / f"{ARTIFACT_PREFIX}_assignments.csv"
    csv_payload = _encode_csv(
        sample_keys=sample_keys,
        cluster_ids=kmeans_fit.cluster_ids,
        center_distances=kmeans_fit.center_distances,
        boundary_disagreement=boundary_disagreement,
        selection_result=selection_result,
        target_counts=targets,
    )
    text_payloads[csv_path] = csv_payload

    feature_cache_sha256 = _sha256_file(args.dino_feature_cache)
    score_cache_sha256 = _sha256_file(args.source_score_cache)
    selector_checkpoint_sha256 = _sha256_file(args.selector_checkpoint)
    cluster_sizes = Counter(int(value) for value in kmeans_fit.cluster_ids.tolist())
    unique_skipped = sum(
        1 for decision in selection_result.decisions.values() if decision.skip_count > 0
    )
    total_skip_events = sum(
        decision.skip_count for decision in selection_result.decisions.values()
    )
    manifest = {
        "format_version": MANIFEST_VERSION,
        "method": "DINO-KMeans + boundary-disagreement + in-cluster DINO dedup",
        "protocol": KMEANS_DBD_DEDUP_PROTOCOL_VERSION,
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
            "path": _repo_relative(args.dino_feature_cache),
            "sha256": feature_cache_sha256,
            "feature_spec_fingerprint": feature_metadata["feature_spec_fingerprint"],
            "key_order_fingerprint": feature_metadata["key_order_fingerprint"],
            "dino_fingerprint": dino_fingerprint,
            "preprocessing_fingerprint": feature_metadata["preprocessing_fingerprint"],
            "shape": [len(sample_keys), 768],
            "dtype": "float32",
            "l2_normalized_before_kmeans_and_dedup": True,
        },
        "selector": {
            "checkpoint": _repo_relative(args.selector_checkpoint),
            "checkpoint_sha256": selector_checkpoint_sha256,
            "epoch": 5,
            "training_design": selector_metadata["training_design"],
            "artifact_role": selector_metadata["artifact_role"],
            "pc_frozen": selector_metadata["pc_frozen"],
            "training_seed_count": len(seed_keys),
            "training_seed_fingerprint": seed_fingerprint,
            "non_pc_decoder_fingerprint": non_pc_fingerprint,
            "selector_fingerprint": selector_fingerprint,
        },
        "source_score_cache": {
            "path": _repo_relative(args.source_score_cache),
            "sha256": score_cache_sha256,
            "score_spec_fingerprint": expected_score_spec["score_spec_fingerprint"],
            "score_formula_version": expected_score_spec["score_formula_version"],
            "dtype": "float32",
            "validated_original_relation": "scores = D_bd * (1 - D_all)",
            "selection_reads": ["boundary_disagreement"],
            "ranking_uses_global_disagreement": False,
            "ranking_uses_scores": False,
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
            "within_cluster_order": "(-boundary_disagreement, sample_key)",
            "center_distance_after_seed": False,
            "dedup": True,
            "dedup_scope": "same_cluster_all_previously_selected",
            "dedup_threshold": args.dedup_threshold,
            "dedup_comparison": "cosine_similarity > threshold",
            "dedup_equal_threshold_is_allowed": True,
            "backfill_policy": [
                "continue_within_cluster",
                "global_boundary_rank_keep_incluster_dedup",
                "global_boundary_rank_relax_dedup",
            ],
            "target_counts": list(targets),
            "first_added": first_added,
            "cluster_sizes": {
                str(cluster_id): cluster_sizes[cluster_id]
                for cluster_id in sorted(cluster_sizes)
            },
            "rounds": [
                {
                    **asdict(record),
                    "quotas": {
                        str(cluster_id): quota
                        for cluster_id, quota in record.quotas.items()
                    },
                }
                for record in selection_result.rounds
            ],
            "unique_dedup_skipped_samples": unique_skipped,
            "total_dedup_skip_events": total_skip_events,
            "uses_boundary_disagreement": True,
            "uses_global_disagreement": False,
            "uses_pc_bacs_value": False,
            "uses_gt": False,
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
    manifest_path = args.output_dir / f"{ARTIFACT_PREFIX}_manifest.json"
    text_payloads[manifest_path] = _encode_json(manifest)

    _stage_and_publish(torch_payloads, text_payloads)
    print(json.dumps(dry_report, indent=2, sort_keys=True))
    print(f"Saved KMeans-D_bd deduplicated artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
