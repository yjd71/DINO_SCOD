from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from select_global_additive import (
    DINO_WEIGHT_PATH,
    PREPROCESS_SPEC,
    SELECTOR_EPOCHS,
    SOURCE_SCORE_FORMULA_VERSION,
    _canonicalize_split_values,
    _catalog_fingerprint,
    _encode_json,
    _expected_source_spec,
    _load_selector_identity,
    _load_split_source,
    _package_version,
    _repo_commit,
    _repo_relative,
    _score_quantiles,
    _sample_key_order_fingerprint,
    _sha256_file,
    _sha256_json,
    _stage_and_publish,
    _validate_source_score_cache,
)
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
)
from utils.dataloader import SelectionPoolDataset
from utils.global_multiplicative import (
    GLOBAL_MULTIPLICATIVE_FORMULA_VERSION,
    GLOBAL_MULTIPLICATIVE_PROTOCOL_VERSION,
    build_global_multiplicative_splits,
    compute_global_multiplicative_score,
    labeled_names_from_keys,
)


REPO_ROOT = Path(__file__).resolve().parent
EXPECTED_CATALOG_COUNT = 4040
EXPECTED_SEED_COUNT = 40
EXPECTED_KMEANS_SEED_FINGERPRINT = (
    "7a5064395bf40d0a1be3826c7166a49566c8d4cfe9c81c02b613699de9bb75bb"
)
FORMAL_TARGET_COUNTS = (41, 202, 404, 808)
MANIFEST_VERSION = 1
ARTIFACT_PREFIX = "global_multiplicative"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact nested labeled splits by the global multiplicative "
            "score D_bd * (1 - D_all), without running KMeans or deduplication."
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
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("--target-counts must contain positive integers.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("--target-counts must be strictly increasing.")
    if targets[0] <= EXPECTED_SEED_COUNT:
        raise ValueError(
            "The smallest target must add at least one sample to the 40-image seed."
        )
    return targets


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "utils" / "global_multiplicative.py",
        REPO_ROOT / "select_global_additive.py",
    ):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _encode_csv(
    *,
    sample_keys: Sequence[str],
    boundary: torch.Tensor,
    global_disagreement: torch.Tensor,
    scores: torch.Tensor,
    seed_keys: set[str],
    selection_result: Any,
    target_counts: Sequence[int],
) -> bytes:
    fieldnames = [
        "sample_key",
        "labeled_name",
        "boundary_disagreement",
        "global_disagreement",
        "global_multiplicative_score",
        "is_seed",
        "global_candidate_rank",
    ]
    for target in target_counts:
        fieldnames.extend(
            (f"selected_{target:04d}", f"selection_rank_{target:04d}")
        )

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for index, key in enumerate(sample_keys):
        row: dict[str, Any] = {
            "sample_key": key,
            "labeled_name": key.rsplit("/", 1)[-1],
            "boundary_disagreement": f"{float(boundary[index]):.10f}",
            "global_disagreement": f"{float(global_disagreement[index]):.10f}",
            "global_multiplicative_score": f"{float(scores[index]):.10f}",
            "is_seed": int(key in seed_keys),
            "global_candidate_rank": selection_result.global_rank.get(key, ""),
        }
        for target in target_counts:
            rank = selection_result.selection_rank[int(target)].get(key)
            row[f"selected_{target:04d}"] = int(rank is not None)
            row[f"selection_rank_{target:04d}"] = "" if rank is None else rank
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
    if len(sample_keys) != EXPECTED_CATALOG_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CATALOG_COUNT} catalog samples, got {len(sample_keys)}."
        )
    if targets[-1] > len(sample_keys):
        raise ValueError("Largest target exceeds the active catalog size.")
    catalog_fingerprint = _catalog_fingerprint(items)

    seed_keys = _canonicalize_split_values(
        _load_split_source(args.seed_split), sample_keys
    )
    if len(seed_keys) != EXPECTED_SEED_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_SEED_COUNT} seed samples, got {len(seed_keys)}."
        )
    seed_fingerprint = compute_labeled_split_fingerprint(seed_keys)
    if seed_fingerprint != EXPECTED_KMEANS_SEED_FINGERPRINT:
        raise ValueError(
            "Seed split is not the fixed KMeans-center 40-seed: expected "
            f"{EXPECTED_KMEANS_SEED_FINGERPRINT}, got {seed_fingerprint}."
        )

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
    scores = compute_global_multiplicative_score(boundary, global_disagreement)
    source_scores = source_payload["scores"].detach().cpu().contiguous()
    if not torch.equal(scores, source_scores):
        raise ValueError(
            "Source score cache mismatch: scores are not the exact float32 "
            "D_bd * (1 - D_all) values."
        )

    selection_result = build_global_multiplicative_splits(
        sample_keys,
        scores,
        seed_keys,
        target_counts=targets,
    )
    split_fingerprints = {
        int(target): _sample_key_order_fingerprint(
            selection_result.splits[int(target)]
        )
        for target in targets
    }
    first_added = selection_result.selection_order[len(seed_keys)]
    source_cache_sha256 = _sha256_file(args.source_score_cache)
    dry_report = {
        "catalog_count": len(sample_keys),
        "catalog_fingerprint": catalog_fingerprint,
        "seed_count": len(seed_keys),
        "seed_fingerprint": seed_fingerprint,
        "seed_origin": "kmeans_center_40",
        "selector_fingerprint": selector_fingerprint,
        "source_score_cache": _repo_relative(args.source_score_cache),
        "source_score_cache_sha256": source_cache_sha256,
        "score_formula": "D_bd * (1 - D_all)",
        "score_formula_version": GLOBAL_MULTIPLICATIVE_FORMULA_VERSION,
        "uses_kmeans_during_selection": False,
        "dedup": False,
        "first_added": first_added,
        "targets": {str(key): value for key, value in split_fingerprints.items()},
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0

    torch_payloads: dict[Path, Any] = {}
    text_payloads: dict[Path, bytes] = {}
    outputs: dict[str, Any] = {}
    for target in targets:
        keys = selection_result.splits[int(target)]
        names = labeled_names_from_keys(keys)
        pt_path = args.output_dir / f"{ARTIFACT_PREFIX}_{target:04d}_keys.pt"
        txt_path = (
            args.output_dir
            / f"{ARTIFACT_PREFIX}_{target:04d}_labeled_names.txt"
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

    csv_path = args.output_dir / f"{ARTIFACT_PREFIX}_scores.csv"
    csv_payload = _encode_csv(
        sample_keys=sample_keys,
        boundary=boundary,
        global_disagreement=global_disagreement,
        scores=scores,
        seed_keys=set(seed_keys),
        selection_result=selection_result,
        target_counts=targets,
    )
    text_payloads[csv_path] = csv_payload

    manifest = {
        "format_version": MANIFEST_VERSION,
        "method": "Global-Multiplicative-BACS",
        "protocol": GLOBAL_MULTIPLICATIVE_PROTOCOL_VERSION,
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
            "training_seed_fingerprint": seed_fingerprint,
            "seed_origin": "kmeans_center_40",
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
            "component_dtype": "float32",
            "exact_source_scores_match": True,
        },
        "selection": {
            "strategy": "global_topk",
            "score_formula": "D_bd * (1 - D_all)",
            "score_formula_version": GLOBAL_MULTIPLICATIVE_FORMULA_VERSION,
            "score_range": [0.0, 1.0],
            "score_quantiles": _score_quantiles(scores),
            "ranking_tie_break": "(-score, sample_key)",
            "uses_kmeans_during_selection": False,
            "seed_origin": "kmeans_center_40",
            "dedup": False,
            "target_counts": list(targets),
            "first_added": first_added,
            "split_fingerprint_scheme": "sha256_json_ordered_sample_keys_v1",
        },
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
        "preprocessing": {
            **PREPROCESS_SPEC,
            "fingerprint": _sha256_json(PREPROCESS_SPEC),
        },
    }
    manifest_path = args.output_dir / f"{ARTIFACT_PREFIX}_manifest.json"
    text_payloads[manifest_path] = _encode_json(manifest)

    _stage_and_publish(torch_payloads, text_payloads)
    print(json.dumps(dry_report, indent=2, sort_keys=True))
    print(f"Saved global-multiplicative artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
