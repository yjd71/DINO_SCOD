"""Create the common deterministic, subset-stratified bootstrap split."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selection.artifacts import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    compute_catalog_fingerprint,
    load_split_keys,
    save_split_keys,
)
from selection.protocol import (  # noqa: E402
    SamplingProtocol,
    add_target_counts_argument,
    protocol_from_args,
)
from utils.checkpoint_pc_hbm import normalize_sample_key  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FORMAL_SUBSETS = ("TR-CAMO", "TR-COD10K")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the common stratified bootstrap for sampling comparisons."
    )
    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument("--train-sets", nargs="+", default=list(FORMAL_SUBSETS))
    add_target_counts_argument(parser)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--debug-allow-custom-counts",
        dest="allow_custom_counts",
        action="store_true",
        help="Allow a non-formal three-budget protocol for isolated smoke tests.",
    )
    parser.add_argument(
        "--subset-quotas",
        nargs="+",
        type=int,
        help="Debug-only quotas corresponding to --train-sets; sum must equal SMALL.",
    )
    return parser.parse_args(argv)


def collect_catalog(data_root: Path, train_sets: Sequence[str]) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    keys: list[str] = []
    for subset in train_sets:
        image_root = data_root / subset / "im"
        if not image_root.is_dir():
            raise FileNotFoundError(f"training image directory does not exist: {image_root}")
        for path in sorted(image_root.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                keys.append(normalize_sample_key(f"{subset}/{path.stem}"))
    keys.sort()
    if not keys:
        raise ValueError("training catalog is empty")
    if len(keys) != len(set(keys)):
        raise ValueError("training catalog contains duplicate stable sample keys")
    return keys


def formal_subset_quotas(protocol: SamplingProtocol) -> dict[str, int]:
    if not protocol.is_formal:
        raise ValueError("custom protocols require explicit debug subset quotas")
    if protocol.bootstrap_count == 40:
        return {"TR-CAMO": 10, "TR-COD10K": 30}
    if protocol.bootstrap_count == 41:
        return {"TR-CAMO": 10, "TR-COD10K": 31}
    raise AssertionError(f"unhandled formal bootstrap count: {protocol.bootstrap_count}")


def build_stratified_bootstrap(
    catalog_keys: Sequence[str],
    *,
    seed: int,
    subset_quotas: Mapping[str, int],
) -> list[str]:
    normalized = [normalize_sample_key(key) for key in catalog_keys]
    if normalized != sorted(normalized):
        raise ValueError("catalog keys must be sorted before bootstrap sampling")
    if len(normalized) != len(set(normalized)):
        raise ValueError("catalog keys must be unique")
    if not subset_quotas:
        raise ValueError("subset quotas must not be empty")

    selected: list[str] = []
    for subset, raw_quota in subset_quotas.items():
        if isinstance(raw_quota, bool) or not isinstance(raw_quota, int):
            raise TypeError("subset quotas must contain integers")
        quota = int(raw_quota)
        if quota <= 0:
            raise ValueError("subset quotas must be positive")
        prefix = normalize_sample_key(subset) + "/"
        candidates = [key for key in normalized if key.startswith(prefix)]
        if len(candidates) < quota:
            raise ValueError(
                f"subset {subset} needs {quota} samples but catalog has {len(candidates)}"
            )
        subset_rng = random.Random(int(seed))
        selected.extend(subset_rng.sample(candidates, quota))
    selected.sort()
    if len(selected) != len(set(selected)):
        raise AssertionError("stratified bootstrap unexpectedly produced duplicate keys")
    return selected


def _resolve_quotas(
    args: argparse.Namespace, protocol: SamplingProtocol
) -> dict[str, int]:
    if protocol.is_formal:
        if tuple(args.train_sets) != FORMAL_SUBSETS:
            raise ValueError(
                "formal protocols require --train-sets TR-CAMO TR-COD10K in that order"
            )
        if args.subset_quotas is not None:
            raise ValueError("formal protocol quotas are fixed and cannot be overridden")
        return formal_subset_quotas(protocol)

    if args.subset_quotas is None:
        raise ValueError("custom debug protocols require explicit --subset-quotas")
    if len(args.subset_quotas) != len(args.train_sets):
        raise ValueError("--subset-quotas must have one value per --train-sets entry")
    quotas = dict(zip(args.train_sets, args.subset_quotas))
    if len(quotas) != len(args.train_sets):
        raise ValueError("--train-sets must not contain duplicates")
    if sum(quotas.values()) != protocol.bootstrap_count:
        raise ValueError("debug subset quotas must sum to the smallest target count")
    return quotas


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = protocol_from_args(args)
    data_root = args.data_root.resolve()
    quotas = _resolve_quotas(args, protocol)

    formal_root = (data_root / "splits").resolve()
    if args.output_dir is None:
        if not protocol.is_formal:
            raise ValueError("custom debug protocols require an explicit isolated --output-dir")
        output_dir = formal_root / "bootstrap" / protocol.name / f"seed{args.seed}"
    else:
        output_dir = args.output_dir.resolve()
    if not protocol.is_formal and _is_relative_to(output_dir, formal_root):
        raise ValueError("custom debug output must be outside the formal data-root/splits tree")

    catalog_keys = collect_catalog(data_root, args.train_sets)
    if len(catalog_keys) < protocol.target_counts[-1]:
        raise ValueError(
            f"catalog has {len(catalog_keys)} samples but the largest target is "
            f"{protocol.target_counts[-1]}"
        )
    split_keys = build_stratified_bootstrap(
        catalog_keys,
        seed=args.seed,
        subset_quotas=quotas,
    )
    if len(split_keys) != protocol.bootstrap_count:
        raise AssertionError("bootstrap size does not match the selected protocol")

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "catalog.pt"
    split_path = output_dir / (
        f"bootstrap_{protocol.bootstrap_count:04d}_seed{args.seed}.pt"
    )
    atomic_torch_save(catalog_keys, catalog_path, refuse_mismatch=True)
    split_fingerprint = save_split_keys(split_path, split_keys)
    reloaded = load_split_keys(
        split_path,
        catalog_keys=catalog_keys,
        expected_count=protocol.bootstrap_count,
    )
    if reloaded != split_keys:
        raise AssertionError("saved bootstrap split failed round-trip validation")

    catalog_fingerprint = compute_catalog_fingerprint(catalog_keys)
    manifest = {
        "schema": "sampling_bootstrap_v1",
        "protocol": {
            "name": protocol.name,
            "target_counts": list(protocol.target_counts),
            "bootstrap_count": protocol.bootstrap_count,
            "is_formal": protocol.is_formal,
        },
        "seed": int(args.seed),
        "train_sets": list(args.train_sets),
        "subset_quotas": dict(quotas),
        "catalog": {
            "path": catalog_path.name,
            "count": len(catalog_keys),
            "fingerprint": catalog_fingerprint,
            "subset_counts": {
                subset: sum(key.startswith(f"{subset}/") for key in catalog_keys)
                for subset in args.train_sets
            },
        },
        "split": {
            "path": split_path.name,
            "count": len(split_keys),
            "fingerprint": split_fingerprint,
        },
    }
    atomic_json_save(manifest, output_dir / "bootstrap_manifest.json")
    print(
        f"Saved {len(split_keys)} bootstrap keys to {split_path} "
        f"(fingerprint={split_fingerprint})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
