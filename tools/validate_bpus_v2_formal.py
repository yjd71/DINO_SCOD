"""Read-only acceptance checks for formal BPUS-v2 41/202/404 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selection.artifacts import load_split_keys, stable_fingerprint
from utils.checkpoint_pc_hbm import state_dict_fingerprint
from utils.dataloader import (
    PCLabeledTrainDataset,
    SelectionPoolDataset,
    UnlabeledPseudoTrainDataset,
)


FORMAL_COUNTS = (41, 202, 404)
FORMAL_SEEDS = (2025, 2026, 2027)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_fingerprint(payload: dict[str, Any], field: str, label: str) -> None:
    if field not in payload:
        raise ValueError(f"{label} is missing {field}")
    expected = payload[field]
    base = {key: value for key, value in payload.items() if key != field}
    if expected != stable_fingerprint(base):
        raise ValueError(f"{label} has an invalid {field}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset/COD"))
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("Dataset/COD/splits/bpus_v2/scout_0041_0202_0404"),
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        default=Path("Dataset/COD/splits/bootstrap/scout_0041_0202_0404"),
    )
    parser.add_argument(
        "--selector-root",
        type=Path,
        default=Path("results/bpus_v2/scout_0041_0202_0404"),
    )
    parser.add_argument("--target-counts", nargs=3, type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    counts = tuple(args.target_counts)
    seeds = tuple(args.seeds)
    if counts != FORMAL_COUNTS:
        raise ValueError("Formal BPUS-v2 acceptance requires --target-counts 41 202 404")
    if seeds != FORMAL_SEEDS:
        raise ValueError("Formal BPUS-v2 acceptance requires --seeds 2025 2026 2027")

    data_root = _resolve(args.data_root)
    split_root = _resolve(args.split_root)
    bootstrap_root = _resolve(args.bootstrap_root)
    selector_root = _resolve(args.selector_root)
    image_roots = [
        str(data_root / subset / "im") for subset in ("TR-CAMO", "TR-COD10K")
    ]
    gt_roots = [
        str(data_root / subset / "gt") for subset in ("TR-CAMO", "TR-COD10K")
    ]

    # Catalog validation is RGB-only. Ground-truth roots are used later solely
    # for the explicitly requested Base/TS loader compatibility check.
    pool = SelectionPoolDataset(image_roots, image_size=392)
    catalog = list(pool.sample_keys)
    if len(catalog) != 4040 or catalog != sorted(catalog):
        raise RuntimeError("Formal catalog must contain 4040 sorted sample keys")
    if len(catalog) != len(set(catalog)):
        raise RuntimeError("Formal catalog sample keys must be unique")
    catalog_set = set(catalog)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        output_dir = split_root / f"seed{seed}"
        bootstrap_path = (
            bootstrap_root / f"seed{seed}" / f"bootstrap_0041_seed{seed}.pt"
        )
        bootstrap = load_split_keys(
            bootstrap_path, catalog_keys=catalog, expected_count=counts[0]
        )
        quotas = {
            "TR-CAMO": sum(key.startswith("TR-CAMO/") for key in bootstrap),
            "TR-COD10K": sum(
                key.startswith("TR-COD10K/") for key in bootstrap
            ),
        }
        if quotas != {"TR-CAMO": 10, "TR-COD10K": 31}:
            raise RuntimeError(f"seed{seed} Bootstrap quota mismatch: {quotas}")

        splits: dict[int, list[str]] = {}
        loader_counts: dict[str, dict[str, int]] = {}
        for count in counts:
            path = output_dir / f"bpus_v2_{count:04d}_seed{seed}.pt"
            raw = _torch_load(path)
            if type(raw) is not list or not all(type(key) is str for key in raw):
                raise TypeError(f"{path} must contain a plain list[str]")
            if raw != sorted(raw) or len(raw) != len(set(raw)) or len(raw) != count:
                raise ValueError(f"{path} is not a sorted unique {count}-key split")
            if not set(raw) <= catalog_set:
                raise ValueError(f"{path} contains keys outside the catalog")
            splits[count] = raw

            base_labeled = PCLabeledTrainDataset(
                image_roots,
                gt_roots,
                None,
                392,
                labeled_indices_pt=str(path),
            )
            ts_labeled = PCLabeledTrainDataset(
                image_roots,
                gt_roots,
                None,
                392,
                labeled_indices_pt=str(path),
            )
            ts_unlabeled = UnlabeledPseudoTrainDataset(
                image_roots,
                None,
                392,
                labeled_indices_pt=str(path),
            )
            expected_unlabeled = len(catalog) - count
            actual = {
                "base_labeled": len(base_labeled),
                "ts_labeled": len(ts_labeled),
                "ts_unlabeled": len(ts_unlabeled),
            }
            expected = {
                "base_labeled": count,
                "ts_labeled": count,
                "ts_unlabeled": expected_unlabeled,
            }
            if actual != expected:
                raise RuntimeError(
                    f"seed{seed} count={count} loader mismatch: {actual} != {expected}"
                )
            loader_counts[str(count)] = actual

        if splits[counts[0]] != bootstrap:
            raise RuntimeError(f"seed{seed} 41 split differs from its common Bootstrap")
        if not set(splits[41]) < set(splits[202]) < set(splits[404]):
            raise RuntimeError(f"seed{seed} splits are not strictly nested")

        acquisition = _torch_load(output_dir / "acquisition_order.pt")
        if not isinstance(acquisition, dict):
            raise TypeError("acquisition_order.pt must contain a mapping")
        _validate_fingerprint(
            acquisition, "acquisition_fingerprint", "acquisition_order.pt"
        )
        if acquisition.get("schema_version") != 2 or acquisition.get("method") != "BPUS-v2":
            raise ValueError("acquisition_order.pt is not BPUS-v2 schema 2")
        acquired = acquisition.get("acquired_keys")
        if not isinstance(acquired, list) or len(acquired) != 363:
            raise ValueError("acquisition_order.pt must contain 363 acquired keys")
        if len(acquired) != len(set(acquired)):
            raise ValueError("acquired keys must be unique")
        for field in ("utility", "value", "novelty", "max_similarity"):
            tensor = acquisition.get(field)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (363,):
                raise ValueError(f"acquisition {field} must have shape [363]")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"acquisition {field} must be finite")
        cumulative = acquisition.get("cumulative_subset_counts")
        if not isinstance(cumulative, dict) or any(
            not isinstance(cumulative.get(subset), torch.Tensor)
            or tuple(cumulative[subset].shape) != (363,)
            for subset in ("TR-CAMO", "TR-COD10K")
        ):
            raise ValueError("acquisition cumulative subset counts are invalid")

        scores = _torch_load(output_dir / "pool_scores.pt")
        prototypes = _torch_load(output_dir / "pool_prototypes.pt")
        if scores.get("schema_version") != 2 or scores.get("method") != "BPUS-v2":
            raise ValueError("pool_scores.pt is not BPUS-v2 schema 2")
        if list(scores.get("sample_keys", ())) != catalog:
            raise ValueError("score cache sample-key order differs from the catalog")
        required_score_fields = {
            "boundary_disagreement",
            "global_disagreement",
            "boundary_value",
            "boundary_mass",
            "valid_boundary",
        }
        if not required_score_fields <= set(scores):
            raise ValueError("pool_scores.pt is missing schema-2 score fields")
        valid_count = int(scores["valid_boundary"].sum())
        if valid_count < counts[-1] - counts[0]:
            raise RuntimeError(f"seed{seed} has insufficient valid candidates")
        if (
            prototypes.get("schema_version") != 2
            or prototypes.get("prototype_level") != "p2"
            or prototypes.get("prototype_dim") != 128
            or tuple(prototypes["prototypes"].shape) != (4040, 128)
        ):
            raise ValueError("pool_prototypes.pt violates the schema-2 P2 contract")

        manifest = json.loads(
            (output_dir / "selection_manifest.json").read_text(encoding="utf-8")
        )
        runtime = json.loads(
            (output_dir / "runtime_report.json").read_text(encoding="utf-8")
        )
        _validate_fingerprint(
            manifest, "manifest_fingerprint", "selection_manifest.json"
        )
        if manifest.get("schema_version") != 2 or manifest.get("acquired_count") != 363:
            raise ValueError("selection manifest identity/count mismatch")
        if runtime.get("kind") != "bpus_v2_runtime" or "elapsed_seconds" not in runtime:
            raise ValueError("runtime_report.json is invalid")

        selector_dir = selector_root / f"seed{seed}" / "selector"
        config = json.loads(
            (selector_dir / "selector_config.json").read_text(encoding="utf-8")
        )
        state = _torch_load(selector_dir / "selector_raw.pth")
        if (
            config.get("schema_version") != 2
            or config.get("kind") != "bpus_v2_selector"
            or config.get("method") != "BPUS-v2"
            or config.get("epochs") != 30
            or config.get("target_counts") != list(counts)
            or config.get("selector_fingerprint") != state_dict_fingerprint(state)
        ):
            raise ValueError(f"seed{seed} Selector identity mismatch")

        rows.append(
            {
                "seed": seed,
                "selector_fingerprint": config["selector_fingerprint"],
                "manifest_fingerprint": manifest["manifest_fingerprint"],
                "valid_candidates": valid_count,
                "bootstrap_quota": quotas,
                "loader_counts": loader_counts,
            }
        )

    print(json.dumps({"catalog_size": len(catalog), "seeds": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
