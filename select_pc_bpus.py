"""Score the RGB training pool and build nested PC-BPUS splits.

This entry point is intentionally independent of the KMeans/RDSM pipeline.  It
loads a legacy decoder trained on the common bootstrap, evaluates original and
horizontally flipped RGB views, and performs deterministic CPU-FP32 greedy
acquisition from boundary prototypes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model.base_model import BaseModel
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
from selection.pc_bpus import (
    BOUNDARY_SCORE_VERSION,
    PROTOTYPE_VERSION,
    build_prototype_payload,
    build_score_payload,
    greedy_acquire,
    score_and_prototype,
    validate_prototype_payload,
    validate_score_payload,
)
from selection.protocol import SamplingProtocol
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    state_dict_fingerprint,
)
from utils.dataloader import SelectionPoolDataset


REPO_ROOT = Path(__file__).resolve().parent
DINO_WEIGHT_PATH = REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth"
FORMAL_CATALOG_SIZE = 4040
EXPECTED_P2_DIM = 128


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Dataset/COD", type=Path)
    parser.add_argument(
        "--train-sets", nargs="+", default=["TR-CAMO", "TR-COD10K"]
    )
    parser.add_argument(
        "--target-counts",
        nargs=3,
        required=True,
        type=_positive_int,
        metavar=("SMALL", "MEDIUM", "LARGE"),
    )
    parser.add_argument(
        "--debug-custom-counts",
        action="store_true",
        help="Allow a non-formal budget triple; output must be under a debug directory.",
    )
    parser.add_argument("--bootstrap-split", required=True, type=Path)
    parser.add_argument("--selector-checkpoint", required=True, type=Path)
    parser.add_argument("--selector-config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", default=8, type=_positive_int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--eps", default=1e-6, type=_positive_float)
    parser.add_argument("--boundary-mass-eps", default=1e-6, type=_positive_float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Validate and reuse both existing cache files without forwarding the model.",
    )
    cache.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Atomically replace both cache files after a complete rescore.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate identity, cache, acquisition and saved splits without writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate catalog, bootstrap and Selector identities only.",
    )
    return parser.parse_args(argv)


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_paths(args: argparse.Namespace) -> None:
    args.data_root = _resolve(REPO_ROOT, args.data_root)
    args.bootstrap_split = _resolve(REPO_ROOT, args.bootstrap_split)
    args.selector_checkpoint = _resolve(REPO_ROOT, args.selector_checkpoint)
    if args.selector_config is None:
        args.selector_config = args.selector_checkpoint.with_name("selector_config.json")
    else:
        args.selector_config = _resolve(REPO_ROOT, args.selector_config)
    args.output_dir = _resolve(REPO_ROOT, args.output_dir)


def _image_roots(args: argparse.Namespace) -> list[str]:
    return [str(args.data_root / name / "im") for name in args.train_sets]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False


def _load_selector_identity(
    args: argparse.Namespace,
    *,
    protocol: SamplingProtocol,
    split_fingerprint: str,
    dino_fingerprint: str,
    catalog_fingerprint: str,
) -> tuple[dict[str, torch.Tensor], str, Mapping[str, Any]]:
    for path, label in (
        (args.selector_checkpoint, "Selector checkpoint"),
        (args.selector_config, "Selector config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    with args.selector_config.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    expected = {
        "kind": "pc_bpus_selector",
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "seed": int(args.seed),
        "split_fingerprint": split_fingerprint,
        "dino_weight_fingerprint": dino_fingerprint,
        "catalog_fingerprint": catalog_fingerprint,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"Selector config identity mismatch for {field}: "
                f"expected {value!r}, found {config.get(field)!r}."
            )
    state = _torch_load(args.selector_checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("selector_raw.pth must contain a non-empty state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("selector_raw.pth is not a raw tensor state_dict")
    if any(key.startswith("pc_hbm.") for key in state):
        raise ValueError("PC-BPUS Selector state unexpectedly contains PC-HBM parameters")
    fingerprint = state_dict_fingerprint(state)
    if config.get("selector_fingerprint") != fingerprint:
        raise ValueError("Selector checkpoint fingerprint disagrees with selector_config.json")
    return state, fingerprint, config


def _preprocessing_fingerprint(args: argparse.Namespace) -> str:
    return stable_fingerprint(
        {
            "schema": "pc_bpus_preprocessing_v1",
            "image_size": [392, 392],
            "resize": "bilinear_antialias",
            "rgb": True,
            "to_tensor_scale": "uint8_div_255",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "views": ["original", "horizontal_flip"],
            "view_alignment": "flip_back_then_mean",
            "eps": float(args.eps),
            "boundary_mass_eps": float(args.boundary_mass_eps),
            "boundary_score_version": BOUNDARY_SCORE_VERSION,
            "prototype_version": PROTOTYPE_VERSION,
        }
    )


def _cache_validation_kwargs(
    *,
    sample_keys: Sequence[str],
    catalog_fingerprint: str,
    image_fingerprint: str,
    dino_fingerprint: str,
    selector_fingerprint: str,
    preprocessing_fingerprint: str,
) -> dict[str, Any]:
    return {
        "expected_sample_keys": sample_keys,
        "expected_catalog_fingerprint": catalog_fingerprint,
        "expected_image_fingerprint": image_fingerprint,
        "expected_dino_fingerprint": dino_fingerprint,
        "expected_selector_fingerprint": selector_fingerprint,
        "expected_preprocessing_fingerprint": preprocessing_fingerprint,
    }


def _score_pool(
    args: argparse.Namespace,
    *,
    pool: SelectionPoolDataset,
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device(args.device)
    model = BaseModel(pc_cfg=None)
    model.decoder.load_state_dict(state, strict=True)
    model.dino.requires_grad_(False)
    model.dino.eval()
    if getattr(model.decoder, "pc_hbm", None) is not None:
        raise RuntimeError("PC-BPUS must instantiate BaseModel(pc_cfg=None)")
    model.to(device).eval()
    loader = DataLoader(
        pool,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    observed_keys: list[str] = []
    fields: dict[str, list[torch.Tensor]] = {
        "d_boundary": [],
        "d_global": [],
        "value": [],
        "boundary_mass": [],
        "valid_boundary": [],
        "prototypes": [],
    }
    with torch.inference_mode():
        for keys, images in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            result = score_and_prototype(
                model,
                images,
                eps=args.eps,
                boundary_mass_eps=args.boundary_mass_eps,
                use_amp=bool(args.amp and device.type == "cuda"),
                expected_p2_shape=(EXPECTED_P2_DIM, 28, 28),
            )
            observed_keys.extend(list(keys))
            fields["d_boundary"].append(result.d_boundary.detach().float().cpu())
            fields["d_global"].append(result.d_global.detach().float().cpu())
            fields["value"].append(result.value.detach().float().cpu())
            fields["boundary_mass"].append(result.boundary_mass.detach().float().cpu())
            fields["valid_boundary"].append(result.valid.detach().bool().cpu())
            fields["prototypes"].append(result.prototype.detach().float().cpu())
    if observed_keys != list(pool.sample_keys):
        raise RuntimeError("DataLoader sample-key order drifted from the catalog")
    merged = {name: torch.cat(chunks, dim=0) for name, chunks in fields.items()}
    if merged["prototypes"].shape != (len(pool), EXPECTED_P2_DIM):
        raise RuntimeError(
            "Expected boundary P2 prototypes with shape "
            f"[{len(pool)},{EXPECTED_P2_DIM}], found {tuple(merged['prototypes'].shape)}."
        )
    score_data = {key: merged[key] for key in (
        "d_boundary", "d_global", "value", "boundary_mass", "valid_boundary"
    )}
    prototype_data = {
        "prototypes": merged["prototypes"],
        "valid_boundary": merged["valid_boundary"],
    }
    return score_data, prototype_data


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _diagnostics(
    scores: Mapping[str, torch.Tensor], prototypes: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    valid = scores["valid_boundary"].bool()
    proto = prototypes["prototypes"].float()
    norms = torch.linalg.vector_norm(proto, dim=1)
    valid_norms = norms[valid]
    warnings: list[str] = []
    valid_ratio = float(valid.float().mean())
    if valid_ratio < 0.25:
        warnings.append("valid_boundary_ratio_below_0.25")
    if int(valid.sum()) > 1:
        indices = torch.nonzero(valid, as_tuple=False).flatten()[:512]
        cosine = proto[indices] @ proto[indices].T
        tri = torch.triu_indices(len(indices), len(indices), offset=1)
        pairwise = cosine[tri[0], tri[1]]
        pairwise_mean = float(pairwise.mean()) if pairwise.numel() else 0.0
    else:
        pairwise_mean = 0.0
    if pairwise_mean > 0.98:
        warnings.append("prototype_pairwise_cosine_above_0.98")
    return {
        "valid_boundary_count": int(valid.sum()),
        "valid_boundary_ratio": valid_ratio,
        "d_boundary": _tensor_stats(scores["d_boundary"]),
        "d_global": _tensor_stats(scores["d_global"]),
        "value": _tensor_stats(scores["value"]),
        "boundary_mass": _tensor_stats(scores["boundary_mass"]),
        "valid_prototype_norm": (
            _tensor_stats(valid_norms) if valid_norms.numel() else None
        ),
        "prototype_pairwise_cosine_mean_first_512": pairwise_mean,
        "warnings": warnings,
    }


def _verify_saved_splits(
    output_dir: Path,
    *,
    seed: int,
    counts: Sequence[int],
    catalog_keys: Sequence[str],
    bootstrap_keys: Sequence[str],
) -> tuple[dict[int, list[str]], dict[str, str]]:
    loaded: dict[int, list[str]] = {}
    fingerprints: dict[str, str] = {}
    for count in counts:
        path = output_dir / f"pc_bpus_{count:04d}_seed{seed}.pt"
        keys = load_split_keys(path, catalog_keys=catalog_keys, expected_count=count)
        loaded[int(count)] = keys
        fingerprints[str(count)] = compute_labeled_split_fingerprint(keys)
    if loaded[int(counts[0])] != list(bootstrap_keys):
        raise RuntimeError("The smallest PC-BPUS split is not exactly the common bootstrap")
    for smaller, larger in zip(counts, counts[1:]):
        if not set(loaded[int(smaller)]) < set(loaded[int(larger)]):
            raise RuntimeError(f"Strict nesting failed for {smaller} subset {larger}")
    return loaded, fingerprints


def _with_fingerprint(
    payload: Mapping[str, Any], *, fingerprint_field: str
) -> dict[str, Any]:
    result = dict(payload)
    if fingerprint_field in result:
        raise ValueError(f"reserved fingerprint field already present: {fingerprint_field}")
    result[fingerprint_field] = stable_fingerprint(result)
    return result


def _validate_fingerprinted_payload(
    saved: Any,
    expected: Mapping[str, Any],
    *,
    fingerprint_field: str,
    label: str,
) -> None:
    if not isinstance(saved, Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected_fingerprint = stable_fingerprint(expected)
    if saved.get(fingerprint_field) != expected_fingerprint:
        raise ValueError(f"{label} fingerprint disagrees with deterministic replay")
    saved_without_fingerprint = {
        key: value for key, value in saved.items() if key != fingerprint_field
    }
    if set(saved_without_fingerprint) != set(expected):
        raise ValueError(f"{label} fields disagree with the expected schema")
    if stable_fingerprint(saved_without_fingerprint) != expected_fingerprint:
        raise ValueError(f"{label} contents disagree with deterministic replay")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.verify_only and args.rebuild_cache:
        raise ValueError("--verify-only cannot be combined with --rebuild-cache")
    protocol = SamplingProtocol.from_counts(
        args.target_counts, allow_custom=bool(args.debug_custom_counts)
    )
    if not protocol.is_formal and "debug" not in str(args.output_dir).lower():
        raise ValueError("Custom counts require an isolated output directory containing 'debug'")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if not DINO_WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"Missing DINOv2 weight: {DINO_WEIGHT_PATH}")
    if not args.bootstrap_split.is_file():
        raise FileNotFoundError(f"Missing common bootstrap: {args.bootstrap_split}")

    os.chdir(REPO_ROOT)
    _set_seed(args.seed, bool(args.deterministic))
    pool = SelectionPoolDataset(_image_roots(args), image_size=392)
    sample_keys = list(pool.sample_keys)
    if sample_keys != sorted(sample_keys) or len(sample_keys) != len(set(sample_keys)):
        raise RuntimeError("Selection catalog must be sorted and unique")
    if protocol.is_formal and len(sample_keys) != FORMAL_CATALOG_SIZE:
        raise RuntimeError(
            f"Formal selection requires {FORMAL_CATALOG_SIZE} images, found {len(sample_keys)}"
        )
    bootstrap_keys = load_split_keys(
        args.bootstrap_split,
        catalog_keys=sample_keys,
        expected_count=protocol.bootstrap_count,
    )
    split_fingerprint = compute_labeled_split_fingerprint(bootstrap_keys)
    catalog_fingerprint = compute_catalog_fingerprint(sample_keys)
    key_order_fingerprint = compute_key_order_fingerprint(sample_keys)
    image_paths = [item["image"] for item in pool.items]
    image_fingerprint = compute_image_fingerprint(sample_keys, image_paths)
    dino_fingerprint = file_sha256(DINO_WEIGHT_PATH)
    preprocessing_fingerprint = _preprocessing_fingerprint(args)
    state, selector_fingerprint, selector_config = _load_selector_identity(
        args,
        protocol=protocol,
        split_fingerprint=split_fingerprint,
        dino_fingerprint=dino_fingerprint,
        catalog_fingerprint=catalog_fingerprint,
    )
    print(
        f"PC-BPUS protocol={protocol.name} seed={args.seed} catalog={len(sample_keys)} "
        f"bootstrap={len(bootstrap_keys)} selector={selector_fingerprint[:12]}"
    )
    if args.dry_run:
        print("Dry run passed; no files were written.")
        return 0

    scores_path = args.output_dir / "pool_scores.pt"
    prototypes_path = args.output_dir / "pool_prototypes.pt"
    existing = (scores_path.exists(), prototypes_path.exists())
    if any(existing) and not all(existing):
        raise RuntimeError("PC-BPUS cache is incomplete; use --rebuild-cache")
    if all(existing) and not (args.reuse_cache or args.rebuild_cache or args.verify_only):
        raise FileExistsError(
            "PC-BPUS cache already exists; pass --reuse-cache to validate it or "
            "--rebuild-cache to replace it atomically."
        )
    if args.reuse_cache and not all(existing):
        raise FileNotFoundError("--reuse-cache requires both pool cache files")
    if args.verify_only and not all(existing):
        raise FileNotFoundError("--verify-only requires both pool cache files")

    identity_kwargs = _cache_validation_kwargs(
        sample_keys=sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_fingerprint=dino_fingerprint,
        selector_fingerprint=selector_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
    )
    if (args.reuse_cache or args.verify_only) and all(existing):
        score_payload = _torch_load(scores_path)
        prototype_payload = _torch_load(prototypes_path)
    else:
        score_data, prototype_data = _score_pool(args, pool=pool, state=state)
        score_payload = build_score_payload(
            sample_keys,
            score_data["d_boundary"],
            score_data["d_global"],
            score_data["value"],
            score_data["boundary_mass"],
            score_data["valid_boundary"],
            catalog_fingerprint=catalog_fingerprint,
            image_fingerprint=image_fingerprint,
            dino_fingerprint=dino_fingerprint,
            selector_fingerprint=selector_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            boundary_mass_eps=args.boundary_mass_eps,
        )
        prototype_payload = build_prototype_payload(
            sample_keys,
            prototype_data["prototypes"],
            prototype_data["valid_boundary"],
            catalog_fingerprint=catalog_fingerprint,
            image_fingerprint=image_fingerprint,
            dino_fingerprint=dino_fingerprint,
            selector_fingerprint=selector_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        replace = bool(args.rebuild_cache)
        atomic_torch_save(score_payload, scores_path, refuse_mismatch=not replace)
        atomic_torch_save(prototype_payload, prototypes_path, refuse_mismatch=not replace)

    score_data = validate_score_payload(
        score_payload,
        **identity_kwargs,
        expected_boundary_mass_eps=args.boundary_mass_eps,
    )
    prototype_data = validate_prototype_payload(
        prototype_payload,
        **identity_kwargs,
        expected_feature_dim=EXPECTED_P2_DIM,
        expected_valid_boundary=score_data["valid_boundary"],
    )
    result = greedy_acquire(
        sample_keys,
        prototype_data["prototypes"],
        score_data["value"],
        score_data["valid_boundary"],
        bootstrap_keys,
        protocol.target_counts,
    )
    if len(result.acquired_keys) != protocol.target_counts[-1] - protocol.bootstrap_count:
        raise RuntimeError("Acquisition length disagrees with the target protocol")

    acquisition_path = args.output_dir / "acquisition_order.pt"
    manifest_path = args.output_dir / "selection_manifest.json"
    score_cache_fingerprint = stable_fingerprint(score_payload)
    prototype_cache_fingerprint = stable_fingerprint(prototype_payload)
    acquisition_base = {
        "schema_version": 1,
        "method": "pc_bpus",
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "seed": int(args.seed),
        "bootstrap_keys": list(bootstrap_keys),
        "acquired_keys": list(result.acquired_keys),
        "utility": result.utility,
        "value": result.value,
        "novelty": result.novelty,
        "max_similarity": result.max_similarity,
        "catalog_fingerprint": catalog_fingerprint,
        "key_order_fingerprint": key_order_fingerprint,
        "image_fingerprint": image_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "bootstrap_fingerprint": split_fingerprint,
        "score_cache_fingerprint": score_cache_fingerprint,
        "prototype_cache_fingerprint": prototype_cache_fingerprint,
    }
    acquisition_payload = _with_fingerprint(
        acquisition_base, fingerprint_field="acquisition_fingerprint"
    )
    if not args.verify_only:
        atomic_torch_save(acquisition_payload, acquisition_path)
        for count in protocol.target_counts:
            save_split_keys(
                args.output_dir / f"pc_bpus_{count:04d}_seed{args.seed}.pt",
                result.splits[count],
            )
    _, split_fingerprints = _verify_saved_splits(
        args.output_dir,
        seed=args.seed,
        counts=protocol.target_counts,
        catalog_keys=sample_keys,
        bootstrap_keys=bootstrap_keys,
    )
    diagnostics = _diagnostics(score_data, prototype_data)
    manifest_base = {
        "schema_version": 1,
        "method": "pc_bpus",
        "kmeans_dependency": False,
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "seed": int(args.seed),
        "catalog_size": len(sample_keys),
        "catalog_fingerprint": catalog_fingerprint,
        "key_order_fingerprint": key_order_fingerprint,
        "image_fingerprint": image_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "selector_config_fingerprint": stable_fingerprint(selector_config),
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "score_cache_fingerprint": score_cache_fingerprint,
        "prototype_cache_fingerprint": prototype_cache_fingerprint,
        "acquisition_fingerprint": acquisition_payload["acquisition_fingerprint"],
        "bootstrap_path": str(args.bootstrap_split),
        "bootstrap_fingerprint": split_fingerprint,
        "split_fingerprints": split_fingerprints,
        "acquired_count": len(result.acquired_keys),
        "score_formula": "relu(D_bd-D_all)*(1-D_all)",
        "novelty_formula": "1-clamp(max_cosine,0,1)",
        "utility_formula": "V*N",
        "tie_break": ["higher_Q", "higher_V", "higher_N", "smaller_sample_key"],
        "diagnostics": diagnostics,
        "files": {
            "pool_scores": scores_path.name,
            "pool_prototypes": prototypes_path.name,
            "acquisition_order": acquisition_path.name,
            "splits": {
                str(count): f"pc_bpus_{count:04d}_seed{args.seed}.pt"
                for count in protocol.target_counts
            },
        },
    }
    manifest = _with_fingerprint(
        manifest_base, fingerprint_field="manifest_fingerprint"
    )
    if args.verify_only:
        acquisition = _torch_load(acquisition_path)
        _validate_fingerprinted_payload(
            acquisition,
            acquisition_base,
            fingerprint_field="acquisition_fingerprint",
            label="acquisition_order.pt",
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing selection manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as stream:
            saved_manifest = json.load(stream)
        _validate_fingerprinted_payload(
            saved_manifest,
            manifest_base,
            fingerprint_field="manifest_fingerprint",
            label="selection_manifest.json",
        )
        print("Verification passed; cache replay and all nested splits are identical.")
        return 0

    atomic_json_save(manifest, manifest_path)
    print(
        f"Saved strict nested splits {protocol.target_counts} to {args.output_dir}; "
        f"valid candidates={diagnostics['valid_boundary_count']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
