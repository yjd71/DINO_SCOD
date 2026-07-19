"""Score the RGB pool and build strict nested BPUS-v2 splits.

The entry point owns a schema-2 cache and artifact namespace.  It accepts only
the 41/202/404 protocol in formal mode, starts from the common bootstrap, and
uses deterministic CPU-FP32 acquisition over boundary-weighted P2 prototypes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
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
from selection.bpus_v2 import (
    BPUS_V2_FORMULA_VERSION,
    BPUS_V2_PROTOTYPE_VERSION,
    build_prototype_payload,
    build_score_payload,
    greedy_acquire_bpus_v2,
    score_and_prototype_bpus_v2,
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
FORMAL_TARGET_COUNTS = (41, 202, 404)
FORMAL_CATALOG_SIZE = 4040
EXPECTED_P2_DIM = 128
SCHEMA_VERSION = 2
METHOD = "BPUS-v2"

_FORMULA_VARIANTS: dict[str, dict[str, str]] = {
    "v1": {
        "value_mode": "hard-gap",
        "reward_mode": "novelty-gate",
        "value_formula": "relu(D_bd-D_all)*(1-D_all)",
        "utility_formula": "V*N",
    },
    "v2-a": {
        "value_mode": "smooth-value",
        "reward_mode": "novelty-gate",
        "value_formula": "D_bd*(1-D_all)",
        "utility_formula": "V*N",
    },
    "v2-b": {
        "value_mode": "hard-gap",
        "reward_mode": "soft-reward",
        "value_formula": "relu(D_bd-D_all)*(1-D_all)",
        "utility_formula": "V*(1+N)",
    },
    "v2": {
        "value_mode": "smooth-value",
        "reward_mode": "soft-reward",
        "value_formula": "D_bd*(1-D_all)",
        "utility_formula": "V*(1+N)",
    },
}


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
        help="Allow non-formal counts or a diagnostic formula in a debug directory.",
    )
    parser.add_argument(
        "--formula-variant",
        choices=tuple(_FORMULA_VARIANTS),
        default="v2",
        help="Formal runs only accept v2; alternatives are diagnostic-only.",
    )
    parser.add_argument("--bootstrap-split", required=True, type=Path)
    parser.add_argument("--selector-checkpoint", required=True, type=Path)
    parser.add_argument("--selector-config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", default=16, type=_positive_int)
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
        help="Validate and reuse both schema-2 pool cache files.",
    )
    cache.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rescore the complete pool and atomically replace both cache files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Replay cache, acquisition, manifest and split validation without writes.",
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


def _assert_v2_cache_namespace(paths: Sequence[Path]) -> None:
    """Refuse to replace cache files owned by another schema or method."""

    for path in paths:
        if not path.is_file():
            continue
        payload = _torch_load(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Existing cache is not a mapping: {path}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Existing cache is not schema {SCHEMA_VERSION} and cannot be replaced: {path}"
            )
        if payload.get("method") != METHOD:
            raise ValueError(
                f"Existing cache belongs to another method and cannot be replaced: {path}"
            )


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


def _formula_spec(variant: str) -> dict[str, Any]:
    spec = dict(_FORMULA_VARIANTS[variant])
    formula_version = BPUS_V2_FORMULA_VERSION
    if variant != "v2":
        formula_version = (
            f"{BPUS_V2_FORMULA_VERSION}:{variant}:"
            f"{spec['value_mode']}:{spec['reward_mode']}"
        )
    spec.update(
        {
            "variant": variant,
            "formula_version": formula_version,
            "novelty_formula": "1-clamp(max_cosine,0,1)",
        }
    )
    return spec


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
        "schema_version": SCHEMA_VERSION,
        "kind": "bpus_v2_selector",
        "method": METHOD,
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "bootstrap_count": protocol.bootstrap_count,
        "seed": int(args.seed),
        "split_fingerprint": split_fingerprint,
        "bootstrap_fingerprint": split_fingerprint,
        "dino_weight_fingerprint": dino_fingerprint,
        "catalog_fingerprint": catalog_fingerprint,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"Selector config identity mismatch for {field}: "
                f"expected {value!r}, found {config.get(field)!r}."
            )
    if config.get("catalog_count") != FORMAL_CATALOG_SIZE and tuple(
        protocol.target_counts
    ) == FORMAL_TARGET_COUNTS:
        raise ValueError("Formal Selector config must bind the 4040-image catalog")
    if tuple(protocol.target_counts) == FORMAL_TARGET_COUNTS:
        if config.get("epochs") != 30:
            raise ValueError("Formal BPUS-v2 Selector must be trained for exactly 30 epochs")
        training_config = config.get("training_config")
        if not isinstance(training_config, Mapping) or training_config.get("epochs") != 30:
            raise ValueError("Selector training_config must record epochs=30")
    training_config = config.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("Selector config must contain training_config")
    if config.get("training_fingerprint") != stable_fingerprint(training_config):
        raise ValueError("Selector training fingerprint is invalid")

    state = _torch_load(args.selector_checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("selector_raw.pth must contain a non-empty state_dict")
    if not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("selector_raw.pth is not a raw tensor state_dict")
    if any(key.startswith("pc_hbm.") for key in state):
        raise ValueError("BPUS-v2 Selector state contains unrelated parameters")
    fingerprint = state_dict_fingerprint(state)
    if config.get("selector_fingerprint") != fingerprint:
        raise ValueError("Selector checkpoint fingerprint disagrees with its config")
    return state, fingerprint, config


def _preprocessing_fingerprint(
    args: argparse.Namespace, formula_spec: Mapping[str, Any]
) -> str:
    return stable_fingerprint(
        {
            "schema": "bpus_v2_preprocessing_v2",
            "image_size": [392, 392],
            "resize": "bilinear_antialias",
            "rgb": True,
            "to_tensor_scale": "uint8_div_255",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "views": ["original", "horizontal_flip"],
            "view_execution": "two_explicit_eval_inference_forwards",
            "view_alignment": "flip_back_then_fp32_mean",
            "sobel": {
                "kernel_scale": 8,
                "padding": "replicate",
                "magnitude": "hypot",
                "outer_border": "zero",
            },
            "p2": {
                "shape": [128, 28, 28],
                "boundary_resize": "bilinear",
                "align_corners": False,
                "local_normalization": "l2",
                "aggregate_normalization": "l2",
            },
            "eps": float(args.eps),
            "boundary_mass_eps": float(args.boundary_mass_eps),
            "formula": dict(formula_spec),
            "prototype_version": BPUS_V2_PROTOTYPE_VERSION,
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
    value_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device(args.device)
    model = BaseModel(pc_cfg=None)
    model.decoder.load_state_dict(state, strict=True)
    model.dino.requires_grad_(False)
    model.dino.eval()
    if getattr(model.decoder, "pc_hbm", None) is not None:
        raise RuntimeError("BPUS-v2 requires BaseModel(pc_cfg=None)")
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
        "boundary_disagreement": [],
        "global_disagreement": [],
        "boundary_value": [],
        "boundary_mass": [],
        "valid_boundary": [],
        "prototypes": [],
    }
    with torch.inference_mode():
        for keys, images in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            result = score_and_prototype_bpus_v2(
                model,
                images,
                eps=args.eps,
                boundary_mass_eps=args.boundary_mass_eps,
                use_amp=bool(args.amp and device.type == "cuda"),
                value_mode=value_mode,
            )
            observed_keys.extend(list(keys))
            fields["boundary_disagreement"].append(
                result.boundary_disagreement.detach().float().cpu()
            )
            fields["global_disagreement"].append(
                result.global_disagreement.detach().float().cpu()
            )
            fields["boundary_value"].append(
                result.boundary_value.detach().float().cpu()
            )
            fields["boundary_mass"].append(
                result.boundary_mass.detach().float().cpu()
            )
            fields["valid_boundary"].append(
                result.valid_boundary.detach().bool().cpu()
            )
            fields["prototypes"].append(result.prototype.detach().float().cpu())
    if observed_keys != list(pool.sample_keys):
        raise RuntimeError("DataLoader sample-key order drifted from the catalog")
    merged = {name: torch.cat(chunks, dim=0) for name, chunks in fields.items()}
    if merged["prototypes"].shape != (len(pool), EXPECTED_P2_DIM):
        raise RuntimeError(
            "Expected P2 prototypes with shape "
            f"[{len(pool)},{EXPECTED_P2_DIM}], found {tuple(merged['prototypes'].shape)}."
        )
    score_data = {
        key: merged[key]
        for key in (
            "boundary_disagreement",
            "global_disagreement",
            "boundary_value",
            "boundary_mass",
            "valid_boundary",
        )
    }
    prototype_data = {
        "prototypes": merged["prototypes"],
        "valid_boundary": merged["valid_boundary"],
    }
    return score_data, prototype_data


def _distribution_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    quantiles = torch.quantile(
        values, torch.tensor([0.50, 0.75, 0.90, 0.95], dtype=torch.float32)
    )
    return {
        "p50": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "max": float(values.max()),
    }


def _diagnostics(
    scores: Mapping[str, torch.Tensor],
    prototypes: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, Any]:
    valid = scores["valid_boundary"].bool().cpu()
    value = scores["boundary_value"].float().cpu()
    proto = prototypes["prototypes"].float().cpu()
    norms = torch.linalg.vector_norm(proto, dim=1)
    valid_norms = norms[valid]
    valid_count = int(valid.sum())
    valid_ratio = float(valid.float().mean())
    warnings: list[str] = []
    if valid_ratio < 0.25:
        warnings.append("valid_boundary_ratio_below_0.25")
    if valid_norms.numel() and float((valid_norms - 1.0).abs().max()) > 1e-4:
        warnings.append("valid_prototype_norm_outside_tolerance")

    subset_seed = int(seed) ^ 0x42505553
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    subset_size = min(512, valid_count)
    if subset_size > 1:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(subset_seed)
        order = torch.randperm(valid_count, generator=generator)[:subset_size]
        chosen = proto[valid_indices[order]]
        cosine = chosen @ chosen.T
        triangle = torch.triu_indices(subset_size, subset_size, offset=1)
        pairwise = cosine[triangle[0], triangle[1]]
        cosine_stats: dict[str, Any] = _distribution_stats(pairwise)
        cosine_stats["min"] = float(pairwise.min())
        cosine_stats["subset_size"] = subset_size
        cosine_stats["subset_seed"] = subset_seed
    else:
        cosine_stats = {
            "subset_size": subset_size,
            "subset_seed": subset_seed,
            "pair_count": 0,
        }
    if cosine_stats.get("mean", 0.0) > 0.98:
        warnings.append("prototype_pairwise_cosine_above_0.98")
    return {
        "boundary_disagreement": _distribution_stats(
            scores["boundary_disagreement"]
        ),
        "global_disagreement": _distribution_stats(scores["global_disagreement"]),
        "boundary_value": _distribution_stats(value),
        "boundary_mass": _distribution_stats(scores["boundary_mass"]),
        "boundary_value_zero_ratio": float((value == 0).float().mean()),
        "valid_boundary_count": valid_count,
        "valid_boundary_ratio": valid_ratio,
        "valid_prototype_norm": (
            _distribution_stats(valid_norms) if valid_norms.numel() else None
        ),
        "random_valid_prototype_cosine": cosine_stats,
        "warnings": warnings,
    }


def _verify_saved_splits(
    output_dir: Path,
    *,
    seed: int,
    counts: Sequence[int],
    catalog_keys: Sequence[str],
    bootstrap_keys: Sequence[str],
) -> tuple[dict[int, list[str]], dict[str, str], dict[str, str]]:
    loaded: dict[int, list[str]] = {}
    fingerprints: dict[str, str] = {}
    txt_fingerprints: dict[str, str] = {}
    for count in counts:
        path = output_dir / f"bpus_v2_{count:04d}_seed{seed}.pt"
        keys = load_split_keys(path, catalog_keys=catalog_keys, expected_count=count)
        txt_path = path.with_suffix(".txt")
        txt_keys = _load_split_text(txt_path, expected_keys=keys)
        if compute_labeled_split_fingerprint(txt_keys) != compute_labeled_split_fingerprint(
            keys
        ):
            raise RuntimeError(f"TXT split fingerprint disagrees with {path.name}")
        loaded[int(count)] = keys
        fingerprints[str(count)] = compute_labeled_split_fingerprint(keys)
        txt_fingerprints[str(count)] = file_sha256(txt_path)
    if loaded[int(counts[0])] != list(bootstrap_keys):
        raise RuntimeError("The smallest BPUS-v2 split is not the common bootstrap")
    for smaller, larger in zip(counts, counts[1:]):
        if not set(loaded[int(smaller)]) < set(loaded[int(larger)]):
            raise RuntimeError(f"Strict nesting failed for {smaller} subset {larger}")
    return loaded, fingerprints, txt_fingerprints


def _canonical_split_text(sample_keys: Sequence[str]) -> bytes:
    keys = list(sample_keys)
    if not keys:
        raise ValueError("TXT split must contain at least one sample key")
    if not all(type(key) is str for key in keys):
        raise TypeError("TXT split must contain only strings")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("TXT split keys must be sorted and unique")
    if any(
        not key
        or key != key.strip()
        or "\\" in key
        or "\n" in key
        or "\r" in key
        for key in keys
    ):
        raise ValueError("TXT split contains a non-canonical sample key")
    return ("\n".join(keys) + "\n").encode("utf-8")


def _save_split_text(path: Path, sample_keys: Sequence[str]) -> str:
    """Atomically save one canonical stable sample key per UTF-8 line."""

    payload = _canonical_split_text(sample_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return file_sha256(path)
        raise FileExistsError(f"refusing to overwrite different TXT split: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_bytes(payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return file_sha256(path)


def _load_split_text(path: Path, *, expected_keys: Sequence[str]) -> list[str]:
    """Reload a TXT split and require exact semantic parity with its PT peer."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing TXT split: {path}")
    expected = list(expected_keys)
    expected_payload = _canonical_split_text(expected)
    payload = path.read_bytes()
    if payload != expected_payload:
        try:
            actual = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"TXT split is not valid UTF-8: {path}") from error
        if actual != expected:
            raise ValueError(f"TXT split disagrees with its PT peer: {path}")
        raise ValueError(f"TXT split is not in canonical UTF-8/LF form: {path}")
    return expected


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


def _save_or_upgrade_manifest(
    manifest: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    path: Path,
) -> None:
    """Save the manifest, allowing only the deterministic TXT-field upgrade."""

    if not path.is_file():
        atomic_json_save(manifest, path)
        return
    with path.open("r", encoding="utf-8") as stream:
        saved = json.load(stream)
    if saved == manifest:
        return

    legacy_base = dict(manifest_base)
    legacy_base.pop("split_txt_fingerprints")
    legacy_files = dict(legacy_base["files"])
    legacy_files.pop("split_txt")
    legacy_base["files"] = legacy_files
    _validate_fingerprinted_payload(
        saved,
        legacy_base,
        fingerprint_field="manifest_fingerprint",
        label="legacy selection_manifest.json",
    )
    atomic_json_save(manifest, path, refuse_mismatch=False)


def _cumulative_subset_counts(
    bootstrap_keys: Sequence[str], acquired_keys: Sequence[str]
) -> dict[str, torch.Tensor]:
    camo = sum(key.startswith("TR-CAMO/") for key in bootstrap_keys)
    cod10k = sum(key.startswith("TR-COD10K/") for key in bootstrap_keys)
    camo_counts: list[int] = []
    cod10k_counts: list[int] = []
    for key in acquired_keys:
        camo += int(key.startswith("TR-CAMO/"))
        cod10k += int(key.startswith("TR-COD10K/"))
        camo_counts.append(camo)
        cod10k_counts.append(cod10k)
    return {
        "TR-CAMO": torch.tensor(camo_counts, dtype=torch.int64),
        "TR-COD10K": torch.tensor(cod10k_counts, dtype=torch.int64),
    }


def _runtime_payload(
    *,
    args: argparse.Namespace,
    protocol: SamplingProtocol,
    started_at: str,
    elapsed_seconds: float,
    score_seconds: float,
    acquisition_seconds: float,
    cache_mode: str,
) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu = None
    if cuda_available:
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "bpus_v2_runtime",
        "method": METHOD,
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "seed": int(args.seed),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "score_seconds": float(score_seconds),
        "acquisition_seconds": float(acquisition_seconds),
        "cache_mode": cache_mode,
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu": gpu,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args(argv)
    _resolve_paths(args)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.verify_only and args.rebuild_cache:
        raise ValueError("--verify-only cannot be combined with --rebuild-cache")
    counts = tuple(args.target_counts)
    if counts != FORMAL_TARGET_COUNTS and not args.debug_custom_counts:
        raise ValueError(
            "Formal BPUS-v2 selection requires --target-counts 41 202 404; "
            "other counts require --debug-custom-counts."
        )
    protocol = SamplingProtocol.from_counts(
        counts, allow_custom=bool(args.debug_custom_counts)
    )
    debug_run = counts != FORMAL_TARGET_COUNTS or args.formula_variant != "v2"
    if debug_run and not args.debug_custom_counts:
        raise ValueError("Diagnostic formula variants require --debug-custom-counts")
    if debug_run and "debug" not in str(args.output_dir).lower():
        raise ValueError("Debug runs require an isolated output directory containing 'debug'")
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
    if counts == FORMAL_TARGET_COUNTS and len(sample_keys) != FORMAL_CATALOG_SIZE:
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
    formula_spec = _formula_spec(args.formula_variant)
    formula_version = str(formula_spec["formula_version"])
    formula_configuration_fingerprint = stable_fingerprint(formula_spec)
    preprocessing_fingerprint = _preprocessing_fingerprint(args, formula_spec)
    state, selector_fingerprint, selector_config = _load_selector_identity(
        args,
        protocol=protocol,
        split_fingerprint=split_fingerprint,
        dino_fingerprint=dino_fingerprint,
        catalog_fingerprint=catalog_fingerprint,
    )
    print(
        f"BPUS-v2 protocol={protocol.name} seed={args.seed} catalog={len(sample_keys)} "
        f"bootstrap={len(bootstrap_keys)} selector={selector_fingerprint[:12]}"
    )
    if args.dry_run:
        print("Dry run passed; no files were written.")
        return 0

    scores_path = args.output_dir / "pool_scores.pt"
    prototypes_path = args.output_dir / "pool_prototypes.pt"
    existing = (scores_path.exists(), prototypes_path.exists())
    if any(existing) and not all(existing):
        raise RuntimeError("BPUS-v2 cache is incomplete; use --rebuild-cache")
    if all(existing):
        _assert_v2_cache_namespace((scores_path, prototypes_path))
    if all(existing) and not (args.reuse_cache or args.rebuild_cache or args.verify_only):
        raise FileExistsError(
            "BPUS-v2 cache already exists; pass --reuse-cache to validate it or "
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
    score_started = time.perf_counter()
    if (args.reuse_cache or args.verify_only) and all(existing):
        score_payload = _torch_load(scores_path)
        prototype_payload = _torch_load(prototypes_path)
        cache_mode = "verify" if args.verify_only else "reuse"
    else:
        score_data, prototype_data = _score_pool(
            args,
            pool=pool,
            state=state,
            value_mode=formula_spec["value_mode"],
        )
        score_payload = build_score_payload(
            sample_keys,
            score_data["boundary_disagreement"],
            score_data["global_disagreement"],
            score_data["boundary_value"],
            score_data["boundary_mass"],
            score_data["valid_boundary"],
            catalog_fingerprint=catalog_fingerprint,
            image_fingerprint=image_fingerprint,
            dino_fingerprint=dino_fingerprint,
            selector_fingerprint=selector_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            boundary_mass_eps=args.boundary_mass_eps,
            formula_version=formula_version,
            prototype_version=BPUS_V2_PROTOTYPE_VERSION,
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
            formula_version=formula_version,
            prototype_version=BPUS_V2_PROTOTYPE_VERSION,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        replace = bool(args.rebuild_cache)
        atomic_torch_save(score_payload, scores_path, refuse_mismatch=not replace)
        atomic_torch_save(prototype_payload, prototypes_path, refuse_mismatch=not replace)
        cache_mode = "rebuild" if replace else "build"
    score_seconds = time.perf_counter() - score_started

    score_data = validate_score_payload(
        score_payload,
        **identity_kwargs,
        expected_boundary_mass_eps=args.boundary_mass_eps,
        expected_formula_version=formula_version,
        expected_prototype_version=BPUS_V2_PROTOTYPE_VERSION,
    )
    prototype_data = validate_prototype_payload(
        prototype_payload,
        **identity_kwargs,
        expected_prototype_dim=EXPECTED_P2_DIM,
        expected_valid_boundary=score_data["valid_boundary"],
        expected_formula_version=formula_version,
        expected_prototype_version=BPUS_V2_PROTOTYPE_VERSION,
    )
    acquisition_started = time.perf_counter()
    result = greedy_acquire_bpus_v2(
        sample_keys,
        prototype_data["prototypes"],
        score_data["boundary_value"],
        score_data["valid_boundary"],
        bootstrap_keys,
        protocol.target_counts,
        reward_mode=formula_spec["reward_mode"],
    )
    acquisition_seconds = time.perf_counter() - acquisition_started
    if len(result.acquired_keys) != protocol.target_counts[-1] - protocol.bootstrap_count:
        raise RuntimeError("Acquisition length disagrees with the target protocol")

    acquisition_path = args.output_dir / "acquisition_order.pt"
    manifest_path = args.output_dir / "selection_manifest.json"
    runtime_path = args.output_dir / "runtime_report.json"
    score_cache_fingerprint = stable_fingerprint(score_payload)
    prototype_cache_fingerprint = stable_fingerprint(prototype_payload)
    acquisition_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bpus_v2_acquisition",
        "method": METHOD,
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "seed": int(args.seed),
        "bootstrap_keys": list(bootstrap_keys),
        "acquired_keys": list(result.acquired_keys),
        "utility": result.utility,
        "value": result.value,
        "novelty": result.novelty,
        "max_similarity": result.max_similarity,
        "cumulative_subset_counts": _cumulative_subset_counts(
            bootstrap_keys, result.acquired_keys
        ),
        "catalog_fingerprint": catalog_fingerprint,
        "key_order_fingerprint": key_order_fingerprint,
        "image_fingerprint": image_fingerprint,
        "dino_fingerprint": dino_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "formula_version": score_payload["formula_version"],
        "formula_fingerprint": score_payload["formula_fingerprint"],
        "formula_configuration_fingerprint": formula_configuration_fingerprint,
        "prototype_version": BPUS_V2_PROTOTYPE_VERSION,
        "prototype_fingerprint": prototype_payload["prototype_fingerprint"],
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
            split_path = (
                args.output_dir / f"bpus_v2_{count:04d}_seed{args.seed}.pt"
            )
            save_split_keys(split_path, result.splits[count])
            _save_split_text(split_path.with_suffix(".txt"), result.splits[count])
    _, split_fingerprints, split_txt_fingerprints = _verify_saved_splits(
        args.output_dir,
        seed=args.seed,
        counts=protocol.target_counts,
        catalog_keys=sample_keys,
        bootstrap_keys=bootstrap_keys,
    )
    diagnostics = _diagnostics(score_data, prototype_data, seed=args.seed)
    manifest_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bpus_v2_selection_manifest",
        "method": METHOD,
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
        "formula_version": score_payload["formula_version"],
        "formula_fingerprint": score_payload["formula_fingerprint"],
        "formula_configuration_fingerprint": formula_configuration_fingerprint,
        "formula": formula_spec,
        "prototype_version": BPUS_V2_PROTOTYPE_VERSION,
        "prototype_fingerprint": prototype_payload["prototype_fingerprint"],
        "prototype_level": "p2",
        "prototype_dim": EXPECTED_P2_DIM,
        "score_cache_fingerprint": score_cache_fingerprint,
        "prototype_cache_fingerprint": prototype_cache_fingerprint,
        "acquisition_fingerprint": acquisition_payload["acquisition_fingerprint"],
        "bootstrap_path": str(args.bootstrap_split),
        "bootstrap_fingerprint": split_fingerprint,
        "split_fingerprints": split_fingerprints,
        "split_txt_fingerprints": split_txt_fingerprints,
        "acquired_count": len(result.acquired_keys),
        "tie_break": ["higher_Q", "higher_V", "higher_N", "smaller_sample_key"],
        "diagnostics": diagnostics,
        "files": {
            "pool_scores": scores_path.name,
            "pool_prototypes": prototypes_path.name,
            "acquisition_order": acquisition_path.name,
            "runtime_report": runtime_path.name,
            "splits": {
                str(count): f"bpus_v2_{count:04d}_seed{args.seed}.pt"
                for count in protocol.target_counts
            },
            "split_txt": {
                str(count): f"bpus_v2_{count:04d}_seed{args.seed}.txt"
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
        if not runtime_path.is_file():
            raise FileNotFoundError(f"Missing runtime report: {runtime_path}")
        with runtime_path.open("r", encoding="utf-8") as stream:
            runtime = json.load(stream)
        runtime_identity = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bpus_v2_runtime",
            "method": METHOD,
            "protocol": protocol.name,
            "target_counts": list(protocol.target_counts),
            "seed": int(args.seed),
        }
        for field, expected in runtime_identity.items():
            if runtime.get(field) != expected:
                raise ValueError(f"runtime_report.json identity mismatch for {field}")
        print("Verification passed; cache replay and all nested splits are identical.")
        return 0

    _save_or_upgrade_manifest(manifest, manifest_base, manifest_path)
    runtime = _runtime_payload(
        args=args,
        protocol=protocol,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
        score_seconds=score_seconds,
        acquisition_seconds=acquisition_seconds,
        cache_mode=cache_mode,
    )
    atomic_json_save(runtime, runtime_path, refuse_mismatch=False)
    print(
        f"Saved strict nested splits {protocol.target_counts} to {args.output_dir}; "
        f"valid candidates={diagnostics['valid_boundary_count']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
