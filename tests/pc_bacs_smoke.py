"""CPU-only synthetic end-to-end smoke test for PC-BACS artifacts.

This intentionally exercises the production dataset, score, cache, KMeans,
selection, CSV, manifest, and atomic split writers without loading DINO weights
or exposing a mock mode in the production CLI.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.pc_bacs_config import PCBACSConfig, SCORE_FORMULA_VERSION
from select_pc_bacs import _save_csv
from utils.dataloader import SelectionPoolDataset
from utils.pc_bacs import (
    atomic_json_save,
    atomic_torch_save,
    build_feature_cache,
    build_nested_splits,
    build_score_cache,
    build_selection_manifest,
    compute_catalog_fingerprint,
    compute_image_fingerprint,
    fit_dino_kmeans,
    preprocessing_fingerprint,
    save_split_keys,
    score_pool,
    validate_feature_cache,
    validate_score_cache,
)


SAMPLE_COUNT = 12
TARGET_COUNTS = (4, 8)
N_CLUSTERS = 4
FEATURE_DIM = 768
INPUT_SIZE = 392


class MockSelector(nn.Module):
    """Deterministic, non-flip-equivariant stand-in for the legacy Decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.modes: list[str] = []

    def forward(self, images: torch.Tensor, *, pc_mode: str = "off"):
        self.modes.append(pc_mode)
        if pc_mode != "off":
            raise AssertionError(f"Synthetic selector must use pc_mode='off', got {pc_mode!r}.")

        gray = (
            0.50 * images[:, 0:1]
            + 0.30 * images[:, 1:2]
            + 0.20 * images[:, 2:3]
        )
        logits = F.interpolate(
            gray,
            size=(98, 98),
            mode="bilinear",
            align_corners=False,
        )
        horizontal_bias = torch.linspace(
            -0.75,
            0.75,
            98,
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 1, 98)
        logits = logits + horizontal_bias
        return logits, logits, logits, logits


def _write_synthetic_rgb_pool(root: Path) -> tuple[list[Path], dict[str, Path]]:
    image_roots: list[Path] = []
    image_paths: dict[str, Path] = {}
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width]

    for subset_index, subset in enumerate(("TR-CAMO", "TR-COD10K")):
        image_root = root / subset / "im"
        image_root.mkdir(parents=True)
        image_roots.append(image_root)
        for local_index in range(SAMPLE_COUNT // 2):
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            pixels[..., 0] = (
                xx * 7 + local_index * 19 + subset_index * 31
            ) % 256
            pixels[..., 1] = (
                yy * 11 + local_index * 23 + subset_index * 17
            ) % 256
            pixels[..., 2] = (
                ((xx // 4 + yy // 3 + local_index + subset_index) % 2) * 180
                + local_index * 9
            ) % 256

            path = image_root / f"sample_{local_index:02d}.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            image_paths[f"{subset}/sample_{local_index:02d}"] = path

    return image_roots, image_paths


def _synthetic_features(sample_count: int) -> torch.Tensor:
    features = torch.zeros(sample_count, FEATURE_DIM, dtype=torch.float32)
    for index in range(sample_count):
        cluster = index % N_CLUSTERS
        features[index, cluster] = 1.0
        features[index, N_CLUSTERS + index] = 0.01 * (index + 1)
    return features


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def _run_pipeline(
    image_roots: list[Path],
    image_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    dataset = SelectionPoolDataset(
        [str(path) for path in image_roots],
        image_size=INPUT_SIZE,
    )
    sample_keys = [dataset[index][0] for index in range(len(dataset))]
    if len(sample_keys) != SAMPLE_COUNT or sample_keys != sorted(sample_keys):
        raise AssertionError("Synthetic RGB catalog is not complete and stably ordered.")
    if set(sample_keys) != set(image_paths):
        raise AssertionError("Dataset stable keys do not match the synthetic RGB catalog.")

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    selector = MockSelector()
    score_rows = score_pool(
        selector,
        loader,
        device="cpu",
        use_amp=False,
    )
    if not selector.modes or set(selector.modes) != {"off"}:
        raise AssertionError("Scoring did not exclusively use the legacy/off path.")
    if [row["sample_key"] for row in score_rows] != sample_keys:
        raise AssertionError("Score rows are not aligned with the stable catalog order.")

    boundary = torch.tensor(
        [row["boundary_disagreement"] for row in score_rows],
        dtype=torch.float32,
    )
    global_disagreement = torch.tensor(
        [row["global_disagreement"] for row in score_rows],
        dtype=torch.float32,
    )
    scores = torch.tensor(
        [row["score"] for row in score_rows],
        dtype=torch.float32,
    )
    for name, values in (
        ("boundary", boundary),
        ("global", global_disagreement),
        ("score", scores),
    ):
        if values.shape != (SAMPLE_COUNT,) or not torch.isfinite(values).all():
            raise AssertionError(f"Invalid synthetic {name} vector: {values!r}")

    catalog_fingerprint = compute_catalog_fingerprint(sample_keys)
    image_fingerprint = compute_image_fingerprint(sample_keys, image_paths)
    preprocess_fingerprint = preprocessing_fingerprint(INPUT_SIZE)
    dino_fingerprint = hashlib.sha256(b"pc-bacs-synthetic-dino").hexdigest()
    selector_fingerprint = hashlib.sha256(b"pc-bacs-mock-selector-v1").hexdigest()
    features = _synthetic_features(SAMPLE_COUNT)

    feature_payload = build_feature_cache(
        sample_keys,
        features,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_weight_fingerprint=dino_fingerprint,
        preprocessing_fingerprint=preprocess_fingerprint,
        input_size=INPUT_SIZE,
    )
    feature_path = output_dir / "pc_bacs_features.pt"
    atomic_torch_save(feature_payload, feature_path)
    saved_feature_payload = _load_torch(feature_path)
    feature_schema = {
        "format_version",
        "sample_keys",
        "key_order_fingerprint",
        "features",
        "feature_type",
        "normalized",
        "input_size",
        "catalog_fingerprint",
        "image_fingerprint",
        "dino_weight_fingerprint",
        "preprocessing_fingerprint",
    }
    if not feature_schema.issubset(saved_feature_payload):
        raise AssertionError("Feature cache is missing required schema fields.")
    validated_keys, validated_features = validate_feature_cache(
        saved_feature_payload,
        expected_sample_keys=sample_keys,
        expected_catalog_fingerprint=catalog_fingerprint,
        expected_image_fingerprint=image_fingerprint,
        expected_dino_weight_fingerprint=dino_fingerprint,
        expected_preprocessing_fingerprint=preprocess_fingerprint,
        feature_dim=FEATURE_DIM,
        input_size=INPUT_SIZE,
    )
    if validated_keys != sample_keys or not torch.equal(validated_features, features):
        raise AssertionError("Feature cache validation changed keys or features.")

    score_payload = build_score_cache(
        sample_keys,
        boundary,
        global_disagreement,
        scores,
        selector_fingerprint=selector_fingerprint,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        preprocessing_fingerprint=preprocess_fingerprint,
        score_formula_version=SCORE_FORMULA_VERSION,
    )
    score_path = output_dir / "pc_bacs_scores.pt"
    atomic_torch_save(score_payload, score_path)
    saved_score_payload = _load_torch(score_path)
    score_schema = {
        "format_version",
        "sample_keys",
        "key_order_fingerprint",
        "boundary_disagreement",
        "global_disagreement",
        "scores",
        "selector_fingerprint",
        "catalog_fingerprint",
        "image_fingerprint",
        "preprocessing_fingerprint",
        "score_formula_version",
    }
    if not score_schema.issubset(saved_score_payload):
        raise AssertionError("Score cache is missing required schema fields.")
    validated_scores = validate_score_cache(
        saved_score_payload,
        expected_sample_keys=sample_keys,
        expected_selector_fingerprint=selector_fingerprint,
        expected_catalog_fingerprint=catalog_fingerprint,
        expected_image_fingerprint=image_fingerprint,
        expected_preprocessing_fingerprint=preprocess_fingerprint,
        expected_score_formula_version=SCORE_FORMULA_VERSION,
    )
    if not torch.equal(validated_scores["scores"], scores):
        raise AssertionError("Score cache validation changed PC-BACS scores.")

    kmeans = fit_dino_kmeans(
        sample_keys,
        validated_features,
        n_clusters=N_CLUSTERS,
        random_seed=2025,
    )
    selection = build_nested_splits(
        sample_keys,
        kmeans.normalized_features,
        kmeans.cluster_ids,
        scores,
        kmeans.seed_keys,
        target_counts=TARGET_COUNTS,
        dedup_threshold=0.98,
    )
    split_4 = selection.splits[4]
    split_8 = selection.splits[8]
    if len(split_4) != 4 or len(split_8) != 8:
        raise AssertionError("Synthetic nested splits do not meet exact budgets 4/8.")
    if not set(split_4).issubset(split_8):
        raise AssertionError("Synthetic 4-sample split is not nested in the 8-sample split.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest: dict[int, dict[str, object]] = {}
    for count, split in ((4, split_4), (8, split_8)):
        split_path = output_dir / f"pc_bacs_{count:04d}_keys.pt"
        split_fingerprint = save_split_keys(split_path, split)
        saved_split = _load_torch(split_path)
        if saved_split != sorted(split) or len(saved_split) != count:
            raise AssertionError(f"Saved {count}-sample split is not a sorted list[str].")
        output_manifest[count] = {
            "path": split_path.name,
            "count": count,
            "fingerprint": split_fingerprint,
        }

    csv_path = output_dir / "pc_bacs_scores.csv"
    _save_csv(
        csv_path,
        sample_keys=sample_keys,
        cluster_ids=kmeans.cluster_ids,
        center_distances=kmeans.center_distances,
        boundary=boundary,
        global_disagreement=global_disagreement,
        scores=scores,
        selection_result=selection,
        target_counts=TARGET_COUNTS,
    )
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        csv_rows = list(reader)
        fields = set(reader.fieldnames or ())
    required_csv_fields = {
        "sample_key",
        "cluster_id",
        "boundary_disagreement",
        "global_disagreement",
        "pc_bacs_score",
        "dino_center_distance",
        "selected_0004",
        "selection_rank_0004",
        "selected_0008",
        "selection_rank_0008",
        "dedup_skipped_count",
    }
    if len(csv_rows) != SAMPLE_COUNT or not required_csv_fields.issubset(fields):
        raise AssertionError("CSV artifact does not satisfy the production schema.")
    if sum(int(row["selected_0004"]) for row in csv_rows) != 4:
        raise AssertionError("CSV 4-sample selection flags are incorrect.")
    if sum(int(row["selected_0008"]) for row in csv_rows) != 8:
        raise AssertionError("CSV 8-sample selection flags are incorrect.")

    config = PCBACSConfig(
        n_clusters=N_CLUSTERS,
        target_counts=TARGET_COUNTS,
        selector_seed_count=N_CLUSTERS,
        feature_batch_size=4,
        score_batch_size=4,
        num_workers=0,
        use_amp=False,
        dedup_threshold=0.98,
    )
    config.validate(sample_count=SAMPLE_COUNT)
    manifest = build_selection_manifest(
        config=config,
        dataset={
            "sample_count": SAMPLE_COUNT,
            "catalog_fingerprint": catalog_fingerprint,
            "image_fingerprint": image_fingerprint,
            "preprocessing_fingerprint": preprocess_fingerprint,
        },
        selector={
            "kind": "synthetic-mock",
            "fingerprint": selector_fingerprint,
            "pc_mode": "off",
        },
        outputs=output_manifest,
        repo_commit="synthetic-smoke",
        selection_result=selection,
        runtime={
            "device": "cpu",
            "dino_loaded": False,
            "cuda_used": False,
        },
        fingerprints={
            "feature_cache": _sha256_file(feature_path),
            "score_cache": _sha256_file(score_path),
        },
        extra={"synthetic_rgb_count": SAMPLE_COUNT},
    )
    manifest_path = output_dir / "pc_bacs_manifest.json"
    atomic_json_save(manifest, manifest_path)
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved_manifest["method"] != "PC-BACS":
        raise AssertionError("Manifest method is not PC-BACS.")
    if saved_manifest["selection"]["score_formula_version"] != SCORE_FORMULA_VERSION:
        raise AssertionError("Manifest score formula version is incorrect.")
    if sorted(int(count) for count in saved_manifest["outputs"]) != [4, 8]:
        raise AssertionError("Manifest does not describe both nested splits.")

    artifact_paths = (
        feature_path,
        score_path,
        output_dir / "pc_bacs_0004_keys.pt",
        output_dir / "pc_bacs_0008_keys.pt",
        csv_path,
        manifest_path,
    )
    return {
        "hashes": {path.name: _sha256_file(path) for path in artifact_paths},
        "splits": {4: list(split_4), 8: list(split_8)},
        "clusters": [int(value) for value in kmeans.cluster_ids],
        "scores": scores.tolist(),
    }


def main() -> int:
    torch.manual_seed(2025)
    np.random.seed(2025)

    with tempfile.TemporaryDirectory(prefix="pc-bacs-smoke-") as temporary:
        root = Path(temporary)
        image_roots, image_paths = _write_synthetic_rgb_pool(root / "Dataset" / "COD")
        output_dir = root / "artifacts"

        first = _run_pipeline(image_roots, image_paths, output_dir)
        second = _run_pipeline(image_roots, image_paths, output_dir)
        if first != second:
            raise AssertionError("Repeated synthetic PC-BACS runs are not byte-for-byte stable.")

    print("PC-BACS synthetic CPU smoke: PASS (12 RGB, nested 4/8, deterministic rerun)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
