"""Real-CUDA DINO PC-HBM forward/backward smoke.

The physical batch sizes are acceptance constraints and are never reduced.
On OOM, only the locked PC-HBM capacity sequence is attempted.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.base_model import BaseModel
from Model.decoder import Decoder
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import pc_hbm_labeled_loss, pc_unlabeled_loss


BATCH_BASE = 16
BATCH_TEACHER = 32
BATCH_STUDENT_LABELED = 32
BATCH_STUDENT_UNLABELED = 32


TUNING_ATTEMPTS: tuple[tuple[str, dict[str, int]], ...] = (
    (
        "default",
        {
            "query_chunk_size": 512,
            "p1_max_tokens": 384,
            "p3_max_tokens": 128,
            "p2_max_tokens": 128,
            "parent_topk": 16,
        },
    ),
    (
        "chunk_256",
        {
            "query_chunk_size": 256,
            "p1_max_tokens": 384,
            "p3_max_tokens": 128,
            "p2_max_tokens": 128,
            "parent_topk": 16,
        },
    ),
    (
        "p1_256",
        {
            "query_chunk_size": 256,
            "p1_max_tokens": 256,
            "p3_max_tokens": 128,
            "p2_max_tokens": 128,
            "parent_topk": 16,
        },
    ),
    (
        "p3_p2_96",
        {
            "query_chunk_size": 256,
            "p1_max_tokens": 256,
            "p3_max_tokens": 96,
            "p2_max_tokens": 96,
            "parent_topk": 16,
        },
    ),
    (
        "topk_12",
        {
            "query_chunk_size": 256,
            "p1_max_tokens": 256,
            "p3_max_tokens": 96,
            "p2_max_tokens": 96,
            "parent_topk": 12,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-CUDA PC-HBM acceptance smoke.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--attempt",
        choices=[name for name, _ in TUNING_ATTEMPTS],
        default=None,
        help="Run only one capacity profile; default tries the locked OOM sequence.",
    )
    return parser.parse_args()


def build_synthetic_memory(config: DinoPCHBMConfig) -> PCMemory:
    """Build a small compatible labeled-only CPU-FP16 retrieval bank."""

    generator = torch.Generator(device="cpu").manual_seed(17)
    image_ids = ["cuda-smoke-memory"]
    parent_count = max(32, config.parent_topk)
    route = {
        name: torch.randn(1, config.memory_dim, generator=generator)
        for name in (
            "x3_global",
            "x3_boundary",
            "x3_uncertain",
            "x3_bg_near",
            "x3_environment",
        )
    }
    route["route_embed"] = torch.randn(1, config.memory_dim, generator=generator)
    route["img_ids"] = image_ids
    metadata = [
        {"image_id": image_ids[0], "region": "fg_boundary", "sample_key": image_ids[0]}
        for _ in range(parent_count)
    ]
    memory = PCMemory(
        config.memory_dim,
        config.value_dim,
        config.geometry_dim,
        storage_dtype=torch.float16,
        config=config,
    )
    memory.append(
        {
            "source": "labeled_only",
            "route": route,
            "parent": {
                "p3_keys": torch.randn(parent_count, config.memory_dim, generator=generator),
                "p3_values": torch.randn(parent_count, config.value_dim, generator=generator),
                "p3_geometry": torch.randn(parent_count, config.geometry_dim, generator=generator),
                "child_ptr": torch.arange(parent_count, dtype=torch.long),
                "parent_meta": metadata,
            },
            "child": {
                "p2_child_keys": torch.randn(parent_count, config.memory_dim, generator=generator),
                "p2_child_geo": torch.randn(parent_count, config.geometry_dim, generator=generator),
                "child_meta": metadata,
            },
        }
    )
    memory.finalize(
        compat_meta=config.expected_memory_meta(producer_fingerprint="cuda-smoke")
    )
    return memory


def extract_features(model: BaseModel, batch_size: int, config: DinoPCHBMConfig):
    images = torch.randn(
        batch_size,
        3,
        config.input_size,
        config.input_size,
        device="cuda",
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model.extract_features(images)
    del images
    return tuple(feature.detach() for feature in features)


def real_training_loss(
    outputs,
    aux,
    config: DinoPCHBMConfig,
    *,
    pc_mode: str,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the same loss family used by the real Base/TS trainers."""

    z_main = aux.get("z_main")
    if not torch.is_tensor(z_main):
        raise AssertionError("Decoder aux is missing z_main logits")
    if pc_mode == "student_core":
        pseudo = torch.rand_like(z_main, dtype=torch.float32)
        confidence = torch.full_like(pseudo, 0.75)
        loss, metrics = pc_unlabeled_loss(
            outputs,
            aux,
            pseudo,
            confidence,
            epoch=1,
            config=config,
        )
    else:
        gt = (torch.rand_like(z_main, dtype=torch.float32) > 0.5).float()
        loss, metrics = pc_hbm_labeled_loss(
            outputs,
            aux,
            gt,
            epoch,
            config,
            pc_mode=pc_mode,
            strict=True,
        )
        for name in ("L_B3", "L_gate", "L_boundary"):
            value = metrics.get(name)
            if not torch.is_tensor(value) or not torch.isfinite(value):
                raise AssertionError(f"Missing or non-finite real labeled metric: {name}")
        if not bool(metrics["L_B3"] > 0):
            raise AssertionError("Real labeled smoke did not activate L_B3")
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite PC-HBM smoke loss")
    return loss, metrics


def assert_finite_gradients(module: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("Backward produced no Decoder gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Backward produced non-finite Decoder gradients")


def assert_module_received_gradients(module: torch.nn.Module, name: str) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError(f"{name} received no gradient from the real PC-HBM loss")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{name} received non-finite gradients")


def peak_gib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def release(*objects: Any) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def run_training_scenario(
    label: str,
    decoder: Decoder,
    model: BaseModel,
    memory: PCMemory,
    config: DinoPCHBMConfig,
    *,
    batch_size: int,
    pc_mode: str,
    epoch: int,
) -> float:
    decoder.train()
    decoder.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    features = extract_features(model, batch_size, config)
    query_ids = [f"{label}-{index}" for index in range(batch_size)]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, aux = decoder(
            features,
            memory=memory,
            pc_mode=pc_mode,
            epoch=epoch,
            return_aux=True,
            query_image_ids=query_ids if pc_mode == "full" else None,
        )
        if not aux.get("pc_active", False):
            raise AssertionError(f"{label}: PC path unexpectedly inactive: {aux.get('fallback_reason')}")
        if pc_mode == "student_core" and not aux.get("mixture_skipped", False):
            raise AssertionError("Student core must skip P1/mixture")
        loss, metrics = real_training_loss(
            outputs,
            aux,
            config,
            pc_mode=pc_mode,
            epoch=epoch,
        )
    loss.backward()
    assert_finite_gradients(decoder)
    if pc_mode == "full":
        assert_module_received_gradients(decoder.pc_hbm.boundary3, "B3 boundary head")
        assert_module_received_gradients(decoder.pc_hbm.gate_mlp, "PC gate MLP")
    peak = peak_gib()
    decoder.zero_grad(set_to_none=True)
    del loss, metrics, outputs, aux, features
    release()
    print(f"[PASS] {label}: batch={batch_size}, mode={pc_mode}, peak={peak:.2f} GiB")
    return peak


def run_teacher_scenario(
    model: BaseModel,
    memory: PCMemory,
    config: DinoPCHBMConfig,
) -> float:
    teacher = Decoder(pc_cfg=config)
    teacher.load_state_dict(model.decoder.state_dict(), strict=True)
    teacher.requires_grad_(False).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    features = extract_features(model, BATCH_TEACHER, config)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _, aux = teacher(
            features,
            memory=memory,
            pc_mode="teacher_pseudo",
            epoch=30,
            return_aux=True,
        )
        probability = aux.get("p_final")
        if not aux.get("pc_active", False) or probability is None:
            raise AssertionError(
                f"Teacher PC path unexpectedly inactive: {aux.get('fallback_reason')}"
            )
        if not torch.isfinite(probability).all():
            raise AssertionError("Teacher p_final contains non-finite values")
        if probability.min() < 0 or probability.max() > 1:
            raise AssertionError("Teacher p_final is not a probability tensor")
    peak = peak_gib()
    del probability, aux, features, teacher
    release()
    print(
        f"[PASS] teacher inference: batch={BATCH_TEACHER}, "
        f"mode=teacher_pseudo, peak={peak:.2f} GiB"
    )
    return peak


def run_attempt(name: str, overrides: dict[str, int], seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = DinoPCHBMConfig(**overrides)
    memory = build_synthetic_memory(config)
    model = BaseModel(pc_cfg=config).cuda()
    model.dino.eval()
    peaks = {
        "base_full_b16": run_training_scenario(
            "base full",
            model.decoder,
            model,
            memory,
            config,
            batch_size=BATCH_BASE,
            pc_mode="full",
            epoch=11,
        ),
        "teacher_b32": run_teacher_scenario(model, memory, config),
        "student_labeled_full_b32": run_training_scenario(
            "student labeled",
            model.decoder,
            model,
            memory,
            config,
            batch_size=BATCH_STUDENT_LABELED,
            pc_mode="full",
            epoch=31,
        ),
        "student_unlabeled_core_b32": run_training_scenario(
            "student unlabeled",
            model.decoder,
            model,
            memory,
            config,
            batch_size=BATCH_STUDENT_UNLABELED,
            pc_mode="student_core",
            epoch=31,
        ),
    }
    del model, memory
    release()
    print(f"[PASS] CUDA smoke profile={name}, max_peak={max(peaks.values()):.2f} GiB")
    return peaks


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def main() -> int:
    args = parse_args()
    checkpoint = REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required local DINO checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires a real CUDA device")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    attempts = (
        [item for item in TUNING_ATTEMPTS if item[0] == args.attempt]
        if args.attempt is not None
        else list(TUNING_ATTEMPTS)
    )
    failures: list[str] = []
    for name, overrides in attempts:
        print(f"[RUN] profile={name}, settings={overrides}")
        try:
            run_attempt(name, overrides, args.seed)
            return 0
        except BaseException as error:
            if not is_cuda_oom(error):
                raise
            failures.append(f"{name}: {error}")
            print(f"[OOM] profile={name}: {error}")
            gc.collect()
            torch.cuda.empty_cache()
    print("[FAIL] CUDA smoke exhausted the locked tuning sequence without reducing batch size.")
    for failure in failures:
        print(f"  - {failure}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
