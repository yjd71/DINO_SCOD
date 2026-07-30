"""Comparable single-GPU profiler for the fixed PC-HBM-Lite protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.base_model_config import Config
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import (
    base_structure_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
)
from Model.PC_HBM.training.ema import update_ema_module
from Model.base_model import BaseModel
from Model.decoder import Decoder
from utils.checkpoint_pc_hbm import (
    CANONICAL_LABELED_SPLIT_COUNT,
    CANONICAL_LABELED_SPLIT_FINGERPRINT,
    save_memory_checkpoint,
    validate_canonical_labeled_indices_pt,
)
from utils.dataloader import build_labeled_memory_loader
from utils.pc_memory_runner import rebuild_memory

PROFILE_SEED = 2025
PROFILE_WARMUP = 10
PROFILE_SAMPLES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=PROFILE_SEED)
    parser.add_argument("--labeled-indices-pt", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "pc_hbm_complex_e35fbfca.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "pc_hbm_lite.json",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _profile(call, warmup: int, samples: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(samples):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(elapsed)
    p95_index = min(
        len(ordered) - 1,
        max(0, math.ceil(0.95 * len(ordered)) - 1),
    )
    return {
        "warmup_iterations": warmup,
        "timed_iterations": samples,
        "mean_ms": statistics.fmean(elapsed),
        "median_ms": statistics.median(elapsed),
        "p95_ms": ordered[p95_index],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _features(
    batch: int,
    device: torch.device,
    *,
    token_size: int,
    encoder_dim: int,
):
    return tuple(
        torch.randn(
            batch,
            token_size * token_size,
            encoder_dim,
            device=device,
        )
        for _ in range(4)
    )


def _load_baseline_protocol(
    path: Path,
    *,
    device_name: str,
) -> dict:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "architecture": "DINO_SCOD_PC_HBM_COMPLEX",
        "device": device_name,
        "seed": PROFILE_SEED,
        "amp": True,
        "labeled_split_count": CANONICAL_LABELED_SPLIT_COUNT,
        "labeled_split_fingerprint": CANONICAL_LABELED_SPLIT_FINGERPRINT,
        "warmup_iterations": PROFILE_WARMUP,
        "timed_iterations": PROFILE_SAMPLES,
    }
    for key, expected_value in expected.items():
        actual = baseline.get(key)
        if actual != expected_value:
            raise RuntimeError(
                "Complex baseline protocol mismatch for "
                f"{key}: expected {expected_value!r}, got {actual!r}"
            )
    for scenario in (
        "base_full_batch16",
        "teacher_pseudo_batch32",
        "ts_step_batch32",
    ):
        metrics = baseline.get(scenario)
        if not isinstance(metrics, dict):
            raise RuntimeError(f"Complex baseline is missing {scenario}")
        if metrics.get("warmup_iterations") != PROFILE_WARMUP:
            raise RuntimeError(
                f"Complex baseline {scenario} did not use 10 warmups"
            )
        if metrics.get("timed_iterations") != PROFILE_SAMPLES:
            raise RuntimeError(
                f"Complex baseline {scenario} did not use 30 timed iterations"
            )
    return baseline


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The Lite profiler requires one CUDA device")
    if args.seed != PROFILE_SEED:
        raise ValueError("The fixed profiling protocol requires seed=2025")
    if not args.baseline_json.is_file():
        raise FileNotFoundError(args.baseline_json)
    _set_seed(args.seed)
    device = torch.device("cuda")
    cfg = Config()
    cfg.train_labeled_indices_pt = str(args.labeled_indices_pt)
    split_fingerprint = validate_canonical_labeled_indices_pt(
        args.labeled_indices_pt
    )
    pc_cfg = DinoPCHBMConfig()
    cfg.train_size = int(pc_cfg.input_size)
    model = BaseModel(pc_cfg=pc_cfg).to(device).eval()
    if args.decoder_checkpoint is not None:
        model.load_decoder_checkpoint(
            str(args.decoder_checkpoint),
            require_pc_complete=True,
        )

    loader = build_labeled_memory_loader(
        l_image_root=cfg.train_imgs,
        l_gt_root=cfg.train_masks,
        l_txt_root=cfg.train_sample_txt,
        l_train_size=cfg.train_size,
        labeled_indices_pt=str(args.labeled_indices_pt),
        batch_size=16,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if len(loader.dataset) != 202:
        raise RuntimeError(
            f"Fixed profiler split must contain 202 samples, got {len(loader.dataset)}"
        )
    memory = PCMemory(config=pc_cfg)
    rebuild_memory(
        model=model,
        memory_decoder=model.decoder,
        memory_loader=loader,
        memory=memory,
        device=device,
        config=pc_cfg,
        use_amp=True,
    )
    if not memory.is_ready():
        raise RuntimeError("Profiler memory rebuild failed")

    decoder = model.decoder.eval()
    # Timings intentionally use precomputed DINO tokens, matching the complex
    # baseline protocol and isolating Decoder/PC cost.
    features16 = _features(
        16,
        device,
        token_size=pc_cfg.token_size,
        encoder_dim=pc_cfg.encoder_dim,
    )
    features32 = _features(
        32,
        device,
        token_size=pc_cfg.token_size,
        encoder_dim=pc_cfg.encoder_dim,
    )
    model.dino = None
    torch.cuda.empty_cache()

    def run_base_full():
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16
        ):
            return decoder(
                features16,
                memory=memory,
                pc_mode="full",
                epoch=13,
                return_aux=True,
            )

    def run_teacher_pseudo():
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16
        ):
            return decoder(
                features32,
                memory=memory,
                pc_mode="teacher_pseudo",
                epoch=30,
                return_aux=True,
            )

    base_metrics = _profile(
        run_base_full,
        PROFILE_WARMUP,
        PROFILE_SAMPLES,
    )
    teacher_metrics = _profile(
        run_teacher_pseudo,
        PROFILE_WARMUP,
        PROFILE_SAMPLES,
    )
    _, full_aux = run_base_full()
    pc = full_aux["pc_hbm"]
    query_valid = pc["query_valid"].float()
    candidate_valid = pc["retrieval_valid"].float()
    query_count = int(query_valid.numel())

    student = Decoder(
        in_dim=pc_cfg.encoder_dim,
        out_dim=pc_cfg.decoder_dim,
        pc_cfg=None,
    ).to(device).train()
    student.load_state_dict(
        {
            name: value
            for name, value in decoder.state_dict().items()
            if not name.startswith("pc_hbm.")
        },
        strict=True,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-4)
    labeled_gt = torch.rand(
        32,
        1,
        pc_cfg.output_size,
        pc_cfg.output_size,
        device=device,
    )

    def run_ts_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16
        ):
            _, teacher_aux = decoder(
                features32,
                memory=memory,
                pc_mode="teacher_pseudo",
                epoch=30,
                return_aux=True,
            )
        pseudo = prepare_pseudo_targets(teacher_aux)
        with torch.autocast("cuda", dtype=torch.float16):
            labeled_outputs, _ = student(
                features32, pc_mode="off", return_aux=True
            )
            labeled_loss = base_structure_loss(
                labeled_outputs, labeled_gt
            )
            unlabeled_outputs, student_aux = student(
                features32, pc_mode="off", return_aux=True
            )
            unlabeled_loss, _ = pc_unlabeled_loss(
                unlabeled_outputs,
                student_aux,
                pseudo["p_soft"],
                pseudo["confidence"],
                pc_cfg,
                teacher_features={"p3_corr": pseudo["p3_corr"]},
            )
            loss = labeled_loss + unlabeled_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            update_ema_module(
                student,
                decoder,
                momentum=pc_cfg.ema_momentum,
                shared_only=True,
                exclude_prefixes=("pc_hbm.",),
            )
        return loss

    ts_metrics = _profile(
        run_ts_step,
        PROFILE_WARMUP,
        PROFILE_SAMPLES,
    )
    memory_state = memory.state_dict()
    with tempfile.TemporaryDirectory() as directory:
        raw_path = Path(directory) / "memory_v2_state.pth"
        checkpoint_path = Path(directory) / "memory_v2_checkpoint.pth"
        torch.save(memory_state, raw_path)
        save_memory_checkpoint(
            checkpoint_path,
            memory,
            compat_meta=memory.compat_meta,
        )
        raw_state_bytes = raw_path.stat().st_size
        checkpoint_bytes = checkpoint_path.stat().st_size
    device_name = torch.cuda.get_device_name(device)
    complex_baseline = _load_baseline_protocol(
        args.baseline_json,
        device_name=device_name,
    )

    result = {
        "architecture": pc_cfg.memory_architecture,
        "device": device_name,
        "torch": torch.__version__,
        "seed": args.seed,
        "amp": True,
        "labeled_split": str(args.labeled_indices_pt),
        "labeled_split_count": len(loader.dataset),
        "labeled_split_fingerprint": split_fingerprint,
        "warmup_iterations": PROFILE_WARMUP,
        "timed_iterations": PROFILE_SAMPLES,
        "pc_parameters": sum(
            parameter.numel()
            for parameter in decoder.pc_hbm.parameters()
        ),
        "memory": {
            "route_tensor_bytes": _tensor_bytes(memory_state["route"]),
            "pair_tensor_bytes": _tensor_bytes(memory_state["pairs"]),
            "total_tensor_bytes": _tensor_bytes(memory_state),
            "raw_state_file_bytes": raw_state_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "route_images": len(memory_state["route"]["img_ids"]),
            "pair_count": int(memory_state["pairs"]["p3_keys"].shape[0]),
        },
        "base_full_batch16": base_metrics,
        "teacher_pseudo_batch32": teacher_metrics,
        "ts_step_batch32": ts_metrics,
        "retrieval": {
            "queries_per_image": query_count / 16.0,
            "candidates_per_query": (
                float(candidate_valid.sum()) / query_count
                if query_count
                else 0.0
            ),
            "candidate_valid_ratio": (
                float(candidate_valid.mean())
                if candidate_valid.numel()
                else 0.0
            ),
            "pair_valid_ratio": (
                float(query_valid.mean()) if query_count else 0.0
            ),
        },
        "complex_baseline": complex_baseline,
        "notes": [
            "Forward timings use precomputed DINO token tensors.",
            "Memory was rebuilt from all 202 labeled samples as CPU FP16.",
            (
                "TS timing includes Teacher pseudo, two raw Student forwards, "
                "Adam backward/step, and shared-legacy EMA."
            ),
        ],
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
