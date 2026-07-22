"""Fixed-physical-batch CUDA AMP smoke for encoder-side PC-HBM v3.

This script intentionally exposes no batch-size override and has no OOM
fallback.  The only reduced capacities are route top-2, parent top-2, and
query chunk 32, supplied by :func:`ddp_smoke_encoder_pc.smoke_config`.  It loads
the real local frozen DINOv2-B/14 checkpoint and the real unchanged original
Decoder with decoder-side PC permanently detached.
"""

from __future__ import annotations

import copy
import gc
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from Model.decoder import Decoder
from Model.PC_HBM.encoder import DinoFeatureBundle
from ddp_smoke_encoder_pc import (
    EncoderPCSmokePipeline,
    SMOKE_IMAGE_IDS,
    smoke_config,
    synthetic_gt,
    synthetic_memory,
)
from Model.PC_HBM.encoder.teacher_pseudo_refiner import (
    teacher_pseudo_refiner_labeled_loss,
)
from Model.PC_HBM.training import (
    EncoderPCStage,
    build_encoder_pc_optimizer,
    configure_encoder_pc_stage,
    encoder_pc_unlabeled_loss,
    prepare_encoder_pc_pseudo_targets,
)
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.pc_memory_runner import module_fingerprint

from ddp_smoke_encoder_pc import smoke_labeled_loss


BASE_BATCH = 16
TS_BATCH = 32


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _peak_gib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / float(1024**3)


def _release(*objects: Any) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def _ids(batch_size: int) -> list[str]:
    return [SMOKE_IMAGE_IDS[index % len(SMOKE_IMAGE_IDS)] for index in range(batch_size)]


def _assert_finite_gradients(model: EncoderPCSmokePipeline, label: str) -> None:
    groups = {
        "adapter": model.student_adapter,
        "decoder": model.student_decoder,
        "refiner": model.student_refiner,
    }
    for name, module in groups.items():
        gradients = [
            (parameter_name, parameter.grad)
            for parameter_name, parameter in module.named_parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise AssertionError(f"{label}: {name} received no gradients")
        non_finite = [
            parameter_name
            for parameter_name, value in gradients
            if not torch.isfinite(value).all()
        ]
        if non_finite:
            raise AssertionError(
                f"{label}: {name} received non-finite gradients in {non_finite[:8]}"
            )


def _load_frozen_dino(device: torch.device) -> tuple[torch.nn.Module, str]:
    dino = torch.hub.load(
        str(REPO_ROOT / "dinov2"),
        "dinov2_vitb14",
        source="local",
        pretrained=False,
    )
    checkpoint = torch.load(
        REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth",
        map_location="cpu",
        weights_only=True,
    )
    dino.load_state_dict(checkpoint)
    dino.requires_grad_(False).eval()
    fingerprint = module_fingerprint(dino)
    return dino.to(device), fingerprint


@torch.inference_mode()
def _extract_real_bundle(
    dino: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> DinoFeatureBundle:
    images = torch.zeros(batch_size, 3, 392, 392, device=device)
    dino.eval()
    with _autocast():
        pairs = dino.get_intermediate_layers(
            x=images,
            n=(2, 5, 8, 11),
            reshape=False,
            return_class_token=True,
            norm=True,
        )
    bundle = DinoFeatureBundle(
        patch_tokens=tuple(pair[0] for pair in pairs),
        cls_tokens=tuple(pair[1] for pair in pairs),
    ).validate()
    if any(value.requires_grad for value in (*bundle.patch_tokens, *bundle.cls_tokens)):
        raise AssertionError("Frozen real DINO emitted differentiable features")
    return bundle


def _ready_memory(
    model: EncoderPCSmokePipeline,
    config,
    *,
    dino_weight_fingerprint: str,
) -> Any:
    producer = module_fingerprint(model.teacher_adapter)
    memory = synthetic_memory(
        config,
        producer_fingerprint=producer,
        dino_weight_fingerprint=dino_weight_fingerprint,
    )
    if memory.compat_meta["producer_fingerprint"] != producer:
        raise AssertionError("CUDA memory producer fingerprint mismatch")
    expected_split = compute_labeled_split_fingerprint(SMOKE_IMAGE_IDS)
    if memory.compat_meta["labeled_split_fingerprint"] != expected_split:
        raise AssertionError("CUDA memory labeled split fingerprint mismatch")
    if memory.compat_meta["dino_weight_fingerprint"] != dino_weight_fingerprint:
        raise AssertionError("CUDA memory frozen-DINO fingerprint mismatch")
    if memory.num_images < 3:
        raise AssertionError("CUDA smoke requires at least three routed images")
    return memory


def _run_base_batch16(
    model: EncoderPCSmokePipeline,
    dino: torch.nn.Module,
    dino_weight_fingerprint: str,
    config,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> float:
    stage = EncoderPCStage.for_epoch(21, config)
    configure_encoder_pc_stage(
        model.student_adapter,
        model.student_decoder,
        model.student_refiner,
        stage,
    )
    model.update_teacher(momentum=0.0)
    memory = _ready_memory(
        model,
        config,
        dino_weight_fingerprint=dino_weight_fingerprint,
    )
    model.student_refiner.call_count = 0
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    bundle = _extract_real_bundle(dino, BASE_BATCH, torch.device("cuda"))
    if bundle.patch_tokens[0].shape[0] != BASE_BATCH:
        raise AssertionError("Base physical batch is not fixed at 16")
    gt = synthetic_gt(BASE_BATCH, torch.device("cuda"))
    with _autocast():
        outputs, aux, refined = model(
            bundle.patch_tokens,
            bundle.cls_tokens,
            memory,
            branch="base",
            stage=stage,
            image_ids=_ids(BASE_BATCH),
        )
        core_loss = smoke_labeled_loss(outputs, aux, gt, config, stage)
        refiner_loss, _ = teacher_pseudo_refiner_labeled_loss(refined, gt, config)
        loss = core_loss + refiner_loss
    if outputs[3].shape != (BASE_BATCH, 1, 98, 98):
        raise AssertionError("Base z_core shape contract failed")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    _assert_finite_gradients(model, "Base batch16")
    scaler.step(optimizer)
    scaler.update()
    model.update_teacher(momentum=config.ema_momentum)
    if model.student_refiner.call_count != 1:
        raise AssertionError("Base epoch21 must execute labeled Refiner once")
    peak = _peak_gib()
    del bundle, gt, outputs, aux, refined, core_loss, refiner_loss, loss, memory
    optimizer.zero_grad(set_to_none=True)
    _release()
    return peak


def _run_ts_batch32(
    model: EncoderPCSmokePipeline,
    dino: torch.nn.Module,
    dino_weight_fingerprint: str,
    config,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> float:
    stage = EncoderPCStage.for_epoch(config.final_epoch, config)
    configure_encoder_pc_stage(
        model.student_adapter,
        model.student_decoder,
        model.student_refiner,
        stage,
    )
    model.update_teacher(momentum=0.0)
    memory = _ready_memory(
        model,
        config,
        dino_weight_fingerprint=dino_weight_fingerprint,
    )
    model.student_refiner.call_count = 0
    model.teacher_refiner.call_count = 0
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    # Student labeled B=32: full core followed by the detached labeled Refiner.
    labeled_bundle = _extract_real_bundle(dino, TS_BATCH, torch.device("cuda"))
    if labeled_bundle.patch_tokens[0].shape[0] != TS_BATCH:
        raise AssertionError("Student labeled physical batch is not fixed at 32")
    labeled_gt = synthetic_gt(TS_BATCH, torch.device("cuda"))
    with _autocast():
        labeled_outputs, labeled_aux, labeled_refined = model(
            labeled_bundle.patch_tokens,
            labeled_bundle.cls_tokens,
            memory,
            branch="student_labeled",
            stage=stage,
            image_ids=_ids(TS_BATCH),
        )
        labeled_core_loss = smoke_labeled_loss(
            labeled_outputs, labeled_aux, labeled_gt, config, stage
        )
        labeled_refiner_loss, _ = teacher_pseudo_refiner_labeled_loss(
            labeled_refined, labeled_gt, config
        )
        labeled_loss = labeled_core_loss + labeled_refiner_loss
    scaler.scale(labeled_loss).backward()
    _assert_finite_gradients(model, "TS labeled batch32")
    if labeled_outputs[3].shape[0] != TS_BATCH:
        raise AssertionError("Student labeled z_core did not retain batch32")
    del labeled_bundle, labeled_gt, labeled_outputs, labeled_aux
    del labeled_refined, labeled_core_loss, labeled_refiner_loss, labeled_loss
    _release()

    # EMA Teacher B=32: the only unlabeled branch allowed to execute Refiner.
    teacher_bundle = _extract_real_bundle(dino, TS_BATCH, torch.device("cuda"))
    if teacher_bundle.patch_tokens[0].shape[0] != TS_BATCH:
        raise AssertionError("Teacher physical batch is not fixed at 32")
    with torch.inference_mode(), _autocast():
        teacher_payload = model.teacher_pseudo(teacher_bundle, memory, stage)
    if teacher_payload["z_core"].shape[0] != TS_BATCH:
        raise AssertionError("Teacher z_core did not retain batch32")
    pseudo = prepare_encoder_pc_pseudo_targets(teacher_payload, config)
    del teacher_bundle, teacher_payload
    _release()

    # Student unlabeled B=32: full Adapter + Decoder z_core, never Refiner.
    unlabeled_bundle = _extract_real_bundle(dino, TS_BATCH, torch.device("cuda"))
    if unlabeled_bundle.patch_tokens[0].shape[0] != TS_BATCH:
        raise AssertionError("Student unlabeled physical batch is not fixed at 32")
    with _autocast():
        unlabeled_outputs, unlabeled_core_aux, unlabeled_refined = model(
            unlabeled_bundle.patch_tokens,
            unlabeled_bundle.cls_tokens,
            memory,
            branch="student_unlabeled",
            stage=stage,
            image_ids=None,
        )
        if unlabeled_refined is not None:
            raise AssertionError("Student unlabeled branch executed Refiner")
        student_aux = {"pseudo_refiner": None, "z_core": unlabeled_outputs[3]}
        unlabeled_loss, _ = encoder_pc_unlabeled_loss(
            unlabeled_outputs,
            student_aux,
            pseudo,
            config,
            ts_epoch=1,
        )
    if student_aux["z_core"] is not unlabeled_outputs[3]:
        raise AssertionError("Student unlabeled supervision is not outputs[3] z_core")
    scaler.scale(unlabeled_loss).backward()
    del unlabeled_bundle, unlabeled_outputs, unlabeled_core_aux
    del unlabeled_refined, student_aux, pseudo, unlabeled_loss
    _release()

    # Exactly one update after the two Student backwards, then three-module EMA.
    scaler.unscale_(optimizer)
    _assert_finite_gradients(model, "TS batch32")
    optimizer_steps = 0
    scaler.step(optimizer)
    optimizer_steps += 1
    scaler.update()
    model.update_teacher(momentum=config.ema_momentum)
    if optimizer_steps != 1:
        raise AssertionError("TS CUDA smoke did not execute exactly one optimizer step")
    if model.teacher_refiner.call_count != 1:
        raise AssertionError("Teacher must execute Refiner exactly once")
    if model.student_refiner.call_count != 1:
        raise AssertionError("Student Refiner must execute labeled branch only")
    peak = _peak_gib()
    del memory
    optimizer.zero_grad(set_to_none=True)
    _release()
    return peak


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("encoder_pc CUDA smoke requires a CUDA device")
    torch.manual_seed(20260721)
    torch.cuda.manual_seed_all(20260721)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    config = smoke_config()
    if (
        config.route_top_img_k,
        config.parent_topk,
        config.query_chunk_size,
    ) != (2, 2, 32):
        raise AssertionError("CUDA smoke capacity contract changed")
    dino, dino_weight_fingerprint = _load_frozen_dino(device)
    model = EncoderPCSmokePipeline(config).to(device)
    real_decoder = Decoder(pc_cfg=None).to(device)
    if real_decoder.pc_hbm is not None or any(
        name.startswith("pc_hbm.") for name in real_decoder.state_dict()
    ):
        raise AssertionError("CUDA smoke Decoder unexpectedly attached PC-HBM")
    model.student_head.decoder = real_decoder
    model.teacher_head.decoder = (
        copy.deepcopy(real_decoder).eval().requires_grad_(False)
    )
    decoder_calls = {"student": 0, "teacher": 0}

    def require_decoder_off(label: str):
        def hook(_module, _args, kwargs):
            if kwargs.get("pc_mode") != "off" or kwargs.get("memory") is not None:
                raise AssertionError(
                    f"{label} original Decoder did not run with PC permanently off"
                )
            decoder_calls[label] += 1

        return hook

    model.student_decoder.register_forward_pre_hook(
        require_decoder_off("student"), with_kwargs=True
    )
    model.teacher_decoder.register_forward_pre_hook(
        require_decoder_off("teacher"), with_kwargs=True
    )
    optimizer = build_encoder_pc_optimizer(
        model.student_adapter,
        model.student_decoder,
        model.student_refiner,
        decoder_warm_started=False,
    )
    # A deterministic unit initial scale keeps this single-step smoke focused
    # on model/AMP correctness instead of the GradScaler warm-down that a
    # long-running trainer would absorb over several skipped startup steps.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=1.0, growth_interval=1000
    )
    base_peak = _run_base_batch16(
        model,
        dino,
        dino_weight_fingerprint,
        config,
        optimizer,
        scaler,
    )
    ts_peak = _run_ts_batch32(
        model,
        dino,
        dino_weight_fingerprint,
        config,
        optimizer,
        scaler,
    )
    if any(parameter.grad is not None for parameter in dino.parameters()):
        raise AssertionError("CUDA smoke produced gradients for frozen DINO")
    if decoder_calls != {"student": 3, "teacher": 1}:
        raise AssertionError(f"Unexpected original Decoder call counts: {decoder_calls}")
    print(
        "[PASS] encoder_pc CUDA AMP: "
        f"Base={BASE_BATCH}, Teacher={TS_BATCH}, StudentL={TS_BATCH}, "
        f"StudentU={TS_BATCH}, route_top=2, parent_top=2, chunk=32, "
        "dino=real_frozen_vitb14, decoder=real_original_off, "
        f"base_peak={base_peak:.2f} GiB, ts_peak={ts_peak:.2f} GiB"
    )


if __name__ == "__main__":
    main()
