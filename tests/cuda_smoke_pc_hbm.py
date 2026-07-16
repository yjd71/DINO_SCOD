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
from Model.decoder import Decoder, build_decoder
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import (
    decoder_base_loss,
    pc_hbm_labeled_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
)


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
        image_rgb = model.prepare_rgb(images)
    del images
    return tuple(feature.detach() for feature in features), image_rgb.detach()


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
        for name in ("L_base", "L_B3", "L_gate", "L_boundary"):
            value = metrics.get(name)
            if not torch.is_tensor(value) or not torch.isfinite(value):
                raise AssertionError(f"Missing or non-finite real labeled metric: {name}")
        if not bool(metrics["L_base"] > 0):
            raise AssertionError("Real joint labeled smoke did not activate L_base")
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


def assert_decoder_group_received_gradients(
    decoder: Decoder,
    *,
    pc_parameters: bool,
    name: str,
) -> None:
    gradients = [
        parameter.grad
        for parameter_name, parameter in decoder.named_parameters()
        if parameter_name.startswith("pc_hbm.") == pc_parameters
        and parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError(f"{name} received no gradient from the joint labeled loss")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{name} received non-finite gradients")


def assert_p1_distillation_gradients(decoder: Decoder) -> None:
    """Require every trainable P1-PRA output family to receive a finite grad."""

    p1 = decoder.pc_hbm.p1_pra
    groups = {
        "P1 query projection": p1.q_proj,
        "P1 key projection": p1.k_proj,
        "P1 value projection": p1.v_proj,
        "P1 gate head": p1.g_head,
        "P1 residual head": p1.r_head,
        "P1 offset head": p1.o_head,
        "P1 suppression head": p1.sup_head,
    }
    for name, module in groups.items():
        assert_module_received_gradients(module, name)


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
    features, image_rgb = extract_features(model, batch_size, config)
    query_ids = [f"{label}-{index}" for index in range(batch_size)]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, aux = decoder(
            features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode=pc_mode,
            epoch=epoch,
            return_aux=True,
            query_image_ids=query_ids if pc_mode == "full" else None,
        )
        if not aux.get("pc_active", False):
            raise AssertionError(f"{label}: PC path unexpectedly inactive: {aux.get('fallback_reason')}")
        if pc_mode == "student_core" and not aux.get("mixture_skipped", False):
            raise AssertionError("Student core must skip mixture")
        if pc_mode == "full":
            if not isinstance(aux.get("p1_pra"), dict):
                raise AssertionError(f"{label}: full mode did not execute P1-PRA")
            if not isinstance(aux.get("mixture"), dict):
                raise AssertionError(f"{label}: full mode did not execute mixture")
            if not all(torch.is_tensor(aux.get(name)) for name in ("z_final", "p_final")):
                raise AssertionError(f"{label}: full mode is missing final prediction tensors")
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
        assert_decoder_group_received_gradients(
            decoder,
            pc_parameters=False,
            name="Base legacy Decoder",
        )
        assert_decoder_group_received_gradients(
            decoder,
            pc_parameters=True,
            name="Base PC-HBM",
        )
        assert_module_received_gradients(decoder.pc_hbm.boundary3, "B3 boundary head")
        assert_module_received_gradients(decoder.pc_hbm.gate_mlp, "PC gate MLP")
    peak = peak_gib()
    decoder.zero_grad(set_to_none=True)
    del loss, metrics, outputs, aux, features, image_rgb
    release()
    print(f"[PASS] {label}: batch={batch_size}, mode={pc_mode}, peak={peak:.2f} GiB")
    return peak


def run_teacher_scenario(
    model: BaseModel,
    memory: PCMemory,
    config: DinoPCHBMConfig,
) -> float:
    teacher = build_decoder(pc_cfg=config)
    teacher.load_state_dict(model.decoder.state_dict(), strict=True)
    teacher.requires_grad_(False).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    features, image_rgb = extract_features(model, BATCH_TEACHER, config)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _, aux = teacher(
            features,
            image_rgb=image_rgb,
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
    del probability, aux, features, image_rgb, teacher
    release()
    print(
        f"[PASS] teacher inference: batch={BATCH_TEACHER}, "
        f"mode=teacher_pseudo, peak={peak:.2f} GiB"
    )
    return peak


def _raw_student_from_enhancer(model: BaseModel) -> Decoder:
    student = build_decoder(pc_cfg=model.pc_cfg, attach_pc=False)
    student.load_state_dict(
        {
            name: value
            for name, value in model.decoder.state_dict().items()
            if not name.startswith("pc_hbm.")
        },
        strict=True,
    )
    return student.cuda()


def _joint_student_from_enhancer(
    model: BaseModel,
    config: DinoPCHBMConfig,
) -> Decoder:
    student = build_decoder(pc_cfg=config)
    student.load_state_dict(model.decoder.state_dict(), strict=True)
    return student.cuda()


def _clone_joint_teacher_target_aux(aux: dict[str, Any]) -> dict[str, Any]:
    """Mirror the joint Trainer target schema outside ``inference_mode``."""

    pc = aux.get("pc_hbm", {}) or {}
    mixture = aux.get("mixture", {}) or {}
    distill = aux.get("distill_features", {}) or {}
    p1 = aux.get("p1_pra", {}) or {}

    def clone_tensor(value: Any, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise AssertionError(f"Joint Teacher aux is missing {name}")
        cloned = value.detach().clone()
        if cloned.is_inference():
            raise AssertionError(f"Joint Teacher target {name} remained an inference tensor")
        if not torch.isfinite(cloned).all():
            raise AssertionError(f"Joint Teacher target {name} contains non-finite values")
        return cloned

    return {
        "p_final": clone_tensor(aux.get("p_final"), "p_final"),
        "z_main": clone_tensor(aux.get("z_main"), "z_main"),
        "pc_hbm": {
            "C23_map": clone_tensor(pc.get("C23_map"), "pc_hbm.C23_map"),
            "route_entropy_norm": clone_tensor(
                pc.get("route_entropy_norm"), "pc_hbm.route_entropy_norm"
            ),
        },
        "mixture": {"pi": clone_tensor(mixture.get("pi"), "mixture.pi")},
        "distill_features": {
            "p3_corr": clone_tensor(
                distill.get("p3_corr"), "distill_features.p3_corr"
            ),
            "p2_refined": clone_tensor(
                distill.get("p2_refined"), "distill_features.p2_refined"
            ),
            "p1": {
                name: clone_tensor(p1.get(name), f"p1_pra.{name}")
                for name in (
                    "B1",
                    "G1_raw_map",
                    "R1_map",
                    "O1_map",
                    "R_sup_map",
                    "valid1_map",
                )
            },
        },
    }


def _prime_joint_student_p1_heads(student: Decoder) -> None:
    """Emulate a warmed P1 enhancer so gradients propagate through q/k/v."""

    p1 = student.pc_hbm.p1_pra
    with torch.no_grad():
        for index, head in enumerate((p1.g_head, p1.r_head, p1.o_head, p1.sup_head), 1):
            head.weight.normal_(mean=0.0, std=1.0e-2)
            if head.bias is not None:
                head.bias.fill_(float(index) * 1.0e-2)


def _assert_student_core_p1_contract(outputs, aux: dict[str, Any]) -> None:
    if aux.get("forward_mode") != "student_core":
        raise AssertionError("Joint unlabeled Student did not run student_core")
    if not isinstance(aux.get("p1_pra"), dict):
        raise AssertionError("student_core did not execute P1-PRA")
    if aux.get("mixture") is not None or not aux.get("mixture_skipped", False):
        raise AssertionError("student_core must skip Adaptive Mixture")
    if aux.get("z_final") is not None or aux.get("p_final") is not None:
        raise AssertionError("student_core main supervision must not expose z_final/p_final")
    if outputs[3].shape != aux["z_main"].shape:
        raise AssertionError("student_core outputs[3] must remain the z_main prediction")
    for branch, fields in (
        ("pc_hbm", ("p3_corr",)),
        ("p2_bra", ("p2_refined",)),
        (
            "p1_pra",
            ("B1", "G1_raw_map", "R1_map", "O1_map", "R_sup_map", "valid1_map"),
        ),
    ):
        values = aux.get(branch)
        if not isinstance(values, dict):
            raise AssertionError(f"student_core aux is missing {branch}")
        missing = [name for name in fields if not torch.is_tensor(values.get(name))]
        if missing:
            raise AssertionError(f"student_core aux {branch} is missing {missing}")


def run_raw_student_labeled_scenario(
    model: BaseModel,
    config: DinoPCHBMConfig,
) -> float:
    student = _raw_student_from_enhancer(model).train()
    torch.cuda.reset_peak_memory_stats()
    features, image_rgb = extract_features(model, BATCH_STUDENT_LABELED, config)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, aux = student(
            features, image_rgb=image_rgb, pc_mode="off", return_aux=True
        )
        if aux.get("pc_active", True):
            raise AssertionError("Raw labeled Student unexpectedly activated PC-HBM")
        gt = (torch.rand_like(aux["z_main"], dtype=torch.float32) > 0.5).float()
        loss = decoder_base_loss(outputs, aux, gt, config)
    loss.backward()
    assert_finite_gradients(student)
    peak = peak_gib()
    del loss, gt, outputs, aux, features, image_rgb, student
    release()
    print(
        f"[PASS] student labeled raw: batch={BATCH_STUDENT_LABELED}, "
        f"mode=off, peak={peak:.2f} GiB"
    )
    return peak


def run_raw_student_unlabeled_scenario(
    model: BaseModel,
    memory: PCMemory,
    config: DinoPCHBMConfig,
) -> float:
    teacher = build_decoder(pc_cfg=config)
    teacher.load_state_dict(model.decoder.state_dict(), strict=True)
    teacher.requires_grad_(False).eval().cuda()
    student = _raw_student_from_enhancer(model).train()
    torch.cuda.reset_peak_memory_stats()
    features, image_rgb = extract_features(model, BATCH_STUDENT_UNLABELED, config)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _, teacher_aux = teacher(
            features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode="teacher_pseudo",
            epoch=30,
            return_aux=True,
        )
    pseudo = prepare_pseudo_targets(teacher_aux, config, strict=True)
    for name in ("hard_target", "hard_valid", "hard_weight"):
        if name not in pseudo:
            raise AssertionError(f"Teacher-only pseudo targets are missing {name}")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, student_aux = student(
            features, image_rgb=image_rgb, pc_mode="off", return_aux=True
        )
        loss, metrics = pc_unlabeled_loss(
            outputs,
            student_aux,
            pseudo["p_soft"],
            pseudo["confidence"],
            epoch=1,
            config=config,
            teacher_features=pseudo["distill_features"],
        )
    loss.backward()
    assert_finite_gradients(student)
    for name in (
        "L_u_soft",
        "L_u_hard",
        "L_u_hard_weighted",
        "hard_ramp",
        "hard_valid_ratio",
        "L_u_feat_p3",
        "L_u_feat_p2",
    ):
        if name not in metrics or not torch.isfinite(metrics[name]):
            raise AssertionError(f"Missing or non-finite teacher-only metric: {name}")
    peak = peak_gib()
    del loss, metrics, outputs, student_aux, pseudo, teacher_aux, features
    del image_rgb, teacher, student
    release()
    print(
        f"[PASS] student unlabeled raw: batch={BATCH_STUDENT_UNLABELED}, "
        f"mode=off+hard+feature_distill, peak={peak:.2f} GiB"
    )
    return peak


def run_joint_student_labeled_scenario(
    model: BaseModel,
    memory: PCMemory,
    config: DinoPCHBMConfig,
) -> float:
    student = _joint_student_from_enhancer(model, config)
    try:
        return run_training_scenario(
            "student labeled joint",
            student,
            model,
            memory,
            config,
            batch_size=BATCH_STUDENT_LABELED,
            pc_mode="full",
            epoch=30,
        )
    finally:
        del student
        release()


def run_joint_student_unlabeled_scenario(
    model: BaseModel,
    memory: PCMemory,
    config: DinoPCHBMConfig,
) -> float:
    """Run corrected-to-corrected P3/P2/P1 distillation at physical batch 32."""

    teacher = build_decoder(pc_cfg=config)
    teacher.load_state_dict(model.decoder.state_dict(), strict=True)
    teacher.requires_grad_(False).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    features, image_rgb = extract_features(model, BATCH_STUDENT_UNLABELED, config)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _, teacher_aux = teacher(
            features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode="teacher_pseudo",
            epoch=30,
            return_aux=True,
        )
        if not isinstance(teacher_aux.get("p1_pra"), dict):
            raise AssertionError("Joint Teacher did not execute P1-PRA")
        if not isinstance(teacher_aux.get("mixture"), dict):
            raise AssertionError("Joint Teacher did not execute Adaptive Mixture")

    # Clone only after leaving inference_mode, exactly as the real Trainer does.
    teacher_target_aux = _clone_joint_teacher_target_aux(teacher_aux)
    del teacher_aux, teacher
    release()
    pseudo = prepare_pseudo_targets(teacher_target_aux, config, strict=True)
    del teacher_target_aux
    confidence = pseudo["confidence"]
    if not torch.isfinite(confidence).all():
        raise AssertionError("Joint Teacher confidence contains non-finite values")
    if not bool((confidence > 0).any()):
        raise AssertionError("Joint Teacher confidence has no positive pixels")
    teacher_features = pseudo.get("distill_features")
    if not isinstance(teacher_features, dict) or not isinstance(
        teacher_features.get("p1"), dict
    ):
        raise AssertionError("Joint pseudo target is missing nested teacher_features['p1']")

    student = _joint_student_from_enhancer(model, config).train()
    _prime_joint_student_p1_heads(student)
    student.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs, student_aux = student(
            features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode="student_core",
            epoch=30,
            return_aux=True,
        )
        if not student_aux.get("pc_active", False):
            raise AssertionError(
                "Joint Student PC path unexpectedly inactive: "
                f"{student_aux.get('fallback_reason')}"
            )
        _assert_student_core_p1_contract(outputs, student_aux)
        loss, metrics = pc_unlabeled_loss(
            outputs,
            student_aux,
            pseudo["p_soft"],
            confidence,
            epoch=1,
            config=config,
            teacher_features=teacher_features,
        )
    if not torch.isfinite(loss):
        raise AssertionError("Joint Student unlabeled loss is non-finite")
    loss.backward()
    assert_finite_gradients(student)
    assert_p1_distillation_gradients(student)
    for name in (
        "L_u_soft",
        "L_u_hard",
        "L_u_side",
        "L_u_feat_p3",
        "L_u_feat_p2",
        "L_u_feat_p1_B1",
        "L_u_feat_p1_G1",
        "L_u_feat_p1_R1",
        "L_u_feat_p1_O1",
        "L_u_feat_p1_R_sup",
        "L_u_feat_p1",
        "L_u_feature",
        "pseudo_conf_mean",
        "pseudo_conf_max",
    ):
        value = metrics.get(name)
        if not torch.is_tensor(value) or not torch.isfinite(value):
            raise AssertionError(f"Missing or non-finite joint unlabeled metric: {name}")
    if not bool(metrics["L_u_feat_p1"] > 0):
        raise AssertionError("Joint P1 distillation did not produce a positive loss")

    peak = peak_gib()
    del loss, metrics, outputs, student_aux, teacher_features, confidence, pseudo
    del features, image_rgb, student
    release()
    print(
        f"[PASS] student unlabeled joint: batch={BATCH_STUDENT_UNLABELED}, "
        f"mode=student_core+P1/no-mixture, peak={peak:.2f} GiB"
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
        "student_labeled_raw_b32": run_raw_student_labeled_scenario(model, config),
        "student_unlabeled_raw_b32": run_raw_student_unlabeled_scenario(
            model, memory, config
        ),
        "student_labeled_joint_b32": run_joint_student_labeled_scenario(
            model, memory, config
        ),
        "student_unlabeled_joint_b32": run_joint_student_unlabeled_scenario(
            model, memory, config
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
