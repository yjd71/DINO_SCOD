"""Two-rank CPU/Gloo smoke test for staged PC-HBM DDP execution.

Run with::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        tests/ddp_smoke.py --cpu

The smoke deliberately uses precomputed DINO features. It keeps the real
original Decoder and every PC-HBM module so the distributed contract exercises
the production graph.
"""

from __future__ import annotations

import argparse
from functools import wraps
import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.decoder import Decoder
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import pc_unlabeled_loss, update_ema_module
from utils.pc_memory_runner import module_fingerprint


class _PCHBMDDPSmokeModule(nn.Module):
    """Expose Base modes and both TS training designs through DDP."""

    def __init__(self, config: DinoPCHBMConfig, *, teacher_only: bool = False) -> None:
        super().__init__()
        self.teacher_only = bool(teacher_only)
        self.student = Decoder(pc_cfg=None if self.teacher_only else config)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        memory: PCMemory | None,
        *,
        pc_mode: str | None = None,
        branch: str | None = None,
        epoch: int = 13,
        query_image_ids: Sequence[str] | None = None,
    ) -> object:
        if branch == "student_labeled":
            mode = "off" if self.teacher_only else "full"
        elif branch == "student_unlabeled":
            mode = "off" if self.teacher_only else "student_core"
        elif branch is None and pc_mode is not None:
            mode = pc_mode
        else:
            raise ValueError(
                "Expected a Base pc_mode or a supported TS branch, got "
                f"pc_mode={pc_mode!r}, branch={branch!r}."
            )

        outputs, aux = self.student(
            features,
            memory=memory,
            pc_mode=mode,
            epoch=epoch,
            return_aux=True,
            query_image_ids=query_image_ids,
        )

        # Return the real nested Student output through DDP for the unlabeled
        # branch.  With find_unused_parameters=True, _DDPSink independently
        # clones the repeated outputs[3]/aux['z_main'] references; the loss
        # contract must therefore not depend on shared storage identity.
        if branch == "student_unlabeled":
            if not aux["mixture_skipped"]:
                raise RuntimeError("Student unlabeled path must skip mixture.")
            if not self.teacher_only and aux["z_final"] is not None:
                raise RuntimeError("student_core must not return z_final.")
            return outputs, aux

        # Baseline supervision is present in every stage.  Keeping this loss
        # inside forward lets DDP discover the exact autograd graph returned by
        # each staged invocation.
        loss = sum(output.float().square().mean() for output in outputs)
        if mode != "off":
            if not aux["pc_active"]:
                raise RuntimeError(
                    f"PC-HBM unexpectedly fell back in {mode}: "
                    f"{aux.get('fallback_reason')}"
                )
            pc_aux = aux["pc_hbm"]
            loss = loss + 0.1 * pc_aux["B3"].float().square().mean()
            loss = loss + 0.01 * _masked_score_mean(
                pc_aux["parent_ret"]["top_parent_scores"],
                pc_aux["parent_ret"]["top_parent_valid"],
            )
            loss = loss + 0.01 * _masked_score_mean(
                pc_aux["route"]["top_img_scores"],
                pc_aux["route"]["top_img_valid"],
            )

        # Full uses P1/mixture while student_core executes P1-PRA, skips the
        # mixture, and exposes z_main as its supervised output.
        if mode == "full":
            loss = loss + aux["z_final"].float().square().mean()
        return loss


def _masked_score_mean(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_float = valid.to(dtype=scores.dtype)
    return (scores * valid_float).sum() / valid_float.sum().clamp_min(1.0)


def _smoke_config() -> DinoPCHBMConfig:
    # One selected token per hierarchy is enough to exercise every module and
    # keeps the two-process CPU smoke comfortably below full training cost.
    return DinoPCHBMConfig(
        route_top_img_k=1,
        parent_topk=2,
        p3_min_tokens=1,
        p3_max_tokens=1,
        p2_min_tokens=1,
        p2_max_tokens=1,
        p1_min_tokens=1,
        p1_max_tokens=1,
        query_chunk_size=16,
    )


def _force_dense_refinement_queries(decoder: nn.Module) -> None:
    """Select every P2/P1 query so the joint smoke has a real valid overlap."""

    if decoder.pc_hbm is None:
        raise AssertionError("Dense refinement queries require an attached PC-HBM engine.")
    for boundary_head, token_count in (
        (decoder.pc_hbm.p2_bra.boundary_head, 28 * 28),
        (decoder.pc_hbm.p1_pra.boundary_head, 98 * 98),
    ):
        boundary_head.top_ratio = 1.0
        boundary_head.min_tokens = token_count
        boundary_head.max_tokens = token_count


def _build_ready_memory(config: DinoPCHBMConfig) -> PCMemory:
    """Build the same labeled-only, ready CPU/FP16 memory on every rank."""

    generator = torch.Generator(device="cpu").manual_seed(20260712)
    image_ids = ("memory-a", "memory-b")
    parent_count = 8
    route = {
        name: torch.randn(
            len(image_ids), config.memory_dim, generator=generator
        )
        for name in (
            "x3_global",
            "x3_boundary",
            "x3_uncertain",
            "x3_bg_near",
            "x3_environment",
        )
    }
    route["route_embed"] = torch.randn(
        len(image_ids), config.memory_dim, generator=generator
    )
    route["img_ids"] = list(image_ids)
    metadata = [
        {
            "image_id": image_ids[index % len(image_ids)],
            "region": "fg_boundary",
        }
        for index in range(parent_count)
    ]
    memory = PCMemory(config=config)
    memory.append(
        {
            "source": "labeled_only",
            "route": route,
            "parent": {
                "p3_keys": torch.randn(
                    parent_count, config.memory_dim, generator=generator
                ),
                "p3_values": 0.1
                * torch.randn(
                    parent_count, config.value_dim, generator=generator
                ),
                "p3_geometry": 0.1
                * torch.randn(
                    parent_count, config.geometry_dim, generator=generator
                ),
                "child_ptr": torch.arange(parent_count),
                "parent_meta": metadata,
            },
            "child": {
                "p2_child_keys": torch.randn(
                    parent_count, config.memory_dim, generator=generator
                ),
                "p2_child_geo": 0.1
                * torch.randn(
                    parent_count, config.geometry_dim, generator=generator
                ),
                "child_meta": metadata,
            },
        }
    )
    memory.finalize(
        compat_meta=config.expected_memory_meta(
            producer_fingerprint="ddp-smoke"
        )
    )
    if not memory.is_ready():
        raise RuntimeError("Failed to build a ready PC-HBM memory.")
    return memory


def _features(rank: int, step: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        9187 + 1009 * rank + step
    )
    return [
        torch.randn(1, 28 * 28, 768, generator=generator)
        for _ in range(4)
    ]


def _tensor_checksum(value: object) -> torch.Tensor:
    """Return two moments for nested tensors without changing their device."""

    first = torch.zeros((), dtype=torch.float64)
    second = torch.zeros((), dtype=torch.float64)

    def visit(item: object) -> None:
        nonlocal first, second
        if torch.is_tensor(item):
            tensor = item.detach().to(device="cpu", dtype=torch.float64)
            first = first + tensor.sum()
            second = second + tensor.square().sum()
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    visit(value)
    return torch.stack((first, second))


def _parameter_checksum(module: nn.Module) -> torch.Tensor:
    return _tensor_checksum([parameter for parameter in module.parameters()])


def _gradient_checksum(module: nn.Module) -> torch.Tensor:
    return _tensor_checksum(
        [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
    )


def _assert_finite_gradient_group(
    module: _PCHBMDDPSmokeModule,
    *,
    pc_parameters: bool,
    expected: bool,
    name: str,
) -> None:
    gradients = [
        parameter.grad
        for parameter_name, parameter in module.student.named_parameters()
        if parameter_name.startswith("pc_hbm.") == pc_parameters
        and parameter.grad is not None
    ]
    if bool(gradients) != expected:
        state = "receive" if expected else "not receive"
        raise AssertionError(f"{name} must {state} gradients in this Base stage.")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{name} received non-finite gradients.")


def _assert_joint_p1_finite_gradients(
    module: _PCHBMDDPSmokeModule,
) -> None:
    """Require finite gradients for the P1 attention and all raw heads."""

    prefixes = {
        "q_proj": "pc_hbm.p1_pra.q_proj.",
        "k_proj": "pc_hbm.p1_pra.k_proj.",
        "v_proj": "pc_hbm.p1_pra.v_proj.",
        "gate": "pc_hbm.p1_pra.g_head.",
        "residual": "pc_hbm.p1_pra.r_head.",
        "offset": "pc_hbm.p1_pra.o_head.",
        "suppression": "pc_hbm.p1_pra.sup_head.",
    }
    named_parameters = dict(module.student.named_parameters())
    for name, prefix in prefixes.items():
        gradients = [
            parameter.grad
            for parameter_name, parameter in named_parameters.items()
            if parameter_name.startswith(prefix) and parameter.grad is not None
        ]
        if not gradients:
            raise AssertionError(
                f"Joint student_core P1 {name} parameters received no gradient."
            )
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError(
                f"Joint student_core P1 {name} received non-finite gradients."
            )
        if name not in {"q_proj", "k_proj", "v_proj"} and not any(
            gradient.detach().abs().sum().item() > 0.0
            for gradient in gradients
        ):
            raise AssertionError(
                f"Joint P1 {name} distillation gradient was identically zero."
            )


def _clone_joint_teacher_features(aux: Mapping[str, object]) -> dict[str, object]:
    """Clone real teacher_pseudo P3/P2/P1 targets outside inference mode."""

    distill = aux.get("distill_features")
    p1 = aux.get("p1_pra")
    if not isinstance(distill, Mapping) or not isinstance(p1, Mapping):
        raise AssertionError("Teacher teacher_pseudo must expose P3/P2 and P1 targets.")

    p3 = distill.get("p3_corr")
    p2 = distill.get("p2_refined")
    if not torch.is_tensor(p3) or not torch.is_tensor(p2):
        raise AssertionError("Teacher teacher_pseudo is missing corrected P3/P2 targets.")

    p1_fields = (
        "B1",
        "G1_raw_map",
        "R1_map",
        "O1_map",
        "R_sup_map",
        "valid1_map",
    )
    missing = [name for name in p1_fields if not torch.is_tensor(p1.get(name))]
    if missing:
        raise AssertionError(f"Teacher teacher_pseudo is missing P1 targets: {missing}")

    # The clone operations intentionally happen after inference_mode exits so
    # these targets are ordinary tensors that can safely participate in the
    # Student autograd graph.  Small deterministic offsets keep every P1 loss
    # live even though the zero-initialized Teacher and Student start equal.
    p1_targets = {
        name: p1[name].detach().clone()
        for name in p1_fields
    }
    p1_targets["B1"] = (0.9 * p1_targets["B1"] + 0.05).clamp(0.0, 1.0)
    for name in ("G1_raw_map", "R1_map", "O1_map", "R_sup_map"):
        p1_targets[name] = p1_targets[name] + 0.01

    return {
        "p3_corr": p3.detach().clone() + 0.01,
        "p2_refined": p2.detach().clone() + 0.01,
        "p1": p1_targets,
    }


def _assert_allreduce_consistent(
    local_checksum: torch.Tensor,
    *,
    name: str,
    atol: float = 1e-7,
) -> None:
    world_size = dist.get_world_size()
    reduced_mean = local_checksum.clone()
    dist.all_reduce(reduced_mean, op=dist.ReduceOp.SUM)
    reduced_mean /= world_size
    error = (local_checksum - reduced_mean).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if error.item() > atol:
        raise AssertionError(
            f"{name} differs across ranks: max checksum error={error.item():.3e}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the required CPU/Gloo smoke (the only supported mode).",
    )
    return parser.parse_args()


def _force_worker_non_libuv_store() -> None:
    """Patch only torch's worker rendezvous reference on non-libuv Windows."""

    if sys.platform != "win32":
        return
    rendezvous_module = importlib.import_module("torch.distributed.rendezvous")
    native_tcp_store = rendezvous_module.TCPStore

    @wraps(native_tcp_store)
    def tcp_store_without_libuv(*args, **kwargs):
        kwargs.setdefault("use_libuv", False)
        return native_tcp_store(*args, **kwargs)

    rendezvous_module.TCPStore = tcp_store_without_libuv


def main() -> None:
    args = _parse_args()
    if not args.cpu:
        raise SystemExit("This smoke test requires the explicit --cpu flag.")

    # Worker-side env:// rendezvous must use the same non-libuv TCPStore as the
    # guarded Windows torchrun parent compatibility hook.
    os.environ["USE_LIBUV"] = "0"
    _force_worker_non_libuv_store()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"Expected exactly 2 ranks, got {world_size}.")

        config = _smoke_config()
        memory = _build_ready_memory(config)
        _assert_allreduce_consistent(
            _tensor_checksum(memory.state_dict()), name="PCMemory"
        )

        torch.manual_seed(314159)
        model = _PCHBMDDPSmokeModule(config)
        ddp = DDP(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-5)

        # Base uses find_unused_parameters=True because the unused set changes
        # by stage: off leaves every PC parameter unused; parent_only activates
        # only the parent/B3 subset; full activates the complete correction
        # path. Original-Decoder supervision remains live in all three stages.
        # The changing reducer graph must not become stale or hang either rank.
        for step, mode in enumerate(("off", "parent_only", "full")):
            optimizer.zero_grad(set_to_none=True)
            loss = ddp(
                _features(rank, step),
                memory if mode != "off" else None,
                pc_mode=mode,
                epoch=13,
                query_image_ids=[f"query-rank-{rank}"],
            )
            if not torch.isfinite(loss):
                raise AssertionError(f"Non-finite Base loss in mode={mode}.")
            loss.backward()
            _assert_finite_gradient_group(
                ddp.module,
                pc_parameters=False,
                expected=True,
                name=f"Base/{mode} original Decoder",
            )
            _assert_finite_gradient_group(
                ddp.module,
                pc_parameters=True,
                expected=mode != "off",
                name=f"Base/{mode} PC-HBM",
            )
            optimizer.step()
            _assert_allreduce_consistent(
                _parameter_checksum(ddp.module), name=f"Base/{mode} parameters"
            )

        # Teacher-only TS uses a raw Student with no PC-HBM parameters.  Its
        # unlabeled objective intentionally leaves a small Decoder subset
        # outside that backward graph, so DDP must discover unused parameters
        # on every invocation.
        # Both backwards still synchronize and accumulate normally.
        torch.manual_seed(271828)
        raw_model = _PCHBMDDPSmokeModule(config, teacher_only=True)
        if raw_model.student.pc_hbm is not None:
            raise AssertionError("Teacher-only TS Student must be raw and contain no PC-HBM.")
        raw_ddp = DDP(raw_model, find_unused_parameters=True)
        raw_optimizer = torch.optim.SGD(raw_ddp.parameters(), lr=1e-5)
        torch.manual_seed(161803)
        frozen_teacher = _PCHBMDDPSmokeModule(config).student
        frozen_teacher.eval().requires_grad_(False)
        teacher_pc_fingerprint = module_fingerprint(frozen_teacher.pc_hbm)

        # One zero_grad and one optimizer step intentionally bracket two normal
        # synchronized backwards.  There is no no_sync(): labeled raw/off and
        # unlabeled raw/off distillation must both all-reduce and accumulate.
        raw_optimizer.zero_grad(set_to_none=True)
        labeled_loss = raw_ddp(
            _features(rank, 10),
            None,
            branch="student_labeled",
            epoch=13,
            query_image_ids=[f"labeled-query-rank-{rank}"],
        )
        labeled_loss.backward()

        unlabeled_outputs, unlabeled_aux = raw_ddp(
            _features(rank, 11),
            None,
            branch="student_unlabeled",
            epoch=13,
            query_image_ids=[f"unlabeled-query-rank-{rank}"],
        )
        pseudo = torch.sigmoid(unlabeled_aux["z_main"].detach())
        confidence = torch.ones_like(pseudo)
        unlabeled_loss, unlabeled_log = pc_unlabeled_loss(
            unlabeled_outputs,
            unlabeled_aux,
            pseudo,
            confidence,
            epoch=1,
            config=config,
            teacher_features={
                "p3_corr": unlabeled_aux["features"]["p3"].detach() + 0.01,
                "p2_refined": unlabeled_aux["features"]["p2"].detach() + 0.01,
            },
        )
        if not torch.isfinite(unlabeled_loss):
            raise AssertionError("Non-finite TS Student unlabeled loss.")
        for name in ("L_u_hard", "L_u_hard_weighted", "hard_ramp"):
            if name not in unlabeled_log or not torch.isfinite(unlabeled_log[name]):
                raise AssertionError(f"Missing or non-finite TS hard metric: {name}")
        unlabeled_loss.backward()
        _assert_allreduce_consistent(
            _gradient_checksum(raw_ddp.module), name="TS accumulated gradients"
        )
        raw_optimizer.step()
        update_ema_module(
            raw_ddp.module.student,
            frozen_teacher,
            momentum=0.995,
            shared_only=True,
            exclude_prefixes=("pc_hbm.",),
        )
        if module_fingerprint(frozen_teacher.pc_hbm) != teacher_pc_fingerprint:
            raise AssertionError("Selective EMA modified frozen Teacher PC-HBM state.")

        # The production failure surfaced at the next labeled forward, when
        # the reducer checked whether the preceding unlabeled reduction had
        # completed. Exercise that exact unlabeled-backward -> next-forward
        # transition instead of ending the raw-Student smoke one iteration too
        # early.
        raw_optimizer.zero_grad(set_to_none=True)
        next_labeled_loss = raw_ddp(
            _features(rank, 12),
            None,
            branch="student_labeled",
            epoch=13,
            query_image_ids=[f"next-labeled-query-rank-{rank}"],
        )
        if not torch.isfinite(next_labeled_loss):
            raise AssertionError("Non-finite next-iteration TS labeled loss.")
        next_labeled_loss.backward()
        raw_optimizer.step()
        update_ema_module(
            raw_ddp.module.student,
            frozen_teacher,
            momentum=0.995,
            shared_only=True,
            exclude_prefixes=("pc_hbm.",),
        )
        if module_fingerprint(frozen_teacher.pc_hbm) != teacher_pc_fingerprint:
            raise AssertionError("Selective EMA modified frozen Teacher PC-HBM state.")
        _assert_allreduce_consistent(
            _parameter_checksum(raw_ddp.module), name="TS raw Student parameters"
        )

        # Joint TS keeps PC-HBM on both Student and Teacher.  Its labeled
        # branch runs full including P1/mixture; the unlabeled branch runs the
        # extended student_core through P1-PRA but stops before mixture.  The
        # two synchronized backwards share one optimizer step.
        torch.manual_seed(141421)
        joint_model = _PCHBMDDPSmokeModule(config)
        _force_dense_refinement_queries(joint_model.student)
        joint_ddp = DDP(joint_model, find_unused_parameters=True)
        joint_optimizer = torch.optim.SGD(joint_ddp.parameters(), lr=1e-5)
        joint_teacher = _PCHBMDDPSmokeModule(config).student
        _force_dense_refinement_queries(joint_teacher)
        joint_teacher.load_state_dict(
            joint_ddp.module.student.state_dict(), strict=True
        )
        joint_teacher.eval().requires_grad_(False)

        joint_optimizer.zero_grad(set_to_none=True)
        joint_labeled_loss = joint_ddp(
            _features(rank, 20),
            memory,
            branch="student_labeled",
            epoch=13,
            query_image_ids=[f"joint-labeled-query-rank-{rank}"],
        )
        if not torch.isfinite(joint_labeled_loss):
            raise AssertionError("Non-finite joint Student labeled/full loss.")
        joint_labeled_loss.backward()

        joint_features = _features(rank, 21)
        with torch.inference_mode():
            teacher_outputs, teacher_aux = joint_teacher(
                joint_features,
                memory=memory,
                pc_mode="teacher_pseudo",
                epoch=13,
                return_aux=True,
                query_image_ids=[f"joint-unlabeled-query-rank-{rank}"],
            )
        if len(teacher_outputs) != 5:
            raise AssertionError("Teacher teacher_pseudo changed the five-output contract.")
        if (
            teacher_aux.get("forward_mode") != "teacher_pseudo"
            or teacher_aux.get("mixture") is None
            or teacher_aux.get("z_final") is None
            or teacher_aux.get("p_final") is None
        ):
            raise AssertionError("Teacher teacher_pseudo must execute P1 and mixture.")

        pseudo_source = teacher_aux["p_final"]
        if not torch.is_tensor(pseudo_source):
            raise AssertionError("Teacher teacher_pseudo must expose probability p_final.")
        pseudo = pseudo_source.detach().clone()
        confidence = torch.ones_like(pseudo)
        teacher_features = _clone_joint_teacher_features(teacher_aux)

        joint_outputs, joint_aux = joint_ddp(
            joint_features,
            memory,
            branch="student_unlabeled",
            epoch=13,
            query_image_ids=[f"joint-unlabeled-query-rank-{rank}"],
        )
        if len(joint_outputs) != 5:
            raise AssertionError("Joint student_core changed the five-output contract.")
        if joint_aux.get("forward_mode") != "student_core":
            raise AssertionError("Joint unlabeled branch must use student_core.")
        if not isinstance(joint_aux.get("p1_pra"), Mapping):
            raise AssertionError("Joint student_core must execute P1-PRA exactly once.")
        if (
            joint_aux.get("mixture") is not None
            or not joint_aux.get("mixture_skipped")
            or joint_aux.get("z_final") is not None
            or joint_aux.get("p_final") is not None
        ):
            raise AssertionError(
                "Joint student_core must stop after P1-PRA and skip mixture outputs."
            )
        student_valid1 = joint_aux["p1_pra"].get("valid1_map")
        teacher_valid1 = teacher_features["p1"]["valid1_map"]
        if (
            not torch.is_tensor(student_valid1)
            or not torch.is_tensor(teacher_valid1)
            or not bool((student_valid1 > 0.5).any())
            or not bool((teacher_valid1 > 0.5).any())
        ):
            raise AssertionError(
                "Joint smoke requires non-empty Student/Teacher P1 valid intersections."
            )

        joint_unlabeled_loss, joint_unlabeled_log = pc_unlabeled_loss(
            joint_outputs,
            joint_aux,
            pseudo,
            confidence,
            epoch=1,
            config=config,
            teacher_features=teacher_features,
        )
        if not torch.isfinite(joint_unlabeled_loss):
            raise AssertionError("Non-finite joint Student unlabeled loss.")
        p1_log_names = (
            "L_u_feat_p1_B1",
            "L_u_feat_p1_G1",
            "L_u_feat_p1_R1",
            "L_u_feat_p1_O1",
            "L_u_feat_p1_R_sup",
            "L_u_feat_p1",
            "L_u_feature",
        )
        for name in p1_log_names:
            value = joint_unlabeled_log.get(name)
            if not torch.is_tensor(value) or not torch.isfinite(value):
                raise AssertionError(
                    f"Missing or non-finite joint P1 distillation metric: {name}"
                )
        if joint_unlabeled_log["L_u_feat_p1"].item() <= 0.0:
            raise AssertionError("Joint P1 distillation was not exercised.")

        joint_unlabeled_loss.backward()
        _assert_joint_p1_finite_gradients(joint_ddp.module)
        _assert_allreduce_consistent(
            _gradient_checksum(joint_ddp.module),
            name="TS joint accumulated gradients",
        )
        joint_optimizer.step()
        update_ema_module(
            joint_ddp.module.student,
            joint_teacher,
            momentum=0.995,
        )
        _assert_allreduce_consistent(
            _parameter_checksum(joint_ddp.module),
            name="TS joint Student parameters",
        )
        _assert_allreduce_consistent(
            _parameter_checksum(joint_teacher),
            name="TS joint Teacher EMA parameters",
        )

        dist.barrier()
        if rank == 0:
            print(
                "DDP PC-HBM smoke passed: 2 ranks/Gloo; "
                "Base off,parent_only,full; TS raw/off and joint full+student_core/P1 "
                "double backward with EMA; "
                "all-reduce checksums consistent.",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
