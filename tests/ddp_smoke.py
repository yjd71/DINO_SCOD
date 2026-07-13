"""Two-rank CPU/Gloo smoke test for staged PC-HBM DDP execution.

Run with::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        tests/ddp_smoke.py --cpu

The smoke deliberately uses precomputed DINO features.  It keeps the real
Decoder and every PC-HBM module, but replaces the four quadratic baseline
TransformerBlocks with tiny shape-compatible blocks so the distributed
contract can be exercised quickly on CPU.
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
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.decoder import Decoder
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import pc_unlabeled_loss, update_ema_module
from utils.pc_memory_runner import module_fingerprint


class _CheapTransformerBlock(nn.Module):
    """Linear-cost stand-in that preserves the Decoder token contract."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kv_scale = nn.Parameter(torch.tensor(0.125))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return self.norm(q + self.kv_scale.tanh() * kv)


class _PCHBMDDPSmokeModule(nn.Module):
    """Expose Base modes or raw teacher-only Student branches through DDP."""

    def __init__(self, config: DinoPCHBMConfig, *, teacher_only: bool = False) -> None:
        super().__init__()
        self.teacher_only = bool(teacher_only)
        self.student = Decoder(pc_cfg=None if self.teacher_only else config)
        dim = config.decoder_dim
        self.student.TransBlock_seg1 = _CheapTransformerBlock(dim)
        self.student.TransBlock_seg2 = _CheapTransformerBlock(dim)
        self.student.TransBlock_seg3 = _CheapTransformerBlock(dim)
        self.student.TransBlock_seg4 = _CheapTransformerBlock(dim)

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

        # Full uses P1/mixture while student_core intentionally stops after
        # P2-BRA and exposes z_main as its supervised output.
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
        # path.  Legacy Decoder supervision remains live in all three stages.
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
                name=f"Base/{mode} legacy Decoder",
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

        # Teacher-only TS uses a raw Student with no PC-HBM parameters and
        # find_unused_parameters=False.  Both backwards synchronize normally.
        torch.manual_seed(271828)
        raw_model = _PCHBMDDPSmokeModule(config, teacher_only=True)
        if raw_model.student.pc_hbm is not None:
            raise AssertionError("Teacher-only TS Student must be raw and contain no PC-HBM.")
        raw_ddp = DDP(raw_model, find_unused_parameters=False)
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
        unlabeled_loss, _ = pc_unlabeled_loss(
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
        _assert_allreduce_consistent(
            _parameter_checksum(raw_ddp.module), name="TS raw Student parameters"
        )

        dist.barrier()
        if rank == 0:
            print(
                "DDP PC-HBM smoke passed: 2 ranks/Gloo; "
                "Base off,parent_only,full; TS raw/off double backward and selective EMA; "
                "all-reduce checksums consistent.",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
