"""Two-rank CPU/Gloo acceptance smoke for encoder-side PC-HBM v3.

The smoke deliberately replaces only frozen-DINO extraction and the unchanged
original Decoder with inexpensive tensor doubles.  The Encoder Adapter, pseudo-label
Refiner, v3 retrieval bank, curriculum/losses, pseudo-confidence path, EMA, and
DDP reducer are production implementations.

Run with::

    python -B -m torch.distributed.run --standalone --nproc_per_node=2 \
        tests/ddp_smoke_encoder_pc.py --cpu
"""

from __future__ import annotations

import argparse
from functools import wraps
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderPCCoreResult,
    EncoderPCHBMAdapter,
    EncoderPCMemory,
    EncoderPCSegmentationHead,
    TeacherPseudoLabelRefiner,
    build_encoder_memory_compat_meta,
)
from Model.PC_HBM.encoder.teacher_pseudo_refiner import (
    teacher_pseudo_refiner_labeled_loss,
)
from Model.PC_HBM.training import (
    EncoderPCStage,
    build_encoder_pc_optimizer,
    configure_encoder_pc_stage,
    encoder_pc_labeled_loss,
    encoder_pc_unlabeled_loss,
    make_ema_encoder_adapter,
    prepare_encoder_pc_pseudo_targets,
    update_ema_encoder_adapter,
    update_ema_module,
)
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.pc_memory_runner import module_fingerprint
from test_encoder_pc_smoke_contract import SyntheticFrozenDinoContract


SMOKE_IMAGE_IDS = ("smoke-image-0", "smoke-image-1", "smoke-image-2")
_SYNTHETIC_DINO = SyntheticFrozenDinoContract()
SYNTHETIC_DINO_FINGERPRINT = module_fingerprint(_SYNTHETIC_DINO)


def smoke_config() -> EncoderPCHBMConfig:
    """Only shrink the three capacity knobs permitted by the v3 plan."""

    return EncoderPCHBMConfig(
        route_top_img_k=2,
        parent_topk=2,
        query_chunk_size=32,
    )


def synthetic_bundle(
    batch_size: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
    seed: int,
) -> DinoFeatureBundle:
    """Build a physical-B bundle from compact expanded frozen feature views."""

    del seed
    images = synthetic_rgb(batch_size, device, dtype)
    bundle = _SYNTHETIC_DINO(images, feature_dtype=dtype)
    if any(
        value.requires_grad
        for value in (*bundle.patch_tokens, *bundle.cls_tokens)
    ):
        raise AssertionError("Synthetic frozen-DINO bundle unexpectedly requires grad")
    return bundle


def synthetic_rgb(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Shape-correct 392x392 RGB view; the Decoder double never materializes it."""

    return torch.zeros(batch_size, 3, 1, 1, device=device, dtype=dtype).expand(
        batch_size, 3, 392, 392
    )


def synthetic_gt(batch_size: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(98, device=device)
    pattern = (((axis[:, None] // 7) + (axis[None, :] // 7)) % 2).float()
    return pattern.reshape(1, 1, 98, 98).expand(batch_size, -1, -1, -1)


def synthetic_memory(
    config: EncoderPCHBMConfig,
    *,
    producer_fingerprint: str,
    dino_weight_fingerprint: str = SYNTHETIC_DINO_FINGERPRINT,
) -> EncoderPCMemory:
    """Create a complete tensorized, labeled-only CPU-FP16 v3 bank."""

    generator = torch.Generator(device="cpu").manual_seed(20260721)
    image_count = len(SMOKE_IMAGE_IDS)
    # One parent per each of the two routed images gives an effective top-2
    # subbank even though the production result tensor retains top-16 padding.
    parents_per_image = 1
    parent_count = image_count * parents_per_image
    image_index = torch.arange(image_count).repeat_interleave(parents_per_image)
    reliability = torch.linspace(0.60, 0.95, parent_count)
    values = torch.zeros(parent_count, config.value_dim)
    values[torch.arange(parent_count), torch.arange(parent_count) % 4] = 1.0
    values[:, 4] = (torch.arange(parent_count) % 2 == 0).float()
    values[:, 5] = 1.0 - values[:, 4]
    values[:, 6] = torch.linspace(-0.5, 0.5, parent_count)
    values[:, 7] = reliability
    geometry = torch.zeros(parent_count, config.geometry_dim)
    geometry[:, 1] = 1.0
    geometry[:, 5] = reliability

    def normal(shape: Sequence[int]) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    memory = EncoderPCMemory()
    memory.append(
        {
            "source": "labeled_only",
            "route": {
                "route_keys": normal((image_count, config.memory_dim)),
                "cls4_keys": normal((image_count, config.memory_dim)),
                "f4_global_keys": normal((image_count, config.memory_dim)),
                "f3_boundary_keys": normal((image_count, config.memory_dim)),
                "image_ids": list(SMOKE_IMAGE_IDS),
            },
            "parent": {
                "f3_parent_keys": normal((parent_count, config.memory_dim)),
                "values": values,
                "geometry": geometry,
                "child_ptr": torch.arange(parent_count),
                "image_index": image_index,
                "region_id": torch.arange(parent_count) % 4,
                "flat_index": torch.arange(parent_count) * 11,
                "reliability": reliability,
            },
            "child": {
                "f2_child_keys": normal((parent_count, config.memory_dim)),
                "f1_detail_keys": normal((parent_count, config.memory_dim)),
                "geometry": geometry.clone(),
                "image_index": image_index.clone(),
                "flat_index": torch.arange(parent_count) * 11,
            },
        }
    )
    split_fingerprint = compute_labeled_split_fingerprint(SMOKE_IMAGE_IDS)
    memory.finalize(
        compat_meta=build_encoder_memory_compat_meta(
            dino_weight_fingerprint=dino_weight_fingerprint,
            producer_fingerprint=producer_fingerprint,
            labeled_split_fingerprint=split_fingerprint,
        )
    )
    if not memory.is_ready() or memory.num_images < 3:
        raise AssertionError("Synthetic encoder memory is not a complete v3 bank")
    for group in (memory.route, memory.parent, memory.child):
        for name, value in group.items():
            if name == "image_ids":
                continue
            if not torch.is_tensor(value) or value.device.type != "cpu":
                raise AssertionError(f"Memory field {name} is not a CPU tensor")
            if value.is_floating_point() and value.dtype != torch.float16:
                raise AssertionError(f"Memory field {name} is not CPU FP16")
    return memory


class SmokeOriginalDecoder(nn.Module):
    """Cheap original-Decoder contract double; internal PC is absent/off."""

    decoder_arch = "legacy_transformer"
    decoder_architecture = "legacy_transformer"

    def __init__(self) -> None:
        super().__init__()
        self.pc_hbm = None
        self.output_scale = nn.Parameter(torch.ones(5))
        self.output_bias = nn.Parameter(torch.linspace(-0.08, 0.08, 5))
        self.p1_scale = nn.Parameter(torch.ones(()))
        self.forward_calls = 0

    def forward(
        self,
        features: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> Any:
        self.forward_calls += 1
        if kwargs.get("pc_mode") != "off" or kwargs.get("memory") is not None:
            raise AssertionError("Encoder-side Decoder double was not called in off mode")
        if kwargs.get("query_image_ids") is not None:
            raise AssertionError("Encoder-side Decoder received retrieval image IDs")
        if len(features) != 4:
            raise AssertionError("Decoder did not receive four DINO feature levels")
        feature_maps = [
            feature.mean(dim=-1).reshape(feature.size(0), 1, 28, 28)
            for feature in features
        ]
        base = torch.stack(feature_maps, dim=0).mean(dim=0)
        base98 = F.interpolate(base, size=(98, 98), mode="bilinear", align_corners=False)
        outputs = tuple(
            base98 * self.output_scale[index] + self.output_bias[index]
            for index in range(5)
        )
        if not kwargs.get("return_aux", False):
            return outputs
        p1 = base98.expand(-1, 128, -1, -1) * self.p1_scale
        return outputs, {"features": {"p1": p1}}


class CountingTeacherPseudoLabelRefiner(TeacherPseudoLabelRefiner):
    """Production Refiner with a non-state call counter for path assertions."""

    def __init__(self, config: EncoderPCHBMConfig) -> None:
        super().__init__(config)
        self.call_count = 0

    def forward(self, *args: Any, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        self.call_count += 1
        return super().forward(*args, **kwargs)


class EncoderPCSmokePipeline(nn.Module):
    """Isomorphic Student/EMA-Teacher heads around the real v3 modules."""

    def __init__(self, config: EncoderPCHBMConfig) -> None:
        super().__init__()
        student_adapter = EncoderPCHBMAdapter(config)
        student_decoder = SmokeOriginalDecoder()
        student_refiner = CountingTeacherPseudoLabelRefiner(config)
        self.student_head = EncoderPCSegmentationHead(
            student_adapter, student_decoder, student_refiner
        )
        self.teacher_head = EncoderPCSegmentationHead(
            make_ema_encoder_adapter(student_adapter),
            _frozen_copy(student_decoder),
            _frozen_copy(student_refiner),
        )

    @property
    def student_adapter(self) -> EncoderPCHBMAdapter:
        return self.student_head.adapter

    @property
    def student_decoder(self) -> SmokeOriginalDecoder:
        return self.student_head.decoder

    @property
    def student_refiner(self) -> CountingTeacherPseudoLabelRefiner:
        return self.student_head.pseudo_refiner

    @property
    def teacher_adapter(self) -> EncoderPCHBMAdapter:
        return self.teacher_head.adapter

    @property
    def teacher_decoder(self) -> SmokeOriginalDecoder:
        return self.teacher_head.decoder

    @property
    def teacher_refiner(self) -> CountingTeacherPseudoLabelRefiner:
        return self.teacher_head.pseudo_refiner

    def forward(
        self,
        patches: Sequence[torch.Tensor],
        cls_tokens: Sequence[torch.Tensor],
        memory: EncoderPCMemory | None,
        *,
        branch: str,
        stage: EncoderPCStage,
        image_ids: Sequence[str] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any], Any]:
        bundle = DinoFeatureBundle(tuple(patches), tuple(cls_tokens)).validate()
        if branch in {"base", "student_labeled"}:
            core = self.student_head(
                role="labeled_core",
                bundle=bundle,
                memory=memory,
                mode=stage.mode,
                stage=stage.adapter_flags(require_same_image_positive=True),
                epoch=stage.epoch,
                query_image_ids=image_ids,
                return_aux=True,
            )
            refined = None
            if stage.enable_refiner or branch == "student_labeled":
                refined = self.student_head(
                    role="labeled_refiner",
                    core_result=core,
                    epoch=stage.epoch,
                )
            return core.outputs, core.aux, refined
        if branch == "student_unlabeled":
            core = self.student_head(
                role="student_core",
                bundle=bundle,
                memory=memory,
                stage=stage.adapter_flags(require_same_image_positive=False),
                epoch=stage.epoch,
                return_aux=True,
            )
            return core.outputs, core.aux, None
        raise ValueError(f"unsupported smoke branch: {branch}")

    @torch.no_grad()
    def teacher_pseudo(
        self,
        bundle: DinoFeatureBundle,
        memory: EncoderPCMemory,
        stage: EncoderPCStage,
    ) -> Mapping[str, Any]:
        payload = self.teacher_head(
            role="teacher_pseudo",
            bundle=bundle,
            memory=memory,
            stage=stage.adapter_flags(require_same_image_positive=False),
            epoch=stage.epoch,
            return_aux=True,
        )
        return {
            **payload,
            "encoder_pc_hbm": payload["aux"]["encoder_pc_hbm"],
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student, teacher in (
            (self.student_adapter, self.teacher_adapter),
            (self.student_decoder, self.teacher_decoder),
            (self.student_refiner, self.teacher_refiner),
        ):
            update_ema_module(student, teacher, momentum=momentum)


def _frozen_copy(module: nn.Module) -> Any:
    import copy

    result = copy.deepcopy(module).eval()
    result.requires_grad_(False)
    return result


def _assert_stage_gradients(model: EncoderPCSmokePipeline, stage: EncoderPCStage) -> None:
    expected = {
        "bootstrap": True,
        "router": stage.enable_route_parent,
        "verifier": stage.enable_verification,
        "route_context": stage.enable_verification,
        "injector": stage.enable_f4_f3,
        "propagation": stage.enable_f2_f1,
        "decoder": True,
        "refiner": stage.enable_refiner,
    }
    modules = {
        "bootstrap": model.student_adapter.bootstrap,
        "router": model.student_adapter.router,
        "verifier": model.student_adapter.verifier,
        "route_context": model.student_adapter.route_context,
        "injector": model.student_adapter.injector,
        "propagation": model.student_adapter.propagation,
        "decoder": model.student_decoder,
        "refiner": model.student_refiner,
    }
    for name, module in modules.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if bool(gradients) != bool(expected[name]):
            raise AssertionError(
                f"stage {stage.epoch} {name}: gradients={bool(gradients)}, "
                f"expected={expected[name]}"
            )
        if any(not torch.isfinite(value).all() for value in gradients):
            raise AssertionError(f"stage {stage.epoch} {name} has non-finite gradients")


def _tensor_checksum(module: nn.Module) -> torch.Tensor:
    first = torch.zeros((), dtype=torch.float64)
    second = torch.zeros((), dtype=torch.float64)
    for value in module.state_dict().values():
        if torch.is_tensor(value):
            cpu = value.detach().to(device="cpu", dtype=torch.float64)
            first += cpu.sum()
            second += cpu.square().sum()
    return torch.stack((first, second))


def _assert_rank_consistent(value: torch.Tensor, name: str, atol: float = 1.0e-8) -> None:
    local = value.to(dtype=torch.float64)
    average = local.clone()
    dist.all_reduce(average)
    average /= dist.get_world_size()
    error = (local - average).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if float(error) > atol:
        raise AssertionError(f"{name} differs across ranks by {float(error):.3e}")


def _query_ids(batch_size: int, rank: int) -> list[str]:
    return [SMOKE_IMAGE_IDS[(rank + index) % len(SMOKE_IMAGE_IDS)] for index in range(batch_size)]


def _smoke_field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container[name]
    return getattr(container, name)


def smoke_labeled_loss(
    outputs: Sequence[torch.Tensor],
    aux: Mapping[str, Any],
    gt: torch.Tensor,
    config: EncoderPCHBMConfig,
    stage: EncoderPCStage,
) -> torch.Tensor:
    """Run the same staged labeled objective used by the Base trainer."""

    loss, _ = encoder_pc_labeled_loss(outputs, aux, gt, config, stage)
    return loss


def _run_base_curriculum(
    ddp: DDP,
    config: EncoderPCHBMConfig,
    rank: int,
) -> None:
    model: EncoderPCSmokePipeline = ddp.module
    optimizer = build_encoder_pc_optimizer(
        model.student_adapter,
        model.student_decoder,
        model.student_refiner,
        decoder_warm_started=False,
    )
    memory_ema = make_ema_encoder_adapter(model.student_adapter)
    expected = {
        1: "bootstrap",
        6: "parent_only",
        11: "parent_child_f3",
        16: "hierarchical_full",
        21: "hierarchical_refiner",
    }
    for step, epoch in enumerate(expected):
        stage = EncoderPCStage.for_epoch(epoch, config)
        if stage.name != expected[epoch]:
            raise AssertionError(f"wrong stage at epoch {epoch}: {stage.name}")
        configure_encoder_pc_stage(
            model.student_adapter, model.student_decoder, model.student_refiner, stage
        )
        memory = None
        if stage.enable_route_parent:
            producer = module_fingerprint(memory_ema)
            memory = synthetic_memory(config, producer_fingerprint=producer)
            if memory.compat_meta["producer_fingerprint"] != producer:
                raise AssertionError("Base memory producer fingerprint mismatch")
        bundle = synthetic_bundle(
            1, torch.device("cpu"), dtype=torch.float32, seed=1000 + step + rank
        )
        gt = synthetic_gt(1, torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)
        outputs, aux, refined = ddp(
            bundle.patch_tokens,
            bundle.cls_tokens,
            memory,
            branch="base",
            stage=stage,
            image_ids=_query_ids(1, rank),
        )
        loss = smoke_labeled_loss(outputs, aux, gt, config, stage)
        if stage.enable_refiner:
            refiner_loss, _ = teacher_pseudo_refiner_labeled_loss(refined, gt, config)
            loss = loss + refiner_loss
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite Base loss at epoch {epoch}")
        loss.backward()
        _assert_stage_gradients(model, stage)
        optimizer.step()
        update_ema_encoder_adapter(
            memory_ema,
            model.student_adapter,
            decay=config.memory_adapter_ema_decay,
        )
        del bundle, gt, outputs, aux, refined, loss, memory
    _assert_rank_consistent(_tensor_checksum(model.student_head), "Base student")


def _run_ts_single_step(
    ddp: DDP,
    config: EncoderPCHBMConfig,
    rank: int,
) -> None:
    model: EncoderPCSmokePipeline = ddp.module
    full_stage = EncoderPCStage.for_epoch(config.final_epoch, config)
    configure_encoder_pc_stage(
        model.student_adapter,
        model.student_decoder,
        model.student_refiner,
        full_stage,
    )
    model.update_teacher(momentum=0.0)
    producer = module_fingerprint(model.teacher_adapter)
    memory = synthetic_memory(config, producer_fingerprint=producer)
    if memory.compat_meta["producer_fingerprint"] != producer:
        raise AssertionError("TS memory producer fingerprint mismatch")
    if memory.compat_meta["labeled_split_fingerprint"] != compute_labeled_split_fingerprint(
        SMOKE_IMAGE_IDS
    ):
        raise AssertionError("TS memory labeled split fingerprint mismatch")

    model.student_refiner.call_count = 0
    model.teacher_refiner.call_count = 0
    optimizer = build_encoder_pc_optimizer(
        model.student_adapter,
        model.student_decoder,
        model.student_refiner,
        decoder_warm_started=True,
    )
    optimizer.zero_grad(set_to_none=True)
    device = torch.device("cpu")

    labeled_bundle = synthetic_bundle(1, device, dtype=torch.float32, seed=3000 + rank)
    labeled_gt = synthetic_gt(1, device)
    labeled_outputs, labeled_aux, labeled_refined = ddp(
        labeled_bundle.patch_tokens,
        labeled_bundle.cls_tokens,
        memory,
        branch="student_labeled",
        stage=full_stage,
        image_ids=_query_ids(1, rank),
    )
    labeled_core_loss = smoke_labeled_loss(
        labeled_outputs, labeled_aux, labeled_gt, config, full_stage
    )
    labeled_refiner_loss, _ = teacher_pseudo_refiner_labeled_loss(
        labeled_refined, labeled_gt, config
    )
    (labeled_core_loss + labeled_refiner_loss).backward()
    del labeled_bundle, labeled_outputs, labeled_aux, labeled_refined
    del labeled_gt, labeled_core_loss, labeled_refiner_loss

    unlabeled_bundle = synthetic_bundle(1, device, dtype=torch.float32, seed=4000 + rank)
    with torch.inference_mode():
        teacher_payload = model.teacher_pseudo(unlabeled_bundle, memory, full_stage)
    pseudo = prepare_encoder_pc_pseudo_targets(teacher_payload, config)
    del teacher_payload
    unlabeled_outputs, unlabeled_core_aux, unlabeled_refined = ddp(
        unlabeled_bundle.patch_tokens,
        unlabeled_bundle.cls_tokens,
        memory,
        branch="student_unlabeled",
        stage=full_stage,
        image_ids=None,
    )
    if unlabeled_refined is not None:
        raise AssertionError("Student unlabeled branch executed the Refiner")
    student_aux = {"pseudo_refiner": None, "z_core": unlabeled_outputs[3]}
    unlabeled_loss, _ = encoder_pc_unlabeled_loss(
        unlabeled_outputs, student_aux, pseudo, config, ts_epoch=1
    )
    unlabeled_loss.backward()
    del unlabeled_bundle, unlabeled_outputs, unlabeled_core_aux
    del unlabeled_refined, student_aux, pseudo, unlabeled_loss

    optimizer_steps = 0
    optimizer.step()
    optimizer_steps += 1
    model.update_teacher(momentum=config.ema_momentum)
    if optimizer_steps != 1:
        raise AssertionError("TS smoke did not execute exactly one optimizer step")
    if model.teacher_refiner.call_count != 1:
        raise AssertionError("EMA Teacher must execute Refiner exactly once")
    if model.student_refiner.call_count != 1:
        raise AssertionError("Student Refiner must execute only on labeled data")
    _assert_rank_consistent(_tensor_checksum(model.student_head), "TS student")
    _assert_rank_consistent(_tensor_checksum(model.teacher_head), "TS teacher")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the required two-rank CPU/Gloo smoke (the only supported mode).",
    )
    return parser.parse_args()


def _force_worker_non_libuv_store() -> None:
    """Use PyTorch's reliable non-libuv worker rendezvous on Windows."""

    if sys.platform != "win32":
        return
    rendezvous_module = importlib.import_module("torch.distributed.rendezvous")
    native_tcp_store = rendezvous_module.TCPStore

    @wraps(native_tcp_store)
    def tcp_store_without_libuv(*args: Any, **kwargs: Any):
        kwargs.setdefault("use_libuv", False)
        return native_tcp_store(*args, **kwargs)

    rendezvous_module.TCPStore = tcp_store_without_libuv


def main() -> None:
    args = _parse_args()
    if not args.cpu:
        raise SystemExit("This smoke requires the explicit --cpu flag")
    os.environ["USE_LIBUV"] = "0"
    _force_worker_non_libuv_store()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise RuntimeError("encoder_pc CPU smoke requires exactly two ranks")
        torch.manual_seed(20260721)
        config = smoke_config()
        model = EncoderPCSmokePipeline(config)
        ddp = DDP(model, find_unused_parameters=True)
        if not ddp.find_unused_parameters:
            raise AssertionError("CPU smoke must use find_unused_parameters=True")
        _run_base_curriculum(ddp, config, rank)
        _run_ts_single_step(ddp, config, rank)
        dist.barrier()
        if rank == 0:
            print(
                "[PASS] encoder_pc CPU DDP: ranks=2, backend=gloo, "
                "stages=1/6/11/16/21, TS_steps=1, route_images=3"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
