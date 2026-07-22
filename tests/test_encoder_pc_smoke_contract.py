"""Shared, resource-bounded contracts for encoder-PC v3 smoke tests.

The standalone DDP/CUDA programs import the helpers in this module.  The
expensive research components under test (Adapter, memory retrieval, refiner,
pseudo confidence/loss and EMA) are production implementations.  Only frozen
DINO feature extraction and the unchanged original Decoder are represented by
strict shape-compatible doubles so fixed physical batches remain runnable on
the reference 12 GiB GPU and on two CPU/Gloo ranks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    encoder_pc_labeled_loss,
    encoder_pc_unlabeled_loss,
)
from utils.pc_memory_runner import module_fingerprint
from utils.trainer_base_model_encoder_pc import EncoderPCHBMTrainer


BASE_PHYSICAL_BATCH = 16
TS_PHYSICAL_BATCH = 32
SMOKE_ROUTE_IMAGES = 3
SMOKE_PARENTS_PER_IMAGE = 2
SMOKE_PRODUCER = "encoder-pc-smoke-producer"
SMOKE_SPLIT = "encoder-pc-smoke-split"


def smoke_config() -> EncoderPCHBMConfig:
    """Shrink only retrieval capacity; all architecture dimensions stay fixed."""

    return EncoderPCHBMConfig(
        route_top_img_k=2,
        parent_topk=2,
        query_chunk_size=32,
    )


class SyntheticFrozenDinoContract(nn.Module):
    """Frozen DINO contract double retaining all four 784x768 token levels."""

    def __init__(self, seed: int = 20260721) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for level in range(4):
            self.register_buffer(
                f"patch_{level}",
                torch.randn(1, 784, 768, generator=generator) * 0.02,
                persistent=True,
            )
            self.register_buffer(
                f"cls_{level}",
                torch.randn(1, 768, generator=generator) * 0.02,
                persistent=True,
            )
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> DinoFeatureBundle:
        if images.ndim != 4 or images.shape[1:] != (3, 392, 392):
            raise ValueError("Synthetic DINO requires images [B,3,392,392]")
        dtype = images.dtype if feature_dtype is None else feature_dtype
        patches = tuple(
            getattr(self, f"patch_{level}")
            .to(device=images.device, dtype=dtype)
            .expand(images.size(0), -1, -1)
            for level in range(4)
        )
        cls = tuple(
            getattr(self, f"cls_{level}")
            .to(device=images.device, dtype=dtype)
            .expand(images.size(0), -1)
            for level in range(4)
        )
        return DinoFeatureBundle(patches, cls).validate()


class SmokeOriginalDecoderContract(nn.Module):
    """Small five-output Decoder double that enforces the encoder-PC boundary."""

    decoder_arch = "legacy_transformer"

    def __init__(self) -> None:
        super().__init__()
        self.pc_hbm = None
        self.p1_projection = nn.Conv2d(1, 128, kernel_size=1)
        self.core_head = nn.Conv2d(128, 1, kernel_size=1)
        self.side_bias = nn.Parameter(torch.linspace(-0.10, 0.10, 5))
        nn.init.normal_(self.p1_projection.weight, std=0.02)
        nn.init.zeros_(self.p1_projection.bias)
        nn.init.normal_(self.core_head.weight, std=0.02)
        nn.init.constant_(self.core_head.bias, 1.20)
        self.call_count = 0
        self.last_batch_size = 0
        self.last_pc_mode: str | None = None
        self.last_memory_was_none = False

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        memory=None,
        pc_mode: str = "off",
        epoch: int | None = None,
        return_aux: bool = False,
        query_image_ids=None,
    ):
        del epoch
        if memory is not None or pc_mode != "off" or query_image_ids is not None:
            raise AssertionError(
                "encoder-PC Decoder must receive memory=None, pc_mode='off', "
                "query_image_ids=None"
            )
        if len(features) != 4:
            raise ValueError("Decoder contract requires four DINO feature levels")
        batch = features[0].size(0)
        for feature in features:
            if feature.shape != (batch, 784, 768):
                raise ValueError("Decoder features must remain [B,784,768]")
        token_map = torch.stack(
            [feature.mean(dim=-1) for feature in features], dim=0
        ).mean(dim=0).reshape(batch, 1, 28, 28)
        base_98 = F.interpolate(
            token_map, size=(98, 98), mode="bilinear", align_corners=False
        )
        p1 = self.p1_projection(base_98)
        core = self.core_head(p1)
        outputs = tuple(core + self.side_bias[index] for index in range(5))

        self.call_count += 1
        self.last_batch_size = batch
        self.last_pc_mode = pc_mode
        self.last_memory_was_none = memory is None
        if return_aux:
            return outputs, {"features": {"p1": p1}}
        return outputs


class CountingTeacherPseudoLabelRefiner(TeacherPseudoLabelRefiner):
    """Real refiner with a side-effect-free call counter for role assertions."""

    def __init__(self, config: EncoderPCHBMConfig) -> None:
        super().__init__(config)
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        return super().forward(*args, **kwargs)


class EncoderPCSmokeSystem(nn.Module):
    """Production Adapter/refiner/loss graph around the lightweight Decoder."""

    def __init__(self, config: EncoderPCHBMConfig) -> None:
        super().__init__()
        adapter = EncoderPCHBMAdapter(config)
        decoder = SmokeOriginalDecoderContract()
        refiner = CountingTeacherPseudoLabelRefiner(config)
        self.config = config
        self.head = EncoderPCSegmentationHead(adapter, decoder, refiner)
        self.last_unlabeled_used_output3 = False

    @property
    def adapter(self) -> EncoderPCHBMAdapter:
        return self.head.adapter

    @property
    def decoder(self) -> SmokeOriginalDecoderContract:
        return self.head.decoder

    @property
    def refiner(self) -> CountingTeacherPseudoLabelRefiner:
        return self.head.pseudo_refiner

    def teacher_copy(self) -> "EncoderPCSmokeSystem":
        teacher = copy.deepcopy(self).eval().requires_grad_(False)
        teacher.refiner.call_count = 0
        teacher.decoder.call_count = 0
        return teacher

    def forward(
        self,
        *,
        role: str,
        bundle: DinoFeatureBundle,
        memory: EncoderPCMemory | None,
        stage: EncoderPCStage | None = None,
        gt: torch.Tensor | None = None,
        pseudo: Mapping[str, object] | None = None,
        query_image_ids: Sequence[object] | None = None,
        ts_epoch: int = 1,
    ):
        if role in {"base_labeled", "student_labeled"}:
            if stage is None or gt is None:
                raise ValueError("labeled smoke roles require stage and GT")
            core = self.head(
                role="labeled_core",
                bundle=bundle,
                memory=memory,
                mode=stage.mode,
                stage=stage.adapter_flags(require_same_image_positive=True),
                epoch=stage.epoch,
                query_image_ids=query_image_ids,
                return_aux=True,
            )
            if not isinstance(core, EncoderPCCoreResult):
                raise RuntimeError("labeled core returned an invalid contract")
            core_loss, _ = encoder_pc_labeled_loss(
                core.outputs, core.aux, gt, self.config, stage
            )
            run_refiner = role == "student_labeled" or stage.enable_refiner
            if run_refiner:
                refined = self.head(
                    role="labeled_refiner",
                    core_result=core,
                    epoch=stage.epoch,
                )
                refiner_loss, _ = teacher_pseudo_refiner_labeled_loss(
                    refined, gt, self.config
                )
                core_loss = core_loss + refiner_loss
            return core_loss

        if role == "teacher_pseudo":
            payload = self.head(
                role="teacher_pseudo",
                bundle=bundle,
                memory=memory,
                stage=(stage.adapter_flags(False) if stage is not None else None),
                epoch=self.config.final_epoch + int(ts_epoch),
                return_aux=True,
            )
            encoder_aux = payload["aux"].get("encoder_pc_hbm")
            if not isinstance(encoder_aux, Mapping):
                raise RuntimeError("Teacher smoke payload lacks encoder evidence")
            return {**payload, "encoder_pc_hbm": encoder_aux}

        if role == "student_unlabeled":
            if pseudo is None:
                raise ValueError("student_unlabeled requires pseudo targets")
            core = self.head(
                role="student_core",
                bundle=bundle,
                memory=memory,
                stage=(stage.adapter_flags(False) if stage is not None else None),
                epoch=self.config.final_epoch + int(ts_epoch),
                return_aux=True,
            )
            if not isinstance(core, EncoderPCCoreResult):
                raise RuntimeError("student core returned an invalid contract")
            aux = dict(core.aux)
            aux["z_core"] = core.z_core
            aux["pseudo_refiner"] = None
            self.last_unlabeled_used_output3 = core.z_core is core.outputs[3]
            loss, _ = encoder_pc_unlabeled_loss(
                core.outputs, aux, pseudo, self.config, int(ts_epoch)
            )
            return loss
        raise ValueError(f"unsupported smoke role: {role!r}")


def build_smoke_memory(
    *,
    producer_fingerprint: str = SMOKE_PRODUCER,
    split_fingerprint: str = SMOKE_SPLIT,
    dino_weight_fingerprint: str | None = None,
) -> EncoderPCMemory:
    """Create a ready CPU-FP16 schema-v3 bank with two post-self candidates."""

    generator = torch.Generator(device="cpu").manual_seed(20260722)
    image_count = SMOKE_ROUTE_IMAGES
    parent_count = image_count * SMOKE_PARENTS_PER_IMAGE
    image_index = torch.arange(image_count).repeat_interleave(
        SMOKE_PARENTS_PER_IMAGE
    )
    reliability = torch.linspace(0.60, 0.95, parent_count)
    values = torch.randn(parent_count, 8, generator=generator) * 0.1
    values[:, 7] = reliability
    geometry = torch.zeros(parent_count, 6)
    geometry[:, 1] = 1.0
    geometry[:, 5] = reliability
    memory = EncoderPCMemory()
    memory.append(
        {
            "source": "labeled_only",
            "route": {
                "route_keys": torch.randn(image_count, 128, generator=generator),
                "cls4_keys": torch.randn(image_count, 128, generator=generator),
                "f4_global_keys": torch.randn(image_count, 128, generator=generator),
                "f3_boundary_keys": torch.randn(image_count, 128, generator=generator),
                "image_ids": [f"smoke-image-{index}" for index in range(image_count)],
            },
            "parent": {
                "f3_parent_keys": torch.randn(parent_count, 128, generator=generator),
                "values": values,
                "geometry": geometry,
                "child_ptr": torch.arange(parent_count),
                "image_index": image_index,
                "region_id": torch.arange(parent_count) % 4,
                "flat_index": torch.arange(parent_count) % 784,
                "reliability": reliability,
            },
            "child": {
                "f2_child_keys": torch.randn(parent_count, 128, generator=generator),
                "f1_detail_keys": torch.randn(parent_count, 128, generator=generator),
                "geometry": geometry.clone(),
                "image_index": image_index.clone(),
                "flat_index": torch.arange(parent_count) % 784,
            },
        }
    )
    memory.finalize(
        compat_meta=build_encoder_memory_compat_meta(
            dino_weight_fingerprint=(
                module_fingerprint(SyntheticFrozenDinoContract())
                if dino_weight_fingerprint is None
                else str(dino_weight_fingerprint)
            ),
            producer_fingerprint=producer_fingerprint,
            labeled_split_fingerprint=split_fingerprint,
        )
    )
    return memory


def smoke_images(
    batch_size: int,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    base = torch.linspace(-1.0, 1.0, 392, device=device, dtype=dtype)
    image = base.view(1, 1, 1, 392).expand(1, 3, 392, 392)
    return image.expand(int(batch_size), -1, -1, -1)


def smoke_gt(
    batch_size: int,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    axis = torch.arange(98, device=device)
    mask = ((axis[:, None] + axis[None, :]) % 17 < 8).to(dtype=dtype)
    return mask.view(1, 1, 98, 98).expand(int(batch_size), -1, -1, -1)


def labeled_ids(batch_size: int) -> list[str]:
    return [f"smoke-image-{index % SMOKE_ROUTE_IMAGES}" for index in range(batch_size)]


def assert_finite_gradients(module: nn.Module, *, require_any: bool = True) -> None:
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if require_any and not gradients:
        raise AssertionError(f"{type(module).__name__} received no gradients")
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{type(module).__name__} received non-finite gradients")


def test_smoke_capacity_changes_only_permitted_retrieval_fields() -> None:
    config = smoke_config()
    assert (config.input_size, config.token_size, config.output_size) == (392, 28, 98)
    assert (config.encoder_dim, config.decoder_dim, config.memory_dim) == (768, 128, 128)
    assert (config.route_top_img_k, config.parent_topk, config.query_chunk_size) == (2, 2, 32)
    assert (config.boundary_min_tokens, config.boundary_max_tokens) == (32, 128)


def test_synthetic_dino_and_decoder_keep_full_shape_contracts() -> None:
    dino = SyntheticFrozenDinoContract()
    images = smoke_images(2, "cpu")
    bundle = dino(images)
    assert all(value.shape == (2, 784, 768) for value in bundle.patch_tokens)
    assert all(value.shape == (2, 768) for value in bundle.cls_tokens)
    assert all(not value.requires_grad for value in (*bundle.patch_tokens, *bundle.cls_tokens))

    decoder = SmokeOriginalDecoderContract()
    outputs, aux = decoder(
        bundle.patch_tokens,
        memory=None,
        pc_mode="off",
        return_aux=True,
    )
    assert len(outputs) == 5
    assert all(value.shape == (2, 1, 98, 98) for value in outputs)
    assert aux["features"]["p1"].shape == (2, 128, 98, 98)
    assert decoder.pc_hbm is None
    assert not any(name.startswith("pc_hbm.") for name in decoder.state_dict())


def test_smoke_memory_has_two_candidates_after_labeled_self_exclusion() -> None:
    memory = build_smoke_memory()
    assert memory.is_ready()
    assert memory.num_images == 3
    assert all(
        tensor.device.type == "cpu" and tensor.dtype == torch.float16
        for group in (memory.route, memory.parent, memory.child)
        for tensor in group.values()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    )
    result = EncoderPCHBMAdapter(smoke_config()).router.route(
        memory.route["route_keys"][:1].float(),
        memory,
        query_image_ids=["smoke-image-0"],
        top_img_k=2,
        require_same_image_positive=True,
    )
    assert result["top_img_valid"].sum().item() == 2
    assert result["route_valid"].item()
    assert "smoke-image-0" not in result["top_img_ids"][0]


def test_physical_batch_constants_are_not_configurable() -> None:
    assert BASE_PHYSICAL_BATCH == 16
    assert TS_PHYSICAL_BATCH == 32


def test_base_artifact_metadata_carries_live_frozen_dino_fingerprint() -> None:
    dino = nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
    trainer = object.__new__(EncoderPCHBMTrainer)
    trainer.model_without_ddp = SimpleNamespace(dino=dino)
    trainer.split_state = {"fingerprint": "smoke-split"}
    trainer.cfg = SimpleNamespace(
        baseline_fingerprint="baseline",
        initialization_source="scratch",
    )

    metadata = trainer._artifact_meta(producer_fingerprint="adapter")

    assert metadata["dino_weight_fingerprint"] == module_fingerprint(dino)
    assert metadata["producer_fingerprint"] == "adapter"


def test_system_uses_real_adapter_refiner_and_skips_unlabeled_refiner() -> None:
    config = smoke_config()
    system = EncoderPCSmokeSystem(config)
    assert isinstance(system.adapter, EncoderPCHBMAdapter)
    assert isinstance(system.refiner, TeacherPseudoLabelRefiner)
    stage = EncoderPCStage.for_epoch(21, config)
    memory = build_smoke_memory()
    images = smoke_images(1, "cpu")
    bundle = SyntheticFrozenDinoContract()(images)
    gt = smoke_gt(1, "cpu")
    labeled_loss = system(
        role="student_labeled",
        bundle=bundle,
        memory=memory,
        stage=stage,
        gt=gt,
        query_image_ids=labeled_ids(1),
    )
    assert torch.isfinite(labeled_loss)
    assert system.refiner.call_count == 1
    teacher = system.teacher_copy()
    with torch.inference_mode():
        payload = teacher(
            role="teacher_pseudo",
            bundle=bundle,
            memory=memory,
            stage=stage,
        )
    from Model.PC_HBM.training import prepare_encoder_pc_pseudo_targets

    pseudo = prepare_encoder_pc_pseudo_targets(payload, config)
    before = system.refiner.call_count
    unlabeled_loss = system(
        role="student_unlabeled",
        bundle=bundle,
        memory=memory,
        stage=stage,
        pseudo=pseudo,
    )
    assert torch.isfinite(unlabeled_loss)
    assert system.refiner.call_count == before
    assert system.last_unlabeled_used_output3
    assert system.decoder.last_pc_mode == "off"
    assert system.decoder.last_memory_was_none


def test_decoder_contract_rejects_any_pc_argument() -> None:
    decoder = SmokeOriginalDecoderContract()
    images = smoke_images(1, "cpu")
    bundle = SyntheticFrozenDinoContract()(images)
    with pytest.raises(AssertionError, match="memory=None"):
        decoder(bundle.patch_tokens, memory=object(), pc_mode="off")
