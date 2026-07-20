from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from Model.PC_HBM.encoder.encoder_memory import EncoderPCMemory
import Model.PC_HBM.encoder.encoder_memory_builder as builder_module
from Model.PC_HBM.encoder.encoder_memory_builder import EncoderMemoryBuilder
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.pc_memory_runner import module_fingerprint, rebuild_encoder_memory


def _config(**overrides):
    values = {
        "memory_source": "labeled_only",
        "use_unlabeled_memory_update": False,
        "memory_dim": 128,
        "value_dim": 8,
        "geometry_dim": 6,
        "token_size": 28,
        "region_names": ("fg_core", "fg_boundary", "bg_near", "bg_far"),
        "region_max_quota": (2, 2, 2, 2),
        "region_min_quota": (1, 1, 1, 1),
        "region_sampling_ratio": (1.0, 1.0, 1.0, 1.0),
        "fg_boundary_kernel": 3,
        "bg_near_kernel": 7,
        "sdf_reliability_scale": 0.15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _MemoryDataset(Dataset):
    def __init__(self, names: tuple[str, ...] = ("A.png", "B.png")) -> None:
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        image = torch.full((3, 392, 392), float(index + 1) / 10.0)
        gt = torch.zeros(1, 392, 392)
        gt[:, 98:294, 98:294] = 1.0
        if index % 2:
            gt[:, 140:252, 140:252] = 0.0
        return self.names[index], image, gt


@dataclass(frozen=True)
class _RawBundle:
    images: torch.Tensor
    raw_marker: str = "raw_dino_bundle"


class _TrapDecoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("Decoder.forward must not run during encoder memory rebuild")

    def forward_memory_features(self, *args, **kwargs):
        raise AssertionError(
            "Decoder.forward_memory_features must not run during encoder memory rebuild"
        )


class _FeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dino = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.dino.weight.fill_(0.125)
        self.dino.requires_grad_(False).eval()
        self.decoder = _TrapDecoder()
        self.bundle_calls = 0
        self.forward_calls = 0
        self.last_bundle: _RawBundle | None = None

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("Model.forward must not run during encoder memory rebuild")

    def extract_feature_bundle(self, images: torch.Tensor) -> _RawBundle:
        self.bundle_calls += 1
        bundle = _RawBundle(images=images)
        self.last_bundle = bundle
        return bundle


class _MemoryAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.memory_calls = 0
        self.forward_calls = 0
        self.last_bundle: _RawBundle | None = None

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("Adapter.forward enhancement path must not build memory")

    def forward_memory_features(self, bundle: _RawBundle) -> dict[str, torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        assert bundle.raw_marker == "raw_dino_bundle"
        self.memory_calls += 1
        self.last_bundle = bundle
        batch_size = int(bundle.images.shape[0])
        device = bundle.images.device
        dtype = bundle.images.dtype
        sample_value = bundle.images.mean(dim=(1, 2, 3)).to(dtype=dtype)
        dim_axis = torch.linspace(-0.5, 0.5, 128, device=device, dtype=dtype)
        route_base = sample_value[:, None] + dim_axis[None, :] + self.scale
        token_axis = torch.linspace(-1.0, 1.0, 784, device=device, dtype=dtype)
        tokens = (
            sample_value[:, None, None]
            + token_axis[None, :, None]
            + dim_axis[None, None, :]
            + self.scale
        )
        assert tokens.shape == (batch_size, 784, 128)
        return {
            "route_keys": route_base,
            "cls4_keys": route_base + 0.1,
            "f4_global_keys": route_base + 0.2,
            "f3_boundary_keys": route_base + 0.3,
            "f3_parent_keys": tokens,
            "f2_child_keys": tokens + 1.0,
            "f1_detail_keys": tokens + 2.0,
        }


def _loader(names: tuple[str, ...] = ("A.png", "B.png"), batch_size: int = 2):
    return DataLoader(
        _MemoryDataset(names),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def test_rebuild_uses_raw_bundle_and_ema_adapter_without_decoder() -> None:
    model = _FeatureModel()
    adapter = _MemoryAdapter()
    memory = EncoderPCMemory()

    rebuilt = rebuild_encoder_memory(
        model,
        adapter,
        _loader(),
        memory,
        "cpu",
        config=_config(),
        use_amp=False,
    )

    assert rebuilt is memory
    assert memory.is_ready()
    assert memory.route["image_ids"] == ["A.png", "B.png"]
    assert memory.num_images == 2
    assert memory.num_parents == memory.num_children == 16
    assert model.bundle_calls == 1
    assert model.forward_calls == 0
    assert adapter.memory_calls == 1
    assert adapter.forward_calls == 0
    assert adapter.last_bundle is model.last_bundle
    assert memory.compat_meta["producer_fingerprint"] == module_fingerprint(adapter)
    assert memory.compat_meta["dino_weight_fingerprint"] == module_fingerprint(
        model.dino
    )
    assert memory.compat_meta["route_source"][:2] == (
        "encoder_route_key_v1",
        "route_mlp_640_to_128_v1",
    )
    assert memory.compat_meta["route_source"][2] == (
        "block11_cls",
        "block11_f4_global",
        "block8_f3_boundary",
        "block8_f3_uncertainty",
        "block8_f3_environment",
    )
    assert memory.compat_meta["dino_checkpoint"] == (
        "weight/dinov2_vitb14_pretrain.pth"
    )
    assert memory.compat_meta["labeled_split_fingerprint"] == (
        compute_labeled_split_fingerprint(("A.png", "B.png"))
    )
    assert memory.compat_meta["producer_source"] == "ema_encoder_adapter"
    assert memory.compat_meta["labeled_image_count"] == 2


def test_builder_route_keys_are_gt_independent_and_metadata_is_tensorized(
    monkeypatch,
) -> None:
    adapter = _MemoryAdapter().eval()
    images = torch.stack((_MemoryDataset()[0][1], _MemoryDataset()[1][1]))
    with torch.inference_mode():
        features = adapter.forward_memory_features(_RawBundle(images))
    builder = EncoderMemoryBuilder(_config())
    original_build_regions = builder_module.build_pc_regions

    def checked_build_regions(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        return original_build_regions(*args, **kwargs)

    monkeypatch.setattr(builder_module, "build_pc_regions", checked_build_regions)
    gt_square = torch.stack((_MemoryDataset()[0][2], _MemoryDataset()[1][2]))
    gt_background = torch.zeros_like(gt_square)

    square_entries = builder(
        features=features,
        gt=gt_square,
        image_ids=("A.png", "B.png"),
    )
    background_entries = builder(
        features=features,
        gt=gt_background,
        image_ids=("A.png", "B.png"),
    )

    for name in ("route_keys", "cls4_keys", "f4_global_keys", "f3_boundary_keys"):
        assert square_entries["route"][name] is features[name]
        assert torch.equal(square_entries["route"][name], background_entries["route"][name])
    assert set(square_entries["parent"]) == {
        "f3_parent_keys",
        "values",
        "geometry",
        "child_ptr",
        "image_index",
        "region_id",
        "flat_index",
        "reliability",
    }
    assert set(square_entries["child"]) == {
        "f2_child_keys",
        "f1_detail_keys",
        "geometry",
        "image_index",
        "flat_index",
    }
    assert all(torch.is_tensor(value) for value in square_entries["parent"].values())
    assert all(torch.is_tensor(value) for value in square_entries["child"].values())
    assert all(
        not value.requires_grad
        for group in ("route", "parent", "child")
        for value in square_entries[group].values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    values = square_entries["parent"]["values"]
    assert values.shape[1] == 8
    assert torch.equal(values[:, 7], square_entries["parent"]["reliability"])


def test_rebuild_api_has_no_decoder_parameter_and_requires_exact_contracts() -> None:
    assert "decoder" not in inspect.signature(rebuild_encoder_memory).parameters

    with pytest.raises(AttributeError, match="extract_feature_bundle"):
        rebuild_encoder_memory(
            nn.Identity(),
            _MemoryAdapter(),
            _loader(),
            EncoderPCMemory(),
            "cpu",
            config=_config(),
            use_amp=False,
        )
    with pytest.raises(AttributeError, match="forward_memory_features"):
        rebuild_encoder_memory(
            _FeatureModel(),
            nn.Identity(),
            _loader(),
            EncoderPCMemory(),
            "cpu",
            config=_config(),
            use_amp=False,
        )


def test_rebuild_rejects_unlabeled_configuration_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="labeled_only"):
        rebuild_encoder_memory(
            _FeatureModel(),
            _MemoryAdapter(),
            _loader(),
            EncoderPCMemory(),
            "cpu",
            config=_config(memory_source="pseudo"),
            use_amp=False,
        )
    with pytest.raises(ValueError, match="repeated stable image ids"):
        rebuild_encoder_memory(
            _FeatureModel(),
            _MemoryAdapter(),
            _loader(("same.png", "same.png"), batch_size=1),
            EncoderPCMemory(),
            "cpu",
            config=_config(),
            use_amp=False,
        )


def test_rebuild_rejects_forged_producer_or_split_compatibility() -> None:
    with pytest.raises(ValueError, match="producer_fingerprint"):
        rebuild_encoder_memory(
            _FeatureModel(),
            _MemoryAdapter(),
            _loader(),
            EncoderPCMemory(),
            "cpu",
            config=_config(),
            compat_meta={"producer_fingerprint": "forged"},
            use_amp=False,
        )
    with pytest.raises(ValueError, match="labeled_split_fingerprint"):
        rebuild_encoder_memory(
            _FeatureModel(),
            _MemoryAdapter(),
            _loader(),
            EncoderPCMemory(),
            "cpu",
            config=_config(),
            compat_meta={"labeled_split_fingerprint": "forged"},
            use_amp=False,
        )
