from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import inference as inference_module
from configs.pc_hbm_experiments import (
    apply_experiment_profile,
    experiment_profile_names,
)
from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderPCHBMAdapter,
    EncoderPCSegmentationHead,
    TeacherPseudoLabelRefiner,
)
from Model.PC_HBM.encoder.encoder_memory import (
    EncoderPCMemory,
    build_encoder_memory_compat_meta,
)
from Model.PC_HBM.training.encoder_training import EncoderPCStage
from utils.checkpoint_pc_hbm import save_encoder_pc_checkpoint
from utils.pc_memory_runner import module_fingerprint


class _TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 3)


class _TinyDecoder(nn.Module):
    decoder_architecture = "legacy_transformer"
    decoder_contract_version = 1

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 2)
        self.pc_hbm = None


class _TinyRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(2, 1, 1)


class _ArtifactModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.dino.weight.fill_(0.125)
        self.dino.requires_grad_(False).eval()
        self.encoder_pc_hbm = _TinyAdapter()
        self.decoder = _TinyDecoder()
        self.pseudo_refiner = _TinyRefiner()


def _entry() -> dict:
    reliability = torch.tensor([0.7, 0.8])
    values = torch.zeros(2, 8)
    values[:, 0] = 1.0
    values[:, 7] = reliability
    return {
        "source": "labeled_only",
        "route": {
            "route_keys": torch.randn(2, 128),
            "cls4_keys": torch.randn(2, 128),
            "f4_global_keys": torch.randn(2, 128),
            "f3_boundary_keys": torch.randn(2, 128),
            "image_ids": ["image-a", "image-b"],
        },
        "parent": {
            "f3_parent_keys": torch.randn(2, 128),
            "values": values,
            "geometry": torch.randn(2, 6),
            "child_ptr": torch.tensor([0, 1]),
            "image_index": torch.tensor([0, 1]),
            "region_id": torch.tensor([0, 1]),
            "flat_index": torch.tensor([10, 20]),
            "reliability": reliability,
        },
        "child": {
            "f2_child_keys": torch.randn(2, 128),
            "f1_detail_keys": torch.randn(2, 128),
            "geometry": torch.randn(2, 6),
            "image_index": torch.tensor([0, 1]),
            "flat_index": torch.tensor([10, 20]),
        },
    }


def _memory(producer: str, split: str) -> EncoderPCMemory:
    memory = EncoderPCMemory()
    memory.append(_entry())
    memory.finalize(
        compat_meta=build_encoder_memory_compat_meta(
            dino_weight_fingerprint=module_fingerprint(_ArtifactModel().dino),
            producer_fingerprint=producer,
            labeled_split_fingerprint=split,
        )
    )
    return memory


def _artifact(tmp_path, *, config=None, role="student", design="teacher_student"):
    config = EncoderPCHBMConfig() if config is None else config
    source = _ArtifactModel()
    producer = module_fingerprint(source.encoder_pc_hbm)
    split = "split-fingerprint-a"
    path = tmp_path / f"encoder_pc_{role}_v3.pth"
    payload = save_encoder_pc_checkpoint(
        path,
        epoch=15,
        encoder_pc_hbm=source.encoder_pc_hbm,
        decoder=source.decoder,
        pseudo_refiner=source.pseudo_refiner,
        config=config,
        model_role=role,
        training_design=design,
        artifact_meta={
            "producer_fingerprint": producer,
            "split_fingerprint": split,
            "dino_weight_fingerprint": module_fingerprint(source.dino),
        },
    )
    return path, payload, producer, split


@pytest.mark.parametrize(
    ("role", "design"),
    [("base", "two_stage"), ("student", "teacher_student")],
)
def test_strict_v3_model_and_memory_load_cross_check_fingerprints(
    tmp_path, role, design
):
    config = EncoderPCHBMConfig()
    model_path, _, producer, split = _artifact(
        tmp_path, config=config, role=role, design=design
    )
    memory_path = tmp_path / f"encoder_pc_{role}_memory_v3.pth"
    torch.save(_memory(producer, split).state_dict(), memory_path)
    target = _ArtifactModel()

    artifact = inference_module.load_encoder_pc_model_for_inference(
        model_path, target, config
    )
    loaded = inference_module.load_encoder_pc_memory_for_inference(
        memory_path,
        config,
        model_artifact=artifact,
        encoder_pc_hbm=target.encoder_pc_hbm,
        dino=target.dino,
    )

    assert loaded.is_ready()
    assert loaded.compat_meta["producer_fingerprint"] == producer
    assert loaded.compat_meta["labeled_split_fingerprint"] == split
    assert module_fingerprint(target.encoder_pc_hbm) == producer
    for group in (loaded.route, loaded.parent, loaded.child):
        for value in group.values():
            if torch.is_tensor(value) and value.is_floating_point():
                assert value.device.type == "cpu"
                assert value.dtype == torch.float16


@pytest.mark.parametrize("schema", [1, 2])
def test_formal_memory_rejects_v1_v2_and_only_diagnostic_mode_falls_back(
    tmp_path, schema
):
    config = EncoderPCHBMConfig()
    model_path, _, _, _ = _artifact(tmp_path, config=config)
    target = _ArtifactModel()
    artifact = inference_module.load_encoder_pc_model_for_inference(
        model_path, target, config
    )
    old_memory = {
        "format_version": schema,
        "schema_version": schema,
        "compat_meta": {"schema_version": schema},
    }

    with pytest.raises(RuntimeError, match="schema v3"):
        inference_module.load_encoder_pc_memory_for_inference(
            old_memory,
            config,
            model_artifact=artifact,
            encoder_pc_hbm=target.encoder_pc_hbm,
            dino=target.dino,
        )
    with pytest.warns(RuntimeWarning, match="diagnostic identity fallback"):
        assert (
            inference_module.load_encoder_pc_memory_for_inference(
                old_memory,
                config,
                model_artifact=artifact,
                encoder_pc_hbm=target.encoder_pc_hbm,
                dino=target.dino,
                diagnostic_identity_fallback=True,
            )
            is None
        )


@pytest.mark.parametrize("mismatch", ["producer", "split"])
def test_formal_memory_rejects_producer_and_split_mismatch(tmp_path, mismatch):
    config = EncoderPCHBMConfig()
    model_path, _, producer, split = _artifact(tmp_path, config=config)
    target = _ArtifactModel()
    artifact = inference_module.load_encoder_pc_model_for_inference(
        model_path, target, config
    )
    memory = _memory(
        "wrong-producer" if mismatch == "producer" else producer,
        "wrong-split" if mismatch == "split" else split,
    )

    with pytest.raises(RuntimeError, match=f"compat_mismatch:{mismatch if mismatch == 'producer' else 'labeled_split'}"):
        inference_module.load_encoder_pc_memory_for_inference(
            memory.state_dict(),
            config,
            model_artifact=artifact,
            encoder_pc_hbm=target.encoder_pc_hbm,
            dino=target.dino,
        )


def test_missing_memory_is_formal_error_and_explicit_diagnostic_fallback(tmp_path):
    config = EncoderPCHBMConfig()
    model_path, _, _, _ = _artifact(tmp_path, config=config)
    target = _ArtifactModel()
    artifact = inference_module.load_encoder_pc_model_for_inference(
        model_path, target, config
    )

    with pytest.raises(FileNotFoundError, match="was not provided"):
        inference_module.load_encoder_pc_memory_for_inference(
            None,
            config,
            model_artifact=artifact,
            encoder_pc_hbm=target.encoder_pc_hbm,
            dino=target.dino,
        )
    with pytest.warns(RuntimeWarning, match="diagnostic identity fallback"):
        assert (
            inference_module.load_encoder_pc_memory_for_inference(
                None,
                config,
                model_artifact=artifact,
                encoder_pc_hbm=target.encoder_pc_hbm,
                dino=target.dino,
                diagnostic_identity_fallback=True,
            )
            is None
        )


def test_model_loader_rejects_config_drift_and_invalid_role_design_pair(tmp_path):
    model_path, _, _, _ = _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="live contract"):
        inference_module.load_encoder_pc_model_for_inference(
            model_path,
            _ArtifactModel(),
            EncoderPCHBMConfig(boundary_token_ratio=0.1),
        )

    base_path, _, _, _ = _artifact(
        tmp_path, role="base", design="teacher_student"
    )
    with pytest.raises(RuntimeError, match="role/design mismatch"):
        inference_module.load_encoder_pc_model_for_inference(
            base_path, _ArtifactModel(), EncoderPCHBMConfig()
        )

    student_path, _, _, _ = _artifact(
        tmp_path, role="student", design="two_stage"
    )
    with pytest.raises(RuntimeError, match="role/design mismatch"):
        inference_module.load_encoder_pc_model_for_inference(
            student_path, _ArtifactModel(), EncoderPCHBMConfig()
        )


def test_model_loader_recomputes_and_rejects_forged_producer_metadata(tmp_path):
    _, payload, _, _ = _artifact(tmp_path)
    forged = dict(payload)
    forged["artifact_meta"] = {
        **payload["artifact_meta"],
        "producer_fingerprint": "forged-producer",
    }
    with pytest.raises(RuntimeError, match="loaded Adapter state"):
        inference_module.load_encoder_pc_model_for_inference(
            forged, _ArtifactModel(), EncoderPCHBMConfig()
        )


def test_model_loader_recomputes_and_rejects_dino_fingerprint(tmp_path):
    model_path, _, _, _ = _artifact(tmp_path)
    target = _ArtifactModel()
    with torch.no_grad():
        target.dino.weight.add_(0.5)
    with pytest.raises(RuntimeError, match="DINO fingerprint"):
        inference_module.load_encoder_pc_model_for_inference(
            model_path, target, EncoderPCHBMConfig()
        )


def test_encoder_cli_is_canonical_and_legacy_aliases_conflict(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference.py",
            "--experiment-profile",
            "encoder_pc",
            "--model-checkpoint",
            "model.pth",
            "--memory-checkpoint",
            "memory.pth",
        ],
    )
    args = inference_module.parse_args()
    profile = inference_module.validate_inference_args(args)
    assert profile.pc_placement == "encoder"
    assert args.decoder_checkpoint is None

    args.decoder_checkpoint = "legacy.pth"
    with pytest.raises(ValueError, match="legacy decoder-side"):
        inference_module.validate_inference_args(args)

    args.decoder_checkpoint = None
    args.memory_checkpoint = None
    with pytest.raises(ValueError, match="requires --memory-checkpoint"):
        inference_module.validate_inference_args(args)
    args.diagnostic_identity_fallback = True
    inference_module.validate_inference_args(args)


def test_legacy_cli_rejects_model_checkpoint_and_keeps_decoder_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference.py",
            "--experiment-profile",
            "legacy_off",
            "--checkpoint",
            "legacy.pth",
            "--datasets",
            "CAMO",
        ],
    )
    args = inference_module.parse_args()
    inference_module.validate_inference_args(args)
    assert args.decoder_checkpoint == "legacy.pth"

    args.model_checkpoint = "encoder-v3.pth"
    with pytest.raises(ValueError, match="reserved for encoder_pc"):
        inference_module.validate_inference_args(args)


def test_registered_encoder_ablations_are_wired_and_no_refiner_inference_profile():
    names = experiment_profile_names()
    assert "encoder_pc_f4_f3" in names
    assert "encoder_pc_no_route_loss" in names
    assert "encoder_pc_refiner_inference" not in names

    f4_f3 = EncoderPCHBMConfig()
    apply_experiment_profile(f4_f3, "encoder_pc_f4_f3")
    assert f4_f3.enable_f2_f1_propagation is False
    assert f4_f3.lambda_route == pytest.approx(0.05)
    assert EncoderPCStage.for_epoch(20, f4_f3).enable_f2_f1 is False

    no_route = EncoderPCHBMConfig()
    apply_experiment_profile(no_route, "encoder_pc_no_route_loss")
    assert no_route.enable_f2_f1_propagation is True
    assert no_route.lambda_route == 0.0


class _RecordingCoreDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pc_hbm = None
        self.calls = []

    def forward(self, features, **kwargs):
        self.calls.append(dict(kwargs))
        batch = features[0].shape[0]
        outputs = tuple(
            features[0].new_full((batch, 1, 98, 98), float(index))
            for index in range(5)
        )
        return outputs


def test_formal_head_returns_exact_z_core_and_never_calls_refiner():
    config = EncoderPCHBMConfig()
    adapter = EncoderPCHBMAdapter(config)
    decoder = _RecordingCoreDecoder()
    refiner = TeacherPseudoLabelRefiner(config)
    head = EncoderPCSegmentationHead(adapter, decoder, refiner).eval()
    producer = "adapter-for-forward"
    memory = _memory(producer, "split-forward")
    bundle = DinoFeatureBundle(
        patch_tokens=tuple(torch.zeros(1, 784, 768) for _ in range(4)),
        cls_tokens=tuple(torch.zeros(1, 768) for _ in range(4)),
    ).validate()
    refiner_calls = []
    hook = refiner.register_forward_hook(
        lambda _module, _inputs, _output: refiner_calls.append(1)
    )
    try:
        output = head(
            role="inference",
            bundle=bundle,
            memory=memory,
            return_aux=False,
        )
        diagnostic_output = head(
            role="inference",
            bundle=bundle,
            memory=None,
            allow_memory_fallback=True,
            return_aux=False,
        )
    finally:
        hook.remove()

    assert torch.equal(output, torch.full_like(output, 3.0))
    assert torch.equal(diagnostic_output, torch.full_like(diagnostic_output, 3.0))
    assert refiner_calls == []
    assert len(decoder.calls) == 2
    for call in decoder.calls:
        assert call["memory"] is None
        assert call["pc_mode"] == "off"
        assert call["return_aux"] is False
        assert call["query_image_ids"] is None


class _BenchmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_pc_hbm = nn.Identity()
        self.pseudo_refiner = nn.Identity()
        self.calls = 0

    def inference(
        self,
        images,
        memory=None,
        epoch=None,
        *,
        allow_memory_fallback=False,
    ):
        del memory, epoch, allow_memory_fallback
        self.calls += 1
        features = self.encoder_pc_hbm(images)
        return features[:, :1, :1, :1]


def test_benchmark_uses_fixed_10_50_protocol_and_reports_contract_flags():
    model = _BenchmarkModel()
    memory = SimpleNamespace(
        route={"keys": torch.zeros(2, 3, dtype=torch.float16)},
        parent={"index": torch.zeros(4, dtype=torch.int32)},
        child={},
    )
    report = inference_module.benchmark_model_inference(
        model,
        torch.zeros(2, 3, 4, 4),
        memory=memory,
    )

    assert model.calls == 60
    assert report["warmup_iterations"] == 10
    assert report["timed_iterations"] == 50
    assert report["batch_size"] == 2
    assert report["mean_ms"] > 0
    assert report["p50_ms"] > 0
    assert report["p95_ms"] > 0
    assert report["throughput_samples_per_s"] > 0
    assert report["peak_cuda_memory_bytes"] == 0
    assert report["bank_bytes"] == 2 * 3 * 2 + 4 * 4
    assert report["adapter_executed"] is True
    assert report["refiner_executed"] is False
