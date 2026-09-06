from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import inference
from configs.pc_hbm_dino_config import DinoPCHBMConfig


class FakeDataset:
    def __init__(self, **kwargs):
        self.samples = [
            (
                torch.zeros(3, 8, 8),
                torch.zeros(1, 5, 7),
                "sample.png",
                torch.zeros(3, 8, 8),
            )
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def inference(self, image, memory=None, epoch=None, disable_pc=False):
        self.calls.append((memory, epoch, disable_pc))
        return torch.zeros(image.shape[0], 1, 8, 8)


def _cfg():
    return SimpleNamespace(
        device="cpu",
        test_size=8,
        test_CAMO_imgs="images",
        test_CAMO_masks="masks",
    )


def test_inference_with_memory_forwards_strict_pc_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "TestDataset", FakeDataset)
    saved = []
    monkeypatch.setattr(
        inference.cv2,
        "imwrite",
        lambda path, value: saved.append((path, value.copy())) or True,
    )
    model = FakeModel()
    memory = object()
    inference.inference(
        ["CAMO"],
        model,
        _cfg(),
        str(tmp_path),
        memory=memory,
        epoch=12,
        batch_size=1,
    )
    assert model.calls == [(memory, 12, False)]
    assert saved[0][1].dtype == np.uint8
    assert np.all(saved[0][1] == 127)

def test_missing_memory_automatically_uses_pc_off_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "TestDataset", FakeDataset)
    monkeypatch.setattr(inference.cv2, "imwrite", lambda path, value: True)
    model = FakeModel()
    inference.inference(
        ["CAMO"],
        model,
        _cfg(),
        str(tmp_path),
        memory=None,
    )
    assert model.calls[-1] == (None, 30, True)


@pytest.mark.parametrize(
    "with_memory,disable_pc", [(False, False), (True, False), (False, True)]
)
def test_inference_resizes_to_checkpoint_input_and_preserves_output_size(
    tmp_path, with_memory, disable_pc
):
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    assert inference.cv2.imwrite(
        str(image_root / "sample.jpg"), np.zeros((5, 7, 3), dtype=np.uint8)
    )
    assert inference.cv2.imwrite(
        str(mask_root / "sample.png"), np.zeros((5, 7), dtype=np.uint8)
    )

    class ShapeCheckingModel(FakeModel):
        pc_cfg = DinoPCHBMConfig(input_size=518, token_size=37, output_size=129)

        def inference(self, image, memory=None, epoch=None, disable_pc=False):
            assert image.shape[-2:] == (518, 518)
            return super().inference(image, memory, epoch, disable_pc)

    cfg = _cfg()
    cfg.test_size = 392
    cfg.test_CAMO_imgs = str(image_root)
    cfg.test_CAMO_masks = str(mask_root)
    model = ShapeCheckingModel()
    memory = object() if with_memory else None
    pred_root = tmp_path / "predictions"
    inference.inference(
        ["CAMO"], model, cfg, str(pred_root), memory=memory, disable_pc=disable_pc
    )

    assert model.calls == [(memory, 30, disable_pc or not with_memory)]
    prediction = inference.cv2.imread(
        str(pred_root / "CAMO" / "sample.png"), inference.cv2.IMREAD_GRAYSCALE
    )
    assert prediction is not None and prediction.shape == (5, 7)


def test_inference_cli_contract_rejects_ambiguous_memory_modes():
    assert inference.validate_inference_args(
        SimpleNamespace(disable_pc=False, memory_checkpoint=None)
    ) is True
    assert inference.validate_inference_args(
        SimpleNamespace(disable_pc=False, memory_checkpoint="memory.pth")
    ) is False
    assert inference.validate_inference_args(
        SimpleNamespace(disable_pc=True, memory_checkpoint=None)
    ) is True
    with pytest.raises(ValueError, match="cannot be combined"):
        inference.validate_inference_args(
            SimpleNamespace(disable_pc=True, memory_checkpoint="memory.pth")
        )
    with pytest.raises(ValueError, match="requires --memory-checkpoint"):
        inference.validate_inference_args(
            SimpleNamespace(
                disable_pc=False,
                memory_checkpoint=None,
                allow_memory_mismatch=True,
            )
        )
    assert inference.validate_inference_args(
        SimpleNamespace(
            disable_pc=False,
            memory_checkpoint="memory.pth",
            allow_memory_mismatch=True,
        )
    ) is False


def test_artifact_mismatch_opt_in_only_relaxes_exact_student_fingerprint(
    monkeypatch,
):
    decoder_meta = {
        "training_design": "student_joint",
        "artifact_role": "ts_student",
        "labeled_split_fingerprint": "same-split",
        "baseline_fingerprint": "decoder-fingerprint",
        "pc_frozen": False,
    }
    memory_meta = {
        "training_design": "student_joint",
        "artifact_role": "ts_student_memory",
        "labeled_split_fingerprint": "same-split",
        "baseline_fingerprint": "memory-fingerprint",
        "pc_frozen": False,
    }

    def validate(path, expected):
        actual = decoder_meta if path == "decoder.pth" else memory_meta
        for key, value in expected.items():
            if actual.get(key) != value:
                raise RuntimeError(f"Artifact metadata mismatch for {key}")
        return dict(actual)

    monkeypatch.setattr(inference, "validate_artifact_metadata", validate)
    with pytest.raises(RuntimeError, match="baseline_fingerprint"):
        inference.validate_inference_artifacts(
            "decoder.pth", "memory.pth"
        )

    inference.validate_inference_artifacts(
        "decoder.pth",
        "memory.pth",
        allow_memory_mismatch=True,
    )

    memory_meta["labeled_split_fingerprint"] = "different-split"
    with pytest.raises(RuntimeError, match="labeled_split_fingerprint"):
        inference.validate_inference_artifacts(
            "decoder.pth",
            "memory.pth",
            allow_memory_mismatch=True,
        )


def test_memory_loader_requires_exact_student_producer(monkeypatch):
    cfg = DinoPCHBMConfig()
    producer = nn.Linear(2, 2)
    captured = {}

    def load(path, memory, *, expected_compat, require_producer_match):
        captured.update(expected_compat)
        captured["required"] = require_producer_match

    monkeypatch.setattr(inference, "load_memory_checkpoint", load)
    memory = inference.load_inference_memory(
        "memory.pth", pc_cfg=cfg, producer=producer
    )
    assert memory is not None
    assert captured["producer_role"] == "labeled_student"
    assert captured["required"] is True

    captured.clear()
    memory = inference.load_inference_memory(
        "memory.pth",
        pc_cfg=cfg,
        producer=producer,
        allow_memory_mismatch=True,
    )
    assert memory is not None
    assert captured["producer_role"] == "labeled_student"
    assert captured["required"] is False


def test_memory_mismatch_opt_in_preserves_structural_validation():
    cfg = DinoPCHBMConfig()
    producer = nn.Linear(2, 2)
    compat_meta = cfg.expected_memory_meta(
        producer_fingerprint="a" * 64
    )
    memory_dim = int(compat_meta["memory_dim"])
    state = {
        "format_version": 2,
        "schema_version": 2,
        "compat_meta": compat_meta,
        "memory_dim": memory_dim,
        "storage_dtype": compat_meta["storage_dtype"],
        "route": {
            "global_keys": torch.randn(1, memory_dim).half(),
            "environment_keys": torch.randn(1, memory_dim).half(),
            "img_ids": ["sample"],
        },
        "pairs": {
            "p3_keys": torch.randn(2, memory_dim).half(),
            "p2_keys": torch.randn(2, memory_dim).half(),
            "region_ids": torch.tensor([0, 1]),
            "pair_meta": [
                {"image_id": "sample", "region_id": 0},
                {"image_id": "sample", "region_id": 1},
            ],
        },
        "finalized": True,
    }

    with pytest.raises(RuntimeError, match="compatibility validation failed"):
        inference.load_inference_memory(
            state,
            pc_cfg=cfg,
            producer=producer,
        )

    memory = inference.load_inference_memory(
        state,
        pc_cfg=cfg,
        producer=producer,
        allow_memory_mismatch=True,
    )
    assert memory.is_ready()

    incompatible_state = dict(state)
    incompatible_state["compat_meta"] = dict(compat_meta)
    incompatible_state["compat_meta"]["producer_role"] = "teacher"
    with pytest.raises((ValueError, RuntimeError), match="producer_role"):
        inference.load_inference_memory(
            incompatible_state,
            pc_cfg=cfg,
            producer=producer,
            allow_memory_mismatch=True,
        )
