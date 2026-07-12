from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import inference as inference_module
from configs.pc_hbm_dino_config import DinoPCHBMConfig


def test_inference_applies_sigmoid_once_and_forwards_memory(monkeypatch, tmp_path):
    sample = (
        None,
        np.zeros((1, 2, 3), dtype=np.uint8),
        "sample.png",
        torch.zeros(3, 4, 4),
        None,
    )
    monkeypatch.setattr(inference_module, "TestDataset", lambda **_: [sample])
    monkeypatch.setattr(inference_module, "tqdm", lambda iterable: iterable)
    captured = {}

    def fake_imwrite(path, prediction):
        captured["path"] = path
        captured["prediction"] = prediction.copy()
        return True

    monkeypatch.setattr(inference_module.cv2, "imwrite", fake_imwrite)
    sentinel_memory = object()

    class FakeModel:
        def eval(self):
            return self

        def inference(self, image, memory=None, epoch=None):
            captured["memory"] = memory
            captured["epoch"] = epoch
            return torch.zeros(image.shape[0], 1, 1, 1)

    cfg = SimpleNamespace(
        device=torch.device("cpu"),
        CUDA=False,
        test_size=392,
        test_CAMO_imgs="unused-images",
        test_CAMO_masks="unused-masks",
    )
    inference_module.inference(
        ["CAMO"],
        FakeModel(),
        cfg,
        str(tmp_path),
        memory=sentinel_memory,
        epoch=30,
    )
    assert captured["memory"] is sentinel_memory
    assert captured["epoch"] == 30
    assert captured["prediction"].shape == (2, 3)
    # sigmoid(0) * 255 is 127.5 and the existing uint8 conversion truncates it.
    assert np.all(captured["prediction"] == 127)


def test_incompatible_memory_warns_and_falls_back(monkeypatch):
    def reject(*args, **kwargs):
        raise RuntimeError("compat_mismatch:schema_version")

    monkeypatch.setattr(inference_module, "load_memory_checkpoint", reject)
    with pytest.warns(RuntimeWarning, match="z_main"):
        memory = inference_module.load_inference_memory(
            "incompatible-memory.pth",
            DinoPCHBMConfig(),
        )
    assert memory is None


def test_missing_memory_warns_and_legacy_checkpoint_alias(monkeypatch):
    with pytest.warns(RuntimeWarning, match="z_main"):
        assert inference_module.load_inference_memory(None, DinoPCHBMConfig()) is None
    monkeypatch.setattr(
        "sys.argv",
        ["inference.py", "--checkpoint", "legacy.pth", "--datasets", "CAMO"],
    )
    args = inference_module.parse_args()
    assert args.decoder_checkpoint == "legacy.pth"
    assert args.memory_checkpoint is None
    assert args.datasets == ["CAMO"]
