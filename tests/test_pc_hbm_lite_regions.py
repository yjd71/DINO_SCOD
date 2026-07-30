from __future__ import annotations

import inspect

import pytest
import torch

from Model.PC_HBM.memory import pc_region_builder
from Model.PC_HBM.memory.pc_region_builder import (
    REGION_BG_NEAR,
    REGION_FG_BOUNDARY,
    REGION_IGNORE,
    build_boundary_pair_regions,
)
from Model.PC_HBM.memory.sampling_policy import (
    RegionSamplingRule,
    sample_region_indices,
)


def test_regions_cover_only_boundary_pair_band() -> None:
    gt = torch.zeros(1, 1, 28, 28)
    gt[:, :, 8:20, 9:19] = 1
    regions = build_boundary_pair_regions(gt)
    assert set(regions) == {
        "fg_boundary",
        "bg_near",
        "pair_union",
        "pair_labels",
    }
    assert regions["fg_boundary"].shape == (1, 1, 28, 28)
    assert regions["bg_near"].shape == (1, 1, 28, 28)
    assert regions["pair_labels"].shape == (1, 28, 28)
    assert not (regions["fg_boundary"] & regions["bg_near"]).any()
    assert regions["fg_boundary"].any()
    assert regions["bg_near"].any()
    assert (regions["pair_labels"] == REGION_IGNORE).any()
    assert torch.equal(
        regions["pair_labels"].eq(REGION_FG_BOUNDARY).unsqueeze(1),
        regions["fg_boundary"],
    )
    assert torch.equal(
        regions["pair_labels"].eq(REGION_BG_NEAR).unsqueeze(1),
        regions["bg_near"],
    )


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_empty_and_full_foreground_are_safe(fill: float) -> None:
    regions = build_boundary_pair_regions(torch.full((2, 1, 17, 19), fill))
    assert regions["pair_labels"].shape == (2, 28, 28)
    assert not regions["bg_near"].any()
    assert torch.isfinite(regions["pair_union"].float()).all()
    if fill == 0.0:
        assert not regions["fg_boundary"].any()
        assert (regions["pair_labels"] == REGION_IGNORE).all()
    else:
        assert regions["fg_boundary"][0, 0, 0, 0]
        assert not regions["fg_boundary"][0, 0, 1:-1, 1:-1].any()


def test_single_pixel_foreground_produces_disjoint_regions() -> None:
    gt = torch.zeros(1, 1, 28, 28)
    gt[0, 0, 14, 14] = 1
    regions = build_boundary_pair_regions(gt)
    assert regions["fg_boundary"][0, 0, 14, 14]
    assert not regions["bg_near"][0, 0, 14, 14]
    assert regions["bg_near"].any()
    assert not (regions["fg_boundary"] & regions["bg_near"]).any()


def test_region_builder_has_no_external_morphology_dependency() -> None:
    source = inspect.getsource(pc_region_builder)
    assert "cv2" not in source
    assert "numpy" not in source


def test_uniform_sampling_handles_quota_edges_and_covers_endpoints() -> None:
    rule = {"fg_boundary": RegionSamplingRule(max_count=48, min_count=8, ratio=0.5)}
    assert sample_region_indices(
        torch.zeros(4, 4, dtype=torch.bool),
        "fg_boundary",
        rules=rule,
    ).numel() == 0

    few = torch.zeros(4, 4, dtype=torch.bool)
    few.flatten()[:5] = True
    selected_few = sample_region_indices(few, "fg_boundary", rules=rule)
    assert selected_few.tolist() == [0, 1, 2, 3, 4]

    medium = torch.ones(4, 5, dtype=torch.bool)
    first = sample_region_indices(medium, "fg_boundary", rules=rule)
    second = sample_region_indices(medium, "fg_boundary", rules=rule)
    assert first.numel() == 10
    assert first[0].item() == 0
    assert first[-1].item() == 19
    assert first.unique().numel() == first.numel()
    assert torch.equal(first, second)

    large = torch.ones(10, 10, dtype=torch.bool)
    selected_large = sample_region_indices(large, "fg_boundary", rules=rule)
    assert selected_large.numel() == 48
    assert selected_large[0].item() == 0
    assert selected_large[-1].item() == 99
    assert selected_large.unique().numel() == selected_large.numel()


def test_sampling_validates_mask_and_region() -> None:
    with pytest.raises(ValueError, match=r"\[H,W\]"):
        sample_region_indices(torch.ones(1, 2, 3), "fg_boundary")
    with pytest.raises(KeyError, match="Unknown"):
        sample_region_indices(torch.ones(2, 3), "other")
