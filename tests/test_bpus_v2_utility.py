from __future__ import annotations

import pytest
import torch

from selection.bpus_v2 import (
    compute_bpus_v2_novelty,
    compute_bpus_v2_utility,
    compute_bpus_v2_value,
    compute_variant_utility,
    compute_variant_value,
    resolve_formula_variant,
)


def test_smooth_value_has_no_hard_gap_truncation() -> None:
    d_boundary = torch.tensor([0.2, 0.7])
    d_global = torch.tensor([0.5, 0.1])
    valid = torch.tensor([True, False])

    value = compute_bpus_v2_value(d_boundary, d_global, valid)

    assert value[0].item() == pytest.approx(0.1)
    assert value[1].item() == 0.0


def test_soft_reward_endpoints_and_novelty_clamping() -> None:
    value = torch.tensor([0.25, 0.25])
    novelty = compute_bpus_v2_novelty(torch.tensor([2.0, -3.0]))
    utility = compute_bpus_v2_utility(value, novelty)

    assert torch.equal(novelty, torch.tensor([0.0, 1.0]))
    assert torch.equal(utility, torch.tensor([0.25, 0.5]))


@pytest.mark.parametrize(
    ("variant", "expected_value", "expected_utility"),
    [
        ("v1", 0.0, 0.0),
        ("v2-a", 0.1, 0.025),
        ("v2-b", 0.0, 0.0),
        ("v2", 0.1, 0.125),
    ],
)
def test_all_four_formula_variants(
    variant: str, expected_value: float, expected_utility: float
) -> None:
    value = compute_variant_value(
        torch.tensor([0.2]),
        torch.tensor([0.5]),
        torch.tensor([True]),
        variant=variant,
    )
    utility = compute_variant_utility(
        value, torch.tensor([0.25]), variant=variant
    )

    assert value.item() == pytest.approx(expected_value)
    assert utility.item() == pytest.approx(expected_utility)


def test_unknown_formula_variant_fails() -> None:
    with pytest.raises(ValueError, match="Unknown formula variant"):
        resolve_formula_variant("future")
