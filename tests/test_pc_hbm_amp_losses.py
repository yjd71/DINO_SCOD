from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from Model.PC_HBM.training.losses import (
    _gate_loss,
    _single_boundary_loss,
    probability_bce,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast coverage")
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_probability_bce_is_amp_safe_on_cuda(reduction):
    logits = torch.tensor(
        [-4.0, 0.0, 4.0], device="cuda", dtype=torch.float16, requires_grad=True
    )
    target = torch.tensor([0.0, 1.0, 1.0], device="cuda", dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        probability = torch.sigmoid(logits)
        loss = probability_bce(probability, target, reduction=reduction)

    assert loss.dtype == torch.float32
    assert loss.shape == (() if reduction == "mean" else target.shape)
    scalar_loss = loss if reduction == "mean" else loss.mean()
    assert torch.isfinite(scalar_loss)
    scalar_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_probability_bce_low_precision_clamp_is_finite(dtype):
    probability = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
    target = torch.tensor([0.0, 1.0], dtype=dtype)

    loss = probability_bce(probability, target)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    expected = F.binary_cross_entropy(
        probability.float().clamp(1.0e-4, 1.0 - 1.0e-4), target.float()
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_probability_bce_validates_tensor_shape_dtype_and_reduction():
    probability = torch.full((2,), 0.5)
    target = torch.ones(2)
    with pytest.raises(TypeError, match="torch.Tensor"):
        probability_bce([0.5, 0.5], target)
    with pytest.raises(ValueError, match="same shape"):
        probability_bce(probability, torch.ones(1))
    with pytest.raises(TypeError, match="floating-point"):
        probability_bce(torch.ones(2, dtype=torch.long), target)
    with pytest.raises(ValueError, match="unsupported reduction"):
        probability_bce(probability, target, reduction="invalid")


def test_probability_bce_detaches_and_clamps_target():
    probability = torch.full((3,), 0.5, requires_grad=True)
    target = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)

    loss = probability_bce(probability, target)
    expected = F.binary_cross_entropy(probability, target.detach().clamp(0.0, 1.0))
    torch.testing.assert_close(loss, expected)
    loss.backward()

    assert probability.grad is not None
    assert target.grad is None


def test_probability_bce_accepts_integer_and_boolean_targets():
    probability = torch.tensor([0.25, 0.75], requires_grad=True)

    integer_loss = probability_bce(probability, torch.tensor([0, 1]))
    boolean_loss = probability_bce(probability, torch.tensor([False, True]))

    torch.testing.assert_close(integer_loss, boolean_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast coverage")
def test_boundary_and_gate_probability_losses_backpropagate_under_cuda_autocast():
    gt = torch.zeros(1, 1, 8, 8, device="cuda", dtype=torch.float16)
    gt[:, :, 2:6, 2:6] = 1.0
    boundary_logits = torch.zeros(
        1, 1, 8, 8, device="cuda", dtype=torch.float16, requires_grad=True
    )
    gate_logits = torch.zeros(2, 1, device="cuda", dtype=torch.float16, requires_grad=True)
    z_main = torch.full((1, 1, 8, 8), -2.0, device="cuda", dtype=torch.float16)
    indices = {
        "batch_ids": torch.tensor([0, 0], device="cuda"),
        "flat_indices": torch.tensor([18, 27], device="cuda"),
    }

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        boundary = torch.sigmoid(boundary_logits)
        gate = torch.sigmoid(gate_logits)
        boundary_loss = _single_boundary_loss(boundary, gt, boundary)
        gate_loss = _gate_loss(
            {
                "gate_pc_token": gate,
                "C23_token": torch.zeros_like(gate),
                "boundary_indices3": indices,
                "query_valid": torch.ones(2, device="cuda", dtype=torch.bool),
                "B3": boundary,
            },
            {"z_main": z_main},
            gt,
            gate,
        )
        loss = boundary_loss + gate_loss

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert boundary_logits.grad is not None
    assert gate_logits.grad is not None
    assert torch.isfinite(boundary_logits.grad).all()
    assert torch.isfinite(gate_logits.grad).all()


def test_sparse_boundary_loss_uses_only_valid_elements_and_resizes_mask():
    prediction = torch.tensor(
        [[[[0.25, 0.75], [0.6, 0.4]]]], dtype=torch.float32, requires_grad=True
    )
    gt = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[1.0]]])

    loss = _single_boundary_loss(prediction, gt, prediction, valid_mask=valid_mask)
    expected = probability_bce(prediction, torch.zeros_like(prediction), reduction="none").mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == prediction.numel()


def test_sparse_boundary_loss_excludes_invalid_elements_from_loss_and_gradient():
    prediction = torch.tensor(
        [[[[0.25, 0.75], [0.6, 0.4]]]], dtype=torch.float32, requires_grad=True
    )
    gt = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool)

    loss = _single_boundary_loss(prediction, gt, prediction, valid_mask=valid_mask)
    expected = probability_bce(prediction[:, :, :1, :1], torch.zeros(1, 1, 1, 1))
    torch.testing.assert_close(loss, expected)
    loss.backward()

    assert prediction.grad[0, 0, 0, 0].abs() > 0
    assert torch.count_nonzero(prediction.grad[0, 0, 0, 1:]) == 0
    assert torch.count_nonzero(prediction.grad[0, 0, 1:, :]) == 0


def test_sparse_boundary_loss_preserves_fractional_mask_weights():
    prediction = torch.tensor(
        [[[[0.25, 0.75], [0.6, 0.4]]]], dtype=torch.float32, requires_grad=True
    )
    gt = torch.zeros(1, 1, 2, 2)
    valid_mask = torch.tensor([[[2.0, 0.5], [-1.0, 0.0]]])

    loss = _single_boundary_loss(prediction, gt, prediction, valid_mask=valid_mask)
    per_element = probability_bce(
        prediction, torch.zeros_like(prediction), reduction="none"
    )
    weights = valid_mask.unsqueeze(1).clamp(0.0, 1.0)
    expected = (per_element * weights).sum() / weights.sum()
    torch.testing.assert_close(loss, expected)


def test_all_invalid_sparse_boundary_returns_differentiable_zero():
    prediction = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    gt = torch.zeros_like(prediction)
    valid_mask = torch.zeros(1, 2, 2, dtype=torch.bool)

    loss = _single_boundary_loss(prediction, gt, prediction, valid_mask=valid_mask)

    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0
