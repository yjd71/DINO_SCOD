from __future__ import annotations

import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.losses import binary_pair_loss
from Model.PC_HBM.training.pseudo_label import weighted_structure_loss


def test_lite_losses_are_finite_under_available_autocast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    logits = torch.randn(2, 1, 16, 16, device=device, requires_grad=True)
    target = torch.rand(2, 1, 16, 16, device=device)
    confidence = torch.rand(2, 1, 16, 16, device=device)
    with torch.autocast(device.type, dtype=dtype, enabled=True):
        loss = weighted_structure_loss(logits, target, confidence)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_pair_ce_promotes_low_precision_logits_to_fp32():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.randn(
        2,
        2,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    aux = {
        "pc_hbm": {
            "pair_logits": logits,
            "query_valid": torch.tensor([True, True], device=device),
            "query_batch_ids": torch.tensor([0, 0], device=device),
            "query_flat_indices": torch.tensor([0, 1], device=device),
            "query_mask_map": torch.ones(1, 1, 2, 2, device=device),
        }
    }
    loss, _ = binary_pair_loss(
        aux,
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], device=device),
        logits,
        DinoPCHBMConfig(child_verification_mode="weighted_sum"),
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
