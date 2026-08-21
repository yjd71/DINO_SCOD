"""Self-contained single-GPU smoke for Student-owned PC gradient isolation."""

from copy import deepcopy
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.fusion import P3GatedResidual
from Model.PC_HBM.training import pc_unlabeled_loss, prepare_pseudo_targets


class JointCudaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(16, 16)
        self.pc_hbm = nn.Linear(16, 16)
        self.pc_hbm_readonly = deepcopy(self.pc_hbm).requires_grad_(False).eval()

    def forward(self, labeled, unlabeled):
        labeled_value = self.student(labeled)
        unlabeled_value = self.student(unlabeled)
        return (
            labeled_value + self.pc_hbm(labeled_value),
            unlabeled_value + self.pc_hbm_readonly(unlabeled_value),
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(2025)
    device = torch.device("cuda")
    model = JointCudaModel().to(device)
    optimizer = torch.optim.Adam(
        list(model.student.parameters()) + list(model.pc_hbm.parameters()),
        lr=1.0e-4,
    )
    labeled = torch.randn(8, 16, device=device)
    unlabeled = torch.randn(8, 16, device=device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        labeled_output, unlabeled_output = model(labeled, unlabeled)
        loss = labeled_output.square().mean() + unlabeled_output.square().mean()
    loss.backward()
    if model.student.weight.grad is None or model.pc_hbm.weight.grad is None:
        raise RuntimeError("Trainable Student gradients are missing")
    if model.pc_hbm_readonly.weight.grad is not None:
        raise RuntimeError("Readonly PC-HBM received a gradient")
    if not torch.isfinite(loss.detach()):
        raise RuntimeError("CUDA Student-joint loss is non-finite")
    optimizer.step()
    model.pc_hbm_readonly.load_state_dict(model.pc_hbm.state_dict(), strict=True)
    if not torch.equal(model.pc_hbm.weight, model.pc_hbm_readonly.weight):
        raise RuntimeError("Readonly PC-HBM synchronization failed")

    with torch.inference_mode():
        teacher_aux = {
            "p_final": torch.rand(2, 1, 8, 8, device=device),
            "pc_active": True,
            "fallback_reason": None,
            "forward_mode": "teacher_pseudo",
            "pc_hbm": {
                "query_mask_map": torch.ones(2, 1, 8, 8, device=device),
                "memory_confidence_map": torch.rand(2, 1, 8, 8, device=device),
            },
        }
        pseudo = prepare_pseudo_targets(teacher_aux)
    if any(torch.is_inference(value) for value in pseudo.values()):
        raise RuntimeError("Pseudo targets remained inference tensors")
    mask_outputs = tuple(
        torch.randn(2, 1, 8, 8, device=device, requires_grad=True)
        for _ in range(5)
    )
    mask_aux = {
        "forward_mode": "full",
        "pc_active": True,
        "pc_engine_source": "external_readonly",
        "z_main": mask_outputs[3],
    }
    pseudo_loss, _ = pc_unlabeled_loss(
        mask_outputs,
        mask_aux,
        pseudo["p_soft"],
        pseudo["confidence"],
        DinoPCHBMConfig(),
    )
    pseudo_loss.backward()
    if any(output.grad is None for output in mask_outputs):
        raise RuntimeError("Pseudo-label mask gradients are missing")

    residual = P3GatedResidual(dim=4, p3_ch=4).to(device)
    with torch.no_grad():
        residual.out.weight.copy_(2.0 * torch.eye(4, device=device))
        residual.out.bias.zero_()
    corrected, delta = residual(
        torch.zeros(1, 4, 2, 2, device=device),
        batch_ids=torch.tensor([0], device=device),
        flat_indices=torch.tensor([0], device=device),
        correction_token=torch.ones(1, 4, device=device),
        query_valid=torch.tensor([True], device=device),
    )
    if not torch.equal(delta, 2.0 * torch.ones_like(delta)):
        raise RuntimeError("P3 correction did not use the learned output projection")
    if not torch.equal(corrected[0, :, 0, 0], 2.0 * torch.ones(4, device=device)):
        raise RuntimeError("Projected P3 correction was not written to the selected token")
    print("student_joint_cuda_smoke=ok")


if __name__ == "__main__":
    main()
