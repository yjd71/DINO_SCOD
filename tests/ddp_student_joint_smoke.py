"""Two-rank CPU Gloo smoke for the student-joint DDP dispatch."""

from copy import deepcopy
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


class JointSmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(4, 4, bias=False)
        self.pc_hbm = nn.Linear(4, 4, bias=False)
        self.pc_hbm_readonly = deepcopy(self.pc_hbm).requires_grad_(False).eval()

    def forward(self, labeled, unlabeled):
        labeled_value = self.student(labeled)
        unlabeled_value = self.student(unlabeled)
        return (
            labeled_value + self.pc_hbm(labeled_value),
            unlabeled_value + self.pc_hbm_readonly(unlabeled_value),
        )

    @torch.no_grad()
    def sync_readonly(self):
        self.pc_hbm_readonly.load_state_dict(self.pc_hbm.state_dict(), strict=True)


def _worker(rank, world_size, store_path):
    dist.init_process_group(
        "gloo",
        init_method=Path(store_path).as_uri(),
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(2025 + rank)
        core = JointSmokeModel()
        model = DistributedDataParallel(core, find_unused_parameters=True)
        optimizer = torch.optim.SGD(
            list(core.student.parameters()) + list(core.pc_hbm.parameters()),
            lr=0.01,
        )
        labeled = torch.randn(3, 4)
        unlabeled = torch.randn(3, 4)
        labeled_output, unlabeled_output = model(labeled, unlabeled)
        loss = labeled_output.square().mean() + unlabeled_output.square().mean()
        loss.backward()
        if core.student.weight.grad is None or core.pc_hbm.weight.grad is None:
            raise RuntimeError("Student-joint DDP gradients are missing")
        if core.pc_hbm_readonly.weight.grad is not None:
            raise RuntimeError("Readonly PC-HBM received a gradient")
        optimizer.step()
        core.sync_readonly()
        if not torch.equal(core.pc_hbm.weight, core.pc_hbm_readonly.weight):
            raise RuntimeError("Readonly PC-HBM synchronization failed")

        gathered = [torch.empty_like(core.pc_hbm.weight) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, core.pc_hbm.weight)
        if any(not torch.equal(gathered[0], value) for value in gathered[1:]):
            raise RuntimeError("Trainable PC-HBM diverged across ranks")
        if rank == 0:
            print("student_joint_ddp_smoke=ok")
    finally:
        dist.destroy_process_group()


def main():
    world_size = 2
    with tempfile.TemporaryDirectory(prefix="student-joint-gloo-") as directory:
        store_path = str(Path(directory) / "store")
        mp.spawn(_worker, args=(world_size, store_path), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
