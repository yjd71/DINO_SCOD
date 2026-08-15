from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model.PC_HBM.retrieval import PairVerifier


class _VerifierSmokeModule(nn.Module):
    def __init__(self, mode: str, device: torch.device) -> None:
        super().__init__()
        self.verifier = PairVerifier(
            dim=8,
            tau_parent=0.3,
            tau_child=0.4,
            child_verification_mode=mode,
            verification_strength_init=0.25,
        ).to(device)

    def forward(
        self,
        q3: torch.Tensor,
        q_child: torch.Tensor,
        parent_keys: torch.Tensor,
        child_keys: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.ones(
            parent_keys.shape[:-1], dtype=torch.bool, device=q3.device
        )
        result = self.verifier(
            q3,
            q_child,
            {
                "parent_keys": parent_keys,
                "paired_p2_keys": child_keys,
                "valid": valid,
            },
            torch.ones(q3.shape[0], 1, device=q3.device),
        )
        return result["pair_scores"][valid].mean()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--spawn-processes",
        type=int,
        default=0,
        help="Local file-store fallback when the host torchrun TCPStore is unavailable.",
    )
    return parser.parse_args()


def _device(backend: str, local_rank: int, world_size: int) -> torch.device:
    if backend == "nccl":
        if torch.cuda.device_count() < world_size:
            raise RuntimeError("NCCL smoke requires one CUDA device per rank")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _assert_synchronized(module: nn.Module) -> None:
    flattened = torch.cat(
        [parameter.detach().reshape(-1).float() for parameter in module.parameters()]
    )
    gathered = [torch.empty_like(flattened) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flattened)
    if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
        raise AssertionError("DDP parameters diverged across ranks")


def _run_mode(mode: str, device: torch.device) -> None:
    rank = dist.get_rank()
    torch.manual_seed(1701 + rank)
    module = _VerifierSmokeModule(mode, device)
    ddp = DistributedDataParallel(
        module,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
    q3 = torch.randn(3, 8, device=device)
    q_child = torch.randn(3, 8, device=device)
    parent_keys = torch.randn(3, 2, 3, 8, device=device)
    child_keys = torch.randn(3, 2, 3, 8, device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = ddp(q3, q_child, parent_keys, child_keys)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"{mode} produced a non-finite loss")
    loss.backward()
    missing = [
        name
        for name, parameter in ddp.module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise AssertionError(f"{mode} has unused parameters: {missing}")
    nonfinite = [
        name
        for name, parameter in ddp.module.named_parameters()
        if not bool(torch.isfinite(parameter.grad).all())
    ]
    if nonfinite:
        raise FloatingPointError(f"{mode} has non-finite gradients: {nonfinite}")
    optimizer.step()
    _assert_synchronized(ddp.module)
    dist.barrier()


def _distributed_worker(
    rank: int,
    world_size: int,
    backend: str,
    init_method: str,
) -> None:
    device = _device(backend, rank, world_size)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        _run_mode("weighted_sum", device)
        _run_mode("parent_conditioned", device)
        if dist.get_rank() == 0:
            print(
                "DDP child verifier smoke passed: weighted_sum + "
                "parent_conditioned"
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    if args.spawn_processes:
        if args.spawn_processes < 2:
            raise ValueError("--spawn-processes must be at least 2")
        rendezvous_path = Path(tempfile.gettempdir()) / (
            f"pcv-ddp-{uuid.uuid4().hex}.store"
        )
        mp.spawn(
            _distributed_worker,
            args=(
                args.spawn_processes,
                args.backend,
                rendezvous_path.resolve().as_uri(),
            ),
            nprocs=args.spawn_processes,
            join=True,
        )
        if rendezvous_path.exists():
            rendezvous_path.unlink()
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _distributed_worker(rank, world_size, args.backend, "env://")


if __name__ == "__main__":
    main()
