import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self):
        return self.rank == 0


def init_distributed():
    """Initialize torch.distributed when launched by torchrun."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1

    if distributed:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            backend = 'nccl'
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
    else:
        rank = 0
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return DistributedContext(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def configure_distributed(cfg, context, seed):
    """Attach runtime-only distributed settings to an existing config."""
    cfg.device = context.device
    cfg.CUDA = context.device.type == 'cuda'
    cfg.distributed = context.distributed
    cfg.rank = context.rank
    cfg.local_rank = context.local_rank
    cfg.world_size = context.world_size
    cfg.seed = seed


def wrap_distributed(model, context):
    if not context.distributed:
        return model
    if context.device.type == 'cuda':
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(model)


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value, device):
    """Return the mean of a scalar value across all training processes."""
    if not dist.is_initialized():
        return float(value)
    value_tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    value_tensor /= dist.get_world_size()
    return value_tensor.item()


def synchronize():
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
