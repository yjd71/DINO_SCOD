import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import TestDataset
from tqdm import tqdm
import os
import cv2
import argparse
import warnings
from contextlib import nullcontext

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory.pc_memory import PCMemory
from utils.checkpoint_pc_hbm import load_decoder_compatible, load_memory_checkpoint
from utils.pc_memory_runner import module_fingerprint


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be a non-negative integer')
    return value


def _inference_collate(samples):
    """Stack resized inputs while preserving variable-size output metadata."""

    ori_gts = [sample[1] for sample in samples]
    names = [sample[2] for sample in samples]
    try:
        images = torch.stack([sample[3] for sample in samples], dim=0)
    except RuntimeError as error:
        raise ValueError(
            'Inference inputs must share the configured test_size so they can be batched.'
        ) from error
    return ori_gts, names, images


def inference(
    datasets,
    model,
    cfg,
    pred_root,
    memory=None,
    epoch=30,
    batch_size=1,
    num_workers=0,
    amp=False,
):
    """Run batched inference and restore every prediction to its original size.

    The programmatic defaults retain the legacy one-image behavior. The CLI
    passes throughput-oriented defaults (batch 16 and four loader workers).
    """

    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if num_workers < 0:
        raise ValueError('num_workers must be a non-negative integer')
    device = torch.device(cfg.device)
    cuda_device = device.type == 'cuda'
    amp_enabled = bool(amp and cuda_device)
    model.eval()
    with torch.inference_mode():
        for dataset in datasets:
            assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
            save_path = os.path.join(pred_root, dataset)
            os.makedirs(save_path, exist_ok=True)

            test_dataset = TestDataset(
                image_root=getattr(cfg, f'test_{dataset}_imgs'),
                gt_root=getattr(cfg, f'test_{dataset}_masks'),
                test_size=cfg.test_size
            )
            loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=cuda_device,
                persistent_workers=num_workers > 0,
                collate_fn=_inference_collate,
            )

            for ori_gts, names, images in tqdm(loader):
                images = images.to(device, non_blocking=cuda_device)
                # BaseModel.inference always returns logits. A compatible memory
                # selects z_final; the fallback path selects z_main.
                autocast = (
                    torch.autocast(device_type='cuda', dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    logits = model.inference(images, memory=memory, epoch=epoch)
                if logits.shape[0] != len(names):
                    raise ValueError(
                        'Model inference batch dimension does not match the input batch.'
                    )
                for logit, ori_gt, name in zip(logits, ori_gts, names):
                    logit = F.interpolate(
                        logit.unsqueeze(0),
                        size=ori_gt.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                    # Keep probability conversion at this outer boundary exactly once.
                    prediction = torch.sigmoid(logit) * 255
                    prediction = prediction.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                    output_path = os.path.join(save_path, name)
                    if not cv2.imwrite(output_path, prediction):
                        raise IOError(f'Failed to save prediction: {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run RSBL inference.')
    parser.add_argument(
        '--checkpoint',
        '--decoder-checkpoint',
        dest='decoder_checkpoint',
        default='./results/results_random_decoder1x1/ts_model_pseudo/student_epoch_15.pth',
        help='Raw or nested Decoder/Student checkpoint. --checkpoint remains the legacy alias.',
    )
    parser.add_argument(
        '--memory-checkpoint',
        default=None,
        help='Optional finalized PC-HBM memory checkpoint. Incompatible memory falls back to z_main.',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=30,
        help='Mixture schedule epoch (default: terminal Base epoch).',
    )
    parser.add_argument(
        '--require-producer-match',
        action='store_true',
        help='Also require the memory producer fingerprint to match the expected fingerprint.',
    )
    parser.add_argument(
        '--batch-size',
        type=_positive_int,
        default=16,
        help='Inference batch size (default: 16).',
    )
    parser.add_argument(
        '--num-workers',
        type=_non_negative_int,
        default=4,
        help='DataLoader worker processes (default: 4).',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable CUDA FP16 autocast for faster inference.',
    )
    parser.add_argument('--pred-root', default='./results/results_random_decoder1x1/ts_model_pseudo/predictions')
    parser.add_argument('--datasets', nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])
    return parser.parse_args()


def load_inference_memory(
    path,
    pc_cfg,
    require_producer_match=False,
    producer_fingerprint=None,
):
    """Load compatible CPU-FP16 memory, or warn and return ``None``.

    PC training is fail-fast, but inference remains usable through z_main when
    a memory artifact is absent, unready, or incompatible.
    """

    if path is None:
        warnings.warn(
            'No PC-HBM memory checkpoint was provided; inference will use z_main logits.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    memory = PCMemory(
        pc_cfg.memory_dim,
        pc_cfg.value_dim,
        pc_cfg.geometry_dim,
        storage_dtype=pc_cfg.memory_storage_dtype,
        config=pc_cfg,
    )
    try:
        load_memory_checkpoint(
            path,
            memory,
            expected_compat=pc_cfg.expected_memory_meta(
                producer_fingerprint=(
                    producer_fingerprint if require_producer_match else None
                )
            ),
            require_producer_match=require_producer_match,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        warnings.warn(
            f'PC-HBM memory is unavailable or incompatible ({error}); inference will use z_main logits.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return memory


if __name__ == '__main__':
    from configs.base_model_config import Config
    from Model.base_model import BaseModel
    args = parse_args()
    cfg = Config()
    pc_cfg = DinoPCHBMConfig()
    model = BaseModel(pc_cfg=pc_cfg)
    load_decoder_compatible(
        model.decoder,
        args.decoder_checkpoint,
        require_pc_complete=args.memory_checkpoint is not None,
    )
    memory = load_inference_memory(
        args.memory_checkpoint,
        pc_cfg,
        require_producer_match=args.require_producer_match,
        producer_fingerprint=(
            module_fingerprint(model.decoder)
            if args.require_producer_match
            else None
        ),
    )
    model.to(cfg.device)
    inference(
        args.datasets,
        model,
        cfg,
        args.pred_root,
        memory=memory,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )
