import torch
import torch.nn.functional as F
import numpy as np
from utils.dataloader import TestDataset
from tqdm import tqdm
import os
import cv2
import argparse
import warnings

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory.pc_memory import PCMemory
from utils.checkpoint_pc_hbm import load_decoder_compatible, load_memory_checkpoint
from utils.pc_memory_runner import module_fingerprint


def inference(datasets, model, cfg, pred_root, memory=None, epoch=30):
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

            for _, ori_gt, name, img, _ in tqdm(test_dataset):
                img = img.unsqueeze(0).to(cfg.device, non_blocking=cfg.CUDA)
                # BaseModel.inference always returns logits. A compatible memory
                # selects z_final; the fallback path selects z_main.
                p = model.inference(img, memory=memory, epoch=epoch)
                p = F.interpolate(p, size=ori_gt.shape[1:], mode='bilinear', align_corners=False)
                # Keep probability conversion at this outer boundary exactly once.
                p = torch.sigmoid(p) * 255
                p = p.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                if not cv2.imwrite(os.path.join(save_path, name), p):
                    raise IOError(f'Failed to save prediction: {os.path.join(save_path, name)}')


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
    inference(args.datasets, model, cfg, args.pred_root, memory=memory, epoch=args.epoch)
