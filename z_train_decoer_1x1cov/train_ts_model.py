import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import random
import numpy as np
import argparse


def set_seed(seed=2025, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def parse_args():
    parser = argparse.ArgumentParser(description='Train the RSBL teacher-student model.')
    parser.add_argument('--teacher-pth', default='./results/results_random_decoder1x1/base_model/base_model_epoch_30.pth')
    parser.add_argument('--ts_model-path', default='./results/results_random_decoder1x1/ts_model')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic cuDNN kernels; slower.')
    parser.add_argument(
        '--labeled-indices-pt',
        default=None,
        help='Optional .pt file containing labeled sample stems/keys or integer indices. Overrides train_sample_txt when set.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from utils.distributed import (
        cleanup_distributed,
        configure_distributed,
        init_distributed,
        wrap_distributed,
    )

    distributed_context = init_distributed()
    try:
        set_seed(seed=args.seed + distributed_context.rank, deterministic=args.deterministic)

        from utils.trainer_ts_model import Trainer
        from configs.ts_model_config import Config
        import Model.ts_model as ts_model_module
        from Model.ts_model import TSModel
        from utils.decoer_1x1cov import Conv1x1Decoder

        cfg = Config()
        configure_distributed(cfg, distributed_context, seed=args.seed)
        cfg.save_dir = args.ts_model_path
        cfg.train_labeled_indices_pt = args.labeled_indices_pt
        original_decoder = ts_model_module.Decoder
        ts_model_module.Decoder = Conv1x1Decoder
        try:
            ts_model = TSModel(teacher_pth=args.teacher_pth)
        finally:
            ts_model_module.Decoder = original_decoder
        ts_model = ts_model.to(cfg.device)
        ts_model = wrap_distributed(ts_model, distributed_context)
        trainer = Trainer(model=ts_model, cfg=cfg)
        trainer.train()
    finally:
        cleanup_distributed()
    
