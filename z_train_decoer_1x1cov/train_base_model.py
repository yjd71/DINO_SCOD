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
    parser = argparse.ArgumentParser(description='Train the RSBL base teacher model.')
    parser.add_argument('--base_model-path', default='./results/results_random_decoder1x1/base_model')
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
    set_seed(seed=args.seed, deterministic=args.deterministic)

    from utils.trainer_base_model import Trainer
    from configs.base_model_config import Config
    from Model.base_model import BaseModel
    from utils.decoer_1x1cov import Conv1x1Decoder

    cfg = Config()
    cfg.save_dir = args.base_model_path
    cfg.train_labeled_indices_pt = args.labeled_indices_pt
    base_model = BaseModel()
    base_model.decoder = Conv1x1Decoder()
    base_model = base_model.to(cfg.device)
    trainer = Trainer(model=base_model, cfg=cfg)
    trainer.train()
    
