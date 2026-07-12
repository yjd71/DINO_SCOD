import argparse
import random

import numpy as np
import torch


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
    parser = argparse.ArgumentParser(
        description='Train the RSBL teacher-student model with online teacher pseudo labels.'
    )
    parser.add_argument('--teacher-pth', default='./results/results_random/base_model/base_model_epoch_30.pth')
    parser.add_argument('--ts_model-path', default='./results/results_random/ts_model')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic cuDNN kernels; slower.')
    parser.add_argument(
        '--labeled-indices-pt',
        default=None,
        help='Optional .pt file containing labeled sample stems/keys or integer indices. Overrides train_sample_txt when set.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(seed=args.seed, deterministic=args.deterministic)

    from configs.ts_model_config import Config
    from Model.ts_model import TSModel
    from utils.trainer_ts_model_pseudo import Trainer

    cfg = Config()
    cfg.save_dir = args.ts_model_path
    cfg.train_labeled_indices_pt = args.labeled_indices_pt
    ts_model = TSModel(teacher_pth=args.teacher_pth).to(cfg.device)
    trainer = Trainer(model=ts_model, cfg=cfg)
    trainer.train()
