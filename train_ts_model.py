import torch
import random
import numpy as np


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    set_seed(seed=2025)

    from utils.trainer_ts_model import Trainer
    from configs.ts_model_config import Config
    from Model.ts_model import TSModel

    ts_model = TSModel(teacher_pth="path to your trained teacher model").to('cuda')
    cfg = Config()
    trainer = Trainer(model=ts_model, cfg=cfg)
    trainer.train()
    