import torch
import torch.nn.functional as F
import numpy as np
from utils.dataloader import TestDataset
from tqdm import tqdm
import os
import cv2
import argparse


def inference(datasets, model, cfg, pred_root):
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
                p = model.inference(img)
                p = F.interpolate(p, size=ori_gt.shape[1:], mode='bilinear', align_corners=False)
                p = torch.sigmoid(p) * 255
                p = p.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                if not cv2.imwrite(os.path.join(save_path, name), p):
                    raise IOError(f'Failed to save prediction: {os.path.join(save_path, name)}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run RSBL inference.')
    parser.add_argument('--checkpoint', default='./results/results_random_decoder1x1/ts_model_pseudo/student_epoch_15.pth')
    parser.add_argument('--pred-root', default='./results/results_random_decoder1x1/ts_model_pseudo/predictions')
    parser.add_argument('--datasets', nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])
    return parser.parse_args()


if __name__ == '__main__':
    from configs.base_model_config import Config
    from Model.base_model import BaseModel
    args = parse_args()
    cfg = Config()
    model = BaseModel()
    model.load_decoder_checkpoint(args.checkpoint)
    model.to(cfg.device)
    inference(args.datasets, model, cfg, args.pred_root)
