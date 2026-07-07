import torch
import torch.nn.functional as F
import numpy as np
from utils.dataloader import TestDataset
from tqdm import tqdm
import os
import cv2


def inference(datasets, model, cfg):
    model.eval()
    for dataset in datasets:
        assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
        save_path = os.path.join(f'{cfg.save_dir}/predictions', dataset)
        os.makedirs(save_path, exist_ok=True)

        test_dataset = TestDataset(
            image_root=getattr(cfg, f'test_{dataset}_imgs'),
            gt_root=getattr(cfg, f'test_{dataset}_masks'),
            test_size=cfg.test_size
        )

        for _, ori_gt, name, img, _ in tqdm(test_dataset):
            img = img.unsqueeze(0).cuda()
            p = model.inference(img)
            p = F.interpolate(p, size=ori_gt.shape[1:], mode='bilinear', align_corners=False)
            p = torch.sigmoid(p) * 255
            p = p.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            # save preds
            cv2.imwrite(os.path.join(save_path, name), p)


if __name__ == '__main__':
    from configs.base_model_config import Config
    from Model.base_model import BaseModel
    cfg = Config()
    model = BaseModel()
    model.load_decoder_checkpoint('path to your trained student model checkpoint')
    model.to('cuda')
    inference(['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'], model, cfg)
