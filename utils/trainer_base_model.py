import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import LabeledTrainDataset
import math
import os
from tqdm import tqdm
from utils.metrics import MAE, wF, meanEM, meanFM, SM
import torch.nn.functional as F
import numpy as np


def structure_loss(logits, mask):
	"""
    loss function (ref: F3Net-AAAI-2020)

    pred: logits without activation
    mask: binary mask {0, 1}
    """
	weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
	wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

	pred = torch.sigmoid(logits)
	inter = ((pred * mask) * weit).sum(dim=(2, 3))
	union = ((pred + mask) * weit).sum(dim=(2, 3))
	wiou = 1 - (inter + 1) / (union - inter + 1)
	return (wbce + wiou).mean()


class CosineDecay:
	def __init__(self,
	             optimizer,
	             max_lr,
	             min_lr,
	             max_epoch,
	             test_mode=False):
		self.optimizer = optimizer
		self.max_lr = max_lr
		self.min_lr = min_lr
		self.max_epoch = max_epoch
		self.test_mode = test_mode

		self.current_lr = max_lr
		self.cnt = 0
		if self.max_epoch > 1:
			self.scale = (max_lr - min_lr) / 2
			self.shift = (max_lr + min_lr) / 2
			self.alpha = math.pi / (max_epoch - 1)

	def step(self):
		self.cnt += 1
		self.current_lr = self.scale * math.cos(self.alpha * self.cnt) + self.shift

		if not self.test_mode:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = self.current_lr

	def get_lr(self):
		return self.current_lr


class Trainer():
    def __init__(self, model, cfg, scheduler=None):
        self.model = model
        self.cfg = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.scheduler = CosineDecay(self.optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs) if scheduler is None else scheduler

        self.labeled_train_set = LabeledTrainDataset(
             l_image_root=cfg.train_imgs,
             l_gt_root=cfg.train_masks,
             l_txt_root=cfg.train_sample_txt,
             l_train_size=cfg.train_size,
             rVFlip=True,
             rCrop=True,
             rRotate=False,
             colorEnhance=True,
             rPeper=False
        )
        
        self.labeled_train_dl = DataLoader(
             self.labeled_train_set,
             batch_size=cfg.batch_size,
             shuffle=True,
             num_workers=cfg.num_workers,
             pin_memory=True,
             drop_last=False
        )


        self.current_epoch = 1

        # create save directory
        os.makedirs(cfg.save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()

        for itr_idx in tqdm(range(len(self.labeled_train_dl))):
            self.optimizer.zero_grad()
            # load data
            try:
                 _, l_imgs, l_gt = next(iter(self.labeled_train_dl))
            except StopIteration:
                 self.labeled_train_dl = iter(self.labeled_train_dl)
                 _, l_imgs, l_gt = next(iter(self.labeled_train_dl))

            l_imgs = l_imgs.to('cuda')
            l_gt = l_gt.to('cuda')

            seg_outputs = self.model(x=l_imgs)
            seg_4, seg_3, seg_2, seg_1, seg_g = seg_outputs

            seg_1 = F.interpolate(seg_1, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            seg_2 = F.interpolate(seg_2, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            seg_3 = F.interpolate(seg_3, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            seg_4 = F.interpolate(seg_4, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            seg_g = F.interpolate(seg_g, size=l_gt.shape[2:], mode='bilinear', align_corners=False)

            loss = structure_loss(seg_1, l_gt) + structure_loss(seg_2, l_gt) + structure_loss(seg_3, l_gt) + structure_loss(seg_4, l_gt) + structure_loss(seg_g, l_gt)

            # backward
            loss.backward()
            self.optimizer.step()

            if (itr_idx + 1) % self.cfg.log_interval == 0:
                  print(f'[Train] Epoch: {self.current_epoch}, Iterate: {itr_idx + 1}, Loss: {loss.item()}')

        if self.current_epoch > self.cfg.epochs - 5:
              self.model.save_decoder_checkpoint(os.path.join(self.cfg.save_dir, f'base_model_epoch_{self.current_epoch}.pth'))
        
        self.current_epoch += 1
        
        if self.scheduler is not None:
              self.scheduler.step()

    def train(self):
        print(f'<<< Start Training.')
        print(f'<<< Labeled Data Num: {len(self.labeled_train_set)}.')
        for epoch in range(self.cfg.epochs):
              self.train_epoch()
        print(f'<<< Training Finished.')
        
