import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from utils.dataloader import LabeledTrainDataset, UnlabeledTrainDataset
from utils.distributed import is_main_process, reduce_mean, synchronize, unwrap_model
import math
import os
import logging
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from datetime import datetime


def current_time():
    return datetime.now().strftime('%m-%d %H:%M:%S')


def structure_loss(logits, mask):
    """
    loss function (ref: F3Net-AAAI-2020)

    pred: logits without activation
    mask: binary mask {0, 1}
    """
    # Filter out all-zero masks
    non_zero_mask = mask.sum(dim=(1, 2, 3)) > 0
    if not torch.any(non_zero_mask):
        return logits.sum() * 0.0
    logits = logits[non_zero_mask]
    mask = mask[non_zero_mask]

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
        self.distributed = getattr(cfg, 'distributed', False)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = CosineDecay(self.optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs) if scheduler is None else scheduler

        self.labeled_train_set = LabeledTrainDataset(
             l_image_root=cfg.train_imgs,
             l_gt_root=cfg.train_masks,
             l_txt_root=cfg.train_sample_txt,
             l_train_size=cfg.l_train_size,
             labeled_indices_pt=cfg.train_labeled_indices_pt,
             rVFlip=True,
             rCrop=True,
             rRotate=False,
             colorEnhance=True,
             rPeper=False
        )
        if len(self.labeled_train_set) == 0:
             raise ValueError('>>> Labeled training set is empty.')

        self.unlabeled_train_set = UnlabeledTrainDataset(
             u_image_root=cfg.train_imgs,
             u_gt_root=cfg.sam_labels,
             sampled_txt=cfg.train_sample_txt,
             u_train_size=cfg.u_train_size,
             labeled_indices_pt=cfg.train_labeled_indices_pt
        )
        if len(self.unlabeled_train_set) == 0:
             raise ValueError('>>> Unlabeled training set is empty.')
        
        self.labeled_sampler = (
             DistributedSampler(
                  self.labeled_train_set,
                  num_replicas=cfg.world_size,
                  rank=cfg.rank,
                  shuffle=True,
                  seed=cfg.seed,
             )
             if self.distributed
             else None
        )
        self.unlabeled_sampler = (
             DistributedSampler(
                  self.unlabeled_train_set,
                  num_replicas=cfg.world_size,
                  rank=cfg.rank,
                  shuffle=True,
                  seed=cfg.seed,
             )
             if self.distributed
             else None
        )
        self.labeled_train_dl = DataLoader(
             self.labeled_train_set,
             batch_size=cfg.l_batch_size,
             shuffle=self.labeled_sampler is None,
             sampler=self.labeled_sampler,
             num_workers=cfg.num_workers,
             pin_memory=cfg.CUDA,
             persistent_workers=cfg.num_workers > 0,
             drop_last=False
        )

        self.unlabeled_train_dl = DataLoader(
             self.unlabeled_train_set,
             batch_size=cfg.u_batch_size,
             shuffle=self.unlabeled_sampler is None,
             sampler=self.unlabeled_sampler,
             num_workers=cfg.num_workers,
             pin_memory=cfg.CUDA,
             persistent_workers=cfg.num_workers > 0,
             drop_last=False
        )

        self.current_epoch = 1

        # create save directory
        if is_main_process():
            os.makedirs(cfg.save_dir, exist_ok=True)
        synchronize()

    def _cycle_loader(self, loader):
        while True:
            for batch in loader:
                yield batch

    def train_epoch(self):
        self.model.train()
        labeled_iter = self._cycle_loader(self.labeled_train_dl)
        epoch_loss = 0.0
        epoch_loss_l = 0.0
        epoch_loss_u_s = 0.0
        epoch_loss_u_h = 0.0

        for itr_idx, (u_imgs, u_gt) in enumerate(
            tqdm(self.unlabeled_train_dl, disable=not is_main_process()), start=1
        ):
            self.optimizer.zero_grad(set_to_none=True)

            _, l_imgs, l_gt = next(labeled_iter)

            l_imgs = l_imgs.to(self.cfg.device, non_blocking=self.cfg.CUDA)
            l_gt = l_gt.to(self.cfg.device, non_blocking=self.cfg.CUDA)
            u_imgs = u_imgs.to(self.cfg.device, non_blocking=self.cfg.CUDA)
            u_gt = u_gt.to(self.cfg.device, non_blocking=self.cfg.CUDA)

            l_segs, u_segs, teacher_label = self.model(l_x=l_imgs, u_x=u_imgs)
            
            l_seg_4, l_seg_3, l_seg_2, l_seg_1, l_seg_g = l_segs
            u_seg_4, u_seg_3, u_seg_2, u_seg_1, u_seg_g = u_segs

            t_gt = F.interpolate(teacher_label, size=u_gt.shape[2:], mode='bilinear', align_corners=False)
            t_gt_b = torch.where(t_gt > 0.5, torch.ones_like(t_gt), torch.zeros_like(t_gt))
            
            l_seg_1 = F.interpolate(l_seg_1, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            l_seg_2 = F.interpolate(l_seg_2, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            l_seg_3 = F.interpolate(l_seg_3, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            l_seg_4 = F.interpolate(l_seg_4, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            l_seg_g = F.interpolate(l_seg_g, size=l_gt.shape[2:], mode='bilinear', align_corners=False)
            u_seg_1 = F.interpolate(u_seg_1, size=u_gt.shape[2:], mode='bilinear', align_corners=False)
            u_seg_2 = F.interpolate(u_seg_2, size=u_gt.shape[2:], mode='bilinear', align_corners=False)
            u_seg_3 = F.interpolate(u_seg_3, size=u_gt.shape[2:], mode='bilinear', align_corners=False)
            u_seg_4 = F.interpolate(u_seg_4, size=u_gt.shape[2:], mode='bilinear', align_corners=False)
            u_seg_g = F.interpolate(u_seg_g, size=u_gt.shape[2:], mode='bilinear', align_corners=False)
            
            
            loss_l_seg = structure_loss(l_seg_1, l_gt) + structure_loss(l_seg_2, l_gt) + structure_loss(l_seg_3, l_gt) + \
                         structure_loss(l_seg_4, l_gt) + structure_loss(l_seg_g, l_gt)
            loss_u_seg_s = structure_loss(u_seg_1, t_gt) + structure_loss(u_seg_2, t_gt) + structure_loss(u_seg_3, t_gt) + \
                           structure_loss(u_seg_4, t_gt) + structure_loss(u_seg_g, t_gt)
            loss_u_seg_h = structure_loss(u_seg_1, u_gt*t_gt_b) + structure_loss(u_seg_2, u_gt*t_gt_b) + structure_loss(u_seg_3, u_gt*t_gt_b) + \
                           structure_loss(u_seg_4, u_gt*t_gt_b) + structure_loss(u_seg_g, u_gt*t_gt_b)

            lamda = 2.0
            loss = loss_l_seg + loss_u_seg_s + loss_u_seg_h * lamda
            epoch_loss += loss.item()
            epoch_loss_l += loss_l_seg.item()
            epoch_loss_u_s += loss_u_seg_s.item()
            epoch_loss_u_h += loss_u_seg_h.item()

            # backward
            loss.backward()
            self.optimizer.step()

            unwrap_model(self.model).EMA()  # batch EMA

            if is_main_process() and itr_idx % self.cfg.log_interval == 0:
                print(f'[Train] Epoch: {self.current_epoch}, Iterate: {itr_idx}, Loss: {loss.item()}')
                print(f'Loss_labeled: {loss_l_seg.item()}, Loss_unlabeled_soft: {loss_u_seg_s.item()}, Loss_unlabled_hard: {loss_u_seg_h.item()}')

        num_iters = len(self.unlabeled_train_dl)
        avg_loss = reduce_mean(epoch_loss / num_iters, self.cfg.device)
        avg_loss_l = reduce_mean(epoch_loss_l / num_iters, self.cfg.device)
        avg_loss_u_s = reduce_mean(epoch_loss_u_s / num_iters, self.cfg.device)
        avg_loss_u_h = reduce_mean(epoch_loss_u_h / num_iters, self.cfg.device)
        lr = self.scheduler.get_lr() if self.scheduler is not None else self.optimizer.param_groups[0]['lr']
        if is_main_process():
            print(
                f'[Epoch] Epoch: {self.current_epoch}, '
                f'Avg Loss: {avg_loss}, '
                f'Avg Labeled: {avg_loss_l}, '
                f'Avg Unlabeled Soft: {avg_loss_u_s}, '
                f'Avg Unlabeled Hard: {avg_loss_u_h}, '
                f'LR: {lr}'
            )
        
        if is_main_process() and self.current_epoch > self.cfg.epochs - 5:
             unwrap_model(self.model).save_student(
                 os.path.join(self.cfg.save_dir, f'student_epoch_{self.current_epoch}.pth')
             )
        synchronize()
        
        self.current_epoch += 1
        
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self):
        if is_main_process():
            print(f'<<< Start Training.')
            print(f'<<< Labeled Data Num: {len(self.labeled_train_set)}.')
        for epoch in range(self.cfg.epochs):
            if self.labeled_sampler is not None:
                self.labeled_sampler.set_epoch(epoch)
            if self.unlabeled_sampler is not None:
                self.unlabeled_sampler.set_epoch(epoch)
            if is_main_process():
                print(f'{current_time()} >>> Epoch: {self.current_epoch}/{self.cfg.epochs}')
            self.train_epoch()
        if is_main_process():
            print(f'<<< Training Finished.')
