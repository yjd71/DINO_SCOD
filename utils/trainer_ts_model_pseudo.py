import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from utils.dataloader import LabeledTrainDataset, UnlabeledPseudoTrainDataset
from utils.distributed import is_main_process, reduce_mean, synchronize, unwrap_model
from utils.trainer_ts_model import CosineDecay, current_time, structure_loss


PSEUDO_THRESHOLD = 0.5
HARD_LOSS_WEIGHT = 2.0


class Trainer:
    """Teacher-student trainer whose unlabeled targets are generated online."""

    def __init__(self, model, cfg, scheduler=None):
        self.model = model
        self.cfg = cfg
        self.distributed = getattr(cfg, 'distributed', False)
        self.optimizer = optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = (
            CosineDecay(
                self.optimizer,
                max_lr=cfg.learning_rate,
                min_lr=cfg.min_lr,
                max_epoch=cfg.epochs,
            )
            if scheduler is None
            else scheduler
        )

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
            rPeper=False,
        )
        if len(self.labeled_train_set) == 0:
            raise ValueError('>>> Labeled training set is empty.')

        self.unlabeled_train_set = UnlabeledPseudoTrainDataset(
            u_image_root=cfg.train_imgs,
            sampled_txt=cfg.train_sample_txt,
            u_train_size=cfg.u_train_size,
            labeled_indices_pt=cfg.train_labeled_indices_pt,
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
            drop_last=False,
        )
        self.unlabeled_train_dl = DataLoader(
            self.unlabeled_train_set,
            batch_size=cfg.u_batch_size,
            shuffle=self.unlabeled_sampler is None,
            sampler=self.unlabeled_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.CUDA,
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
        )

        self.current_epoch = 1
        if is_main_process():
            os.makedirs(cfg.save_dir, exist_ok=True)
        synchronize()

    @staticmethod
    def _cycle_loader(loader):
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def _sum_structure_loss(logits, target):
        return sum(structure_loss(logit, target) for logit in logits)

    def train_epoch(self):
        self.model.train()
        labeled_iter = self._cycle_loader(self.labeled_train_dl)
        epoch_loss = 0.0
        epoch_loss_l = 0.0
        epoch_loss_u_s = 0.0
        epoch_loss_u_h = 0.0

        for itr_idx, u_imgs in enumerate(
            tqdm(self.unlabeled_train_dl, disable=not is_main_process()), start=1
        ):
            self.optimizer.zero_grad(set_to_none=True)
            _, l_imgs, l_gt = next(labeled_iter)

            l_imgs = l_imgs.to(self.cfg.device, non_blocking=self.cfg.CUDA)
            l_gt = l_gt.to(self.cfg.device, non_blocking=self.cfg.CUDA)
            u_imgs = u_imgs.to(self.cfg.device, non_blocking=self.cfg.CUDA)

            l_segs, u_segs, teacher_label = self.model(l_x=l_imgs, u_x=u_imgs)

            target_size = u_imgs.shape[-2:]
            soft_target = F.interpolate(
                teacher_label.detach(), size=target_size, mode='bilinear', align_corners=False
            )
            hard_target = (soft_target > PSEUDO_THRESHOLD).to(dtype=soft_target.dtype)

            l_seg_4, l_seg_3, l_seg_2, l_seg_1, l_seg_g = l_segs
            l_logits = [
                F.interpolate(logit, size=l_gt.shape[-2:], mode='bilinear', align_corners=False)
                for logit in (l_seg_1, l_seg_2, l_seg_3, l_seg_4, l_seg_g)
            ]
            u_seg_4, u_seg_3, u_seg_2, u_seg_1, u_seg_g = u_segs
            u_logits = [
                F.interpolate(logit, size=target_size, mode='bilinear', align_corners=False)
                for logit in (u_seg_1, u_seg_2, u_seg_3, u_seg_4, u_seg_g)
            ]

            loss_l_seg = self._sum_structure_loss(l_logits, l_gt)
            loss_u_seg_s = self._sum_structure_loss(u_logits, soft_target)
            loss_u_seg_h = self._sum_structure_loss(u_logits, hard_target)
            loss = loss_l_seg + loss_u_seg_s + HARD_LOSS_WEIGHT * loss_u_seg_h

            loss.backward()
            self.optimizer.step()
            unwrap_model(self.model).EMA()

            epoch_loss += loss.item()
            epoch_loss_l += loss_l_seg.item()
            epoch_loss_u_s += loss_u_seg_s.item()
            epoch_loss_u_h += loss_u_seg_h.item()

            if is_main_process() and itr_idx % self.cfg.log_interval == 0:
                print(f'[Train] Epoch: {self.current_epoch}, Iterate: {itr_idx}, Loss: {loss.item()}')
                print(
                    f'Loss_labeled: {loss_l_seg.item()}, '
                    f'Loss_unlabeled_teacher_soft: {loss_u_seg_s.item()}, '
                    f'Loss_unlabeled_teacher_hard: {loss_u_seg_h.item()}'
                )

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
                f'Avg Unlabeled Teacher Soft: {avg_loss_u_s}, '
                f'Avg Unlabeled Teacher Hard: {avg_loss_u_h}, '
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
            print('<<< Start teacher-pseudo training.')
            print(f'<<< Labeled Data Num: {len(self.labeled_train_set)}.')
            print(f'<<< Unlabeled Data Num: {len(self.unlabeled_train_set)}.')
        for epoch in range(self.cfg.epochs):
            if self.labeled_sampler is not None:
                self.labeled_sampler.set_epoch(epoch)
            if self.unlabeled_sampler is not None:
                self.unlabeled_sampler.set_epoch(epoch)
            if is_main_process():
                print(f'{current_time()} >>> Epoch: {self.current_epoch}/{self.cfg.epochs}')
            self.train_epoch()
        if is_main_process():
            print('<<< Training Finished.')
