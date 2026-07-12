import torch
import torch.nn as nn
from Model.decoder import Decoder
from utils.checkpoint_pc_hbm import load_decoder_compatible

class TSModel(nn.Module):
    def __init__(self, teacher_pth=None, pc_cfg=None, allow_legacy_pc_init=False):
        super(TSModel, self).__init__()
        
        # initialize the DINOv2 model: DINOv2-ViT-B/14 (default)
        self.dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load('./weight/dinov2_vitb14_pretrain.pth', map_location='cpu'))
        self.dino.requires_grad_(False)
        self.dino.eval()
        
        self.pc_cfg = pc_cfg
        self.allow_legacy_pc_init = allow_legacy_pc_init
        self.teacher = Decoder(pc_cfg=pc_cfg)  # Teacher Network Decoder
        self.student = Decoder(pc_cfg=pc_cfg)  # Student Network Decoder

        self.load_teacher(teacher_pth)
        self.load_student(teacher_pth)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        self.teacher.eval()
        return self

    @torch.no_grad()
    def extract_features(self, x):
        layer_indices = (
            list(self.pc_cfg.dino_layer_indices)
            if self.pc_cfg is not None
            else [2, 5, 8, 11]
        )
        return self.dino.get_intermediate_layers(
            x=x,
            n=layer_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )

    def _extract_features(self, x):
        return self.extract_features(x)

    @torch.inference_mode()
    def teacher_pseudo(self, features, memory, epoch):
        _, aux = self.teacher(
            features,
            memory=memory,
            pc_mode='teacher_pseudo',
            epoch=epoch,
            return_aux=True,
        )
        if not aux['pc_active'] or aux['p_final'] is None:
            raise RuntimeError(
                f'Teacher PC-HBM path is inactive: {aux.get("fallback_reason")}'
            )
        return aux

    def student_labeled(self, features, memory, epoch, query_image_ids=None):
        return self.student(
            features,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
            query_image_ids=query_image_ids,
        )

    def student_unlabeled(self, features, memory, epoch):
        return self.student(
            features,
            memory=memory,
            pc_mode='student_core',
            epoch=epoch,
            return_aux=True,
        )

    def forward(
        self,
        l_x=None,
        u_x=None,
        *,
        branch=None,
        features=None,
        memory=None,
        epoch=None,
        query_image_ids=None,
    ):
        if branch is not None:
            if features is None:
                raise ValueError('Precomputed DINO features are required for branch dispatch.')
            if branch == 'student_labeled':
                return self.student_labeled(
                    features, memory, epoch, query_image_ids=query_image_ids
                )
            if branch == 'student_unlabeled':
                return self.student_unlabeled(features, memory, epoch)
            raise ValueError(f'Unsupported TS forward branch: {branch!r}.')

        # Original combined/off API retained for the legacy SAM and pseudo trainers.
        if l_x is None or u_x is None:
            raise ValueError('Legacy TS forward requires both l_x and u_x.')
        l_x_features = self.extract_features(l_x)
        u_x_features = self.extract_features(u_x)
        with torch.no_grad():
            teacher_label = torch.sigmoid(self.teacher(u_x_features, pc_mode='off')[3].detach())
        l_segs = list(self.student(l_x_features, pc_mode='off'))
        u_segs = list(self.student(u_x_features, pc_mode='off'))
        return l_segs, u_segs, teacher_label

    
    def inference(self, x, memory=None, epoch=None):
        x_features = self.extract_features(x)
        if self.student.pc_hbm is None or memory is None:
            return self.student(x_features, pc_mode='off')[3]
        _, aux = self.student(
            x_features,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
        )
        return aux['z_final'] if aux['z_final'] is not None else aux['z_main']

    def load_teacher(self, path):
        if path is None:
            raise ValueError('Teacher checkpoint path is required.')
        assert path.endswith('.pth'), f'Teacher parameters path should end with ".pth", but got: "{path}"'
        self._load_decoder(self.teacher, path, role='teacher')
        print(f'Successfully load teacher parameters from "{path}".')
        self.teacher.requires_grad_(False)
        print('Teacher parameters are not trainable.')

    def load_student(self, path):
        if path is None:
            raise ValueError('Student initialization checkpoint path is required.')
        assert path.endswith('.pth'), f'Student parameters path should end with ".pth", but got: "{path}"'
        self._load_decoder(self.student, path, role='student')
        print(f'Successfully load student parameters from "{path}".')

    def _load_decoder(self, decoder, path, role):
        try:
            return load_decoder_compatible(
                decoder,
                path,
                require_pc_complete=(
                    decoder.pc_hbm is not None and not self.allow_legacy_pc_init
                ),
            )
        except RuntimeError as error:
            if decoder.pc_hbm is not None and not self.allow_legacy_pc_init:
                raise RuntimeError(
                    'PC-HBM training requires a complete Base PC-HBM checkpoint; '
                    'use allow_legacy_pc_init=True only for an explicit all-legacy '
                    'migration experiment.'
                ) from error
            raise RuntimeError(f'Incompatible {role} checkpoint: {error}') from error

    def save_student(self, path):
        assert path.endswith('.pth'), f'Student parameters path should end with ".pth", but got: "{path}"'
        torch.save(self.student.state_dict(), path)
        print(f'Successfully save student parameters to {path}.')

    @torch.no_grad()
    def update_teacher(self, momentum=0.995):
        student_params = dict(self.student.named_parameters())
        teacher_params = dict(self.teacher.named_parameters())
        if student_params.keys() != teacher_params.keys():
            differing = set(student_params).symmetric_difference(teacher_params)
            raise RuntimeError(f'Teacher/student EMA key mismatch: {sorted(differing)}')
        for name, student_value in student_params.items():
            teacher_params[name].mul_(momentum).add_(student_value, alpha=1.0 - momentum)

        student_buffers = dict(self.student.named_buffers())
        teacher_buffers = dict(self.teacher.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            differing = set(student_buffers).symmetric_difference(teacher_buffers)
            raise RuntimeError(f'Teacher/student buffer mismatch: {sorted(differing)}')
        for name, student_value in student_buffers.items():
            teacher_buffers[name].copy_(student_value)

    def EMA(self, alpha=0.995):
        self.update_teacher(momentum=alpha)
