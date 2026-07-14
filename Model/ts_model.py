import torch
import torch.nn as nn
import warnings
from Model.decoder import Decoder
from Model.PC_HBM.training.ema import update_ema_module
from utils.checkpoint_pc_hbm import load_decoder_compatible

class TSModel(nn.Module):
    VALID_TRAINING_DESIGNS = {'teacher_only', 'joint'}

    def __init__(
        self,
        teacher_pth=None,
        student_pth=None,
        pc_cfg=None,
        allow_legacy_pc_init=False,
        training_design='teacher_only',
    ):
        super(TSModel, self).__init__()

        if training_design not in self.VALID_TRAINING_DESIGNS:
            raise ValueError(
                f'Unsupported training_design={training_design!r}; expected one of '
                f'{sorted(self.VALID_TRAINING_DESIGNS)}.'
            )
        if training_design == 'teacher_only' and allow_legacy_pc_init:
            raise ValueError(
                'teacher_only requires a complete Teacher PC-HBM checkpoint; '
                'allow_legacy_pc_init is only valid for joint migration experiments.'
            )
        
        # initialize the DINOv2 model: DINOv2-ViT-B/14 (default)
        self.dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load('./weight/dinov2_vitb14_pretrain.pth', map_location='cpu'))
        self.dino.requires_grad_(False)
        self.dino.eval()
        
        self.pc_cfg = pc_cfg
        self.allow_legacy_pc_init = allow_legacy_pc_init
        self.training_design = training_design
        self.teacher = Decoder(pc_cfg=pc_cfg)  # Teacher Network Decoder
        self.student = Decoder(
            pc_cfg=None if training_design == 'teacher_only' else pc_cfg
        )  # Student Network Decoder

        self.load_teacher(teacher_pth)
        if student_pth is not None:
            self.load_student(student_pth)
        elif training_design == 'teacher_only':
            self._initialize_raw_student_from_teacher()
        else:
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
        distill_features = aux.get('distill_features')
        if not isinstance(distill_features, dict) or not all(
            torch.is_tensor(distill_features.get(key))
            for key in ('p3_corr', 'p2_refined')
        ):
            raise RuntimeError(
                'Teacher PC-HBM path did not return P3/P2 distillation features.'
            )
        if self.training_design == 'joint':
            p1_aux = aux.get('p1_pra')
            required_p1_targets = (
                'B1',
                'G1_raw_map',
                'R1_map',
                'O1_map',
                'R_sup_map',
                'valid1_map',
            )
            if not isinstance(p1_aux, dict) or not all(
                torch.is_tensor(p1_aux.get(key)) for key in required_p1_targets
            ):
                raise RuntimeError(
                    'Joint Teacher PC-HBM path did not return complete P1 '
                    'distillation targets.'
                )
        return aux

    def student_labeled(self, features, memory, epoch, query_image_ids=None):
        if self.training_design == 'teacher_only':
            return self.student(features, pc_mode='off', return_aux=True)
        return self.student(
            features,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
            query_image_ids=query_image_ids,
        )

    def student_unlabeled(self, features, memory, epoch):
        if self.training_design == 'teacher_only':
            return self.student(features, pc_mode='off', return_aux=True)
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
        if self.training_design == 'teacher_only' or self.student.pc_hbm is None:
            return self.student(x_features, pc_mode='off')[3]
        if memory is None:
            warnings.warn(
                'PC-HBM memory is missing; using z_main logits.',
                RuntimeWarning,
                stacklevel=2,
            )
            return self.student(x_features, pc_mode='off')[3]
        _, aux = self.student(
            x_features,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
        )
        if not aux['pc_active']:
            warnings.warn(
                f'PC-HBM inference fallback: {aux.get("fallback_reason")}; '
                'using z_main logits.',
                RuntimeWarning,
                stacklevel=2,
            )
            return aux['z_main']
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

    def _initialize_raw_student_from_teacher(self):
        raw_state = {
            name: value
            for name, value in self.teacher.state_dict().items()
            if not name.startswith('pc_hbm.')
        }
        self.student.load_state_dict(raw_state, strict=True)
        print('Initialized raw Student parameters from the Teacher legacy decoder.')

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
        update_ema_module(
            self.student,
            self.teacher,
            momentum=momentum,
            shared_only=self.training_design == 'teacher_only',
            exclude_prefixes=('pc_hbm.',),
        )

    def EMA(self, alpha=0.995):
        self.update_teacher(momentum=alpha)
