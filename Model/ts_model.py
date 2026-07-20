import torch
import torch.nn as nn
import warnings
from collections.abc import Mapping
from Model.BGFBR import ImageNetRGBAdapter
from Model.decoder import build_decoder, resolve_decoder_arch
from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderPCCoreResult,
    EncoderPCHBMAdapter,
    EncoderPCSegmentationHead,
    EncoderPCStageFlags,
    TeacherPseudoLabelRefiner,
)
from Model.PC_HBM.training.ema import update_ema_module
from utils.checkpoint_pc_hbm import (
    load_encoder_pc_checkpoint,
    load_decoder_compatible,
    save_decoder_checkpoint as save_decoder_artifact,
)
from utils.pc_memory_runner import module_fingerprint

class TSModel(nn.Module):
    VALID_TRAINING_DESIGNS = {'teacher_only', 'joint'}

    def __init__(
        self,
        teacher_pth=None,
        student_pth=None,
        pc_cfg=None,
        allow_legacy_pc_init=False,
        training_design='teacher_only',
        decoder_arch=None,
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
        self.decoder_arch = resolve_decoder_arch(decoder_arch, pc_cfg)
        self.pc_placement = str(getattr(pc_cfg, 'pc_placement', 'decoder'))
        if self.pc_placement not in {'decoder', 'encoder'}:
            raise ValueError(
                f'Unsupported pc_placement={self.pc_placement!r}; expected '
                "'decoder' or 'encoder'."
            )
        self.allow_legacy_pc_init = allow_legacy_pc_init
        self.training_design = training_design
        self.rgb_adapter = ImageNetRGBAdapter()
        self.teacher = build_decoder(
            self.decoder_arch,
            pc_cfg=pc_cfg,
            attach_pc=self.pc_placement == 'decoder',
        )
        self.student = build_decoder(
            self.decoder_arch,
            pc_cfg=pc_cfg,
            attach_pc=(
                self.pc_placement == 'decoder'
                and training_design != 'teacher_only'
            ),
        )

        self.encoder_pc_profile_v3 = (
            self.pc_placement == 'encoder'
            and isinstance(pc_cfg, EncoderPCHBMConfig)
        )
        self.teacher_encoder_pc_hbm = None
        self.student_encoder_pc_hbm = None
        self.teacher_pseudo_refiner = None
        self.student_pseudo_refiner = None
        self.teacher_encoder_pc_head = None
        self.student_encoder_pc_head = None
        self.encoder_base_artifact_meta = None

        if getattr(self, 'encoder_pc_profile_v3', False):
            if training_design != 'teacher_only':
                raise ValueError(
                    'encoder_pc TS uses the fixed EMA Teacher/Student protocol; '
                    "set training_design='teacher_only'."
                )
            if allow_legacy_pc_init:
                raise ValueError('encoder_pc TS never permits legacy PC initialization.')
            if student_pth is not None:
                raise ValueError(
                    'encoder_pc TS initializes both roles from one Base v3 artifact; '
                    'student_pth is not supported.'
                )
            self.teacher_encoder_pc_hbm = EncoderPCHBMAdapter(pc_cfg)
            self.student_encoder_pc_hbm = EncoderPCHBMAdapter(pc_cfg)
            self.teacher_pseudo_refiner = TeacherPseudoLabelRefiner(pc_cfg)
            self.student_pseudo_refiner = TeacherPseudoLabelRefiner(pc_cfg)
            object.__setattr__(
                self,
                'teacher_encoder_pc_head',
                EncoderPCSegmentationHead(
                    self.teacher_encoder_pc_hbm,
                    self.teacher,
                    self.teacher_pseudo_refiner,
                ),
            )
            object.__setattr__(
                self,
                'student_encoder_pc_head',
                EncoderPCSegmentationHead(
                    self.student_encoder_pc_hbm,
                    self.student,
                    self.student_pseudo_refiner,
                ),
            )
            self._load_encoder_pc_base(teacher_pth)
            self._freeze_encoder_teacher()
            return

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
        if getattr(self, 'encoder_pc_profile_v3', False):
            self.teacher_encoder_pc_hbm.eval()
            self.teacher_pseudo_refiner.eval()
        return self

    @torch.no_grad()
    def extract_feature_bundle(self, x):
        if x.dim() != 4:
            raise ValueError(f'Expected image batch [B,C,H,W], got {tuple(x.shape)}.')
        self.dino.eval()
        layer_indices = (
            tuple(self.pc_cfg.dino_layer_indices)
            if self.pc_cfg is not None
            else (2, 5, 8, 11)
        )
        features = self.dino.get_intermediate_layers(
            x=x,
            n=layer_indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        if len(features) != 4:
            raise RuntimeError(
                f'DINO returned {len(features)} feature levels instead of four.'
            )

        patch_tokens = []
        cls_tokens = []
        for level, feature_pair in enumerate(features, start=1):
            if not isinstance(feature_pair, (tuple, list)) or len(feature_pair) != 2:
                raise RuntimeError(
                    f'DINO level {level} did not return a (patch_tokens, cls_token) pair.'
                )
            patch, cls = feature_pair
            patch_tokens.append(patch)
            cls_tokens.append(cls)
        return DinoFeatureBundle(
            patch_tokens=tuple(patch_tokens),
            cls_tokens=tuple(cls_tokens),
        ).validate()

    @torch.no_grad()
    def extract_features(self, x):
        """Return the historical four-level patch-only DINO interface."""

        return self.extract_feature_bundle(x).patch_tokens

    def _extract_features(self, x):
        return self.extract_features(x)

    def prepare_rgb(self, x):
        """Invert ImageNet normalization once for all TS decoder branches."""

        return self.rgb_adapter(x)

    @torch.inference_mode()
    def teacher_pseudo(self, features, memory, epoch, image_rgb=None):
        if getattr(self, 'encoder_pc_profile_v3', False):
            if not isinstance(features, DinoFeatureBundle):
                raise TypeError('encoder_pc teacher_pseudo requires DinoFeatureBundle.')
            payload = self.teacher_encoder_pc_head(
                role='teacher_pseudo',
                bundle=features,
                image_rgb=image_rgb,
                memory=memory,
                stage=self._encoder_full_stage(require_same_image_positive=False),
                epoch=epoch,
                return_aux=True,
            )
            encoder_aux = payload['aux'].get('encoder_pc_hbm')
            if not isinstance(encoder_aux, Mapping):
                raise RuntimeError('Teacher pseudo payload is missing encoder evidence.')
            return {
                **payload,
                'encoder_pc_hbm': encoder_aux,
            }
        if getattr(self, 'pc_placement', 'decoder') == 'encoder':
            _, aux = self.teacher(
                features,
                image_rgb=image_rgb,
                memory=None,
                pc_mode='off',
                return_aux=True,
            )
            return aux
        _, aux = self.teacher(
            features,
            image_rgb=image_rgb,
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

    def student_labeled(
        self, features, memory, epoch, query_image_ids=None, image_rgb=None
    ):
        if self.encoder_pc_profile_v3:
            if not isinstance(features, DinoFeatureBundle):
                raise TypeError('encoder_pc student_labeled requires DinoFeatureBundle.')
            core = self.student_encoder_pc_head(
                role='labeled_core',
                bundle=features,
                image_rgb=image_rgb,
                memory=memory,
                mode='full',
                stage=self._encoder_full_stage(require_same_image_positive=True),
                epoch=epoch,
                query_image_ids=query_image_ids,
                return_aux=True,
            )
            if not isinstance(core, EncoderPCCoreResult):
                raise RuntimeError('Student labeled core returned an invalid result.')
            refined = self.student_encoder_pc_head(
                role='labeled_refiner',
                core_result=core,
                epoch=epoch,
            )
            aux = dict(core.aux)
            aux['z_core'] = core.z_core
            aux['pseudo_refiner'] = refined
            return core.outputs, aux
        if getattr(self, 'pc_placement', 'decoder') == 'encoder':
            return self.student(
                features,
                image_rgb=image_rgb,
                memory=None,
                pc_mode='off',
                return_aux=True,
            )
        if self.training_design == 'teacher_only':
            return self.student(
                features, image_rgb=image_rgb, pc_mode='off', return_aux=True
            )
        return self.student(
            features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
            query_image_ids=query_image_ids,
        )

    def student_unlabeled(self, features, memory, epoch, image_rgb=None):
        if self.encoder_pc_profile_v3:
            if not isinstance(features, DinoFeatureBundle):
                raise TypeError('encoder_pc student_unlabeled requires DinoFeatureBundle.')
            core = self.student_encoder_pc_head(
                role='student_core',
                bundle=features,
                image_rgb=image_rgb,
                memory=memory,
                stage=self._encoder_full_stage(require_same_image_positive=False),
                epoch=epoch,
                return_aux=True,
            )
            if not isinstance(core, EncoderPCCoreResult):
                raise RuntimeError('Student unlabeled core returned an invalid result.')
            aux = dict(core.aux)
            aux['z_core'] = core.z_core
            # This explicit sentinel is checked by encoder_pc_unlabeled_loss.
            aux['pseudo_refiner'] = None
            return core.outputs, aux
        if getattr(self, 'pc_placement', 'decoder') == 'encoder':
            return self.student(
                features,
                image_rgb=image_rgb,
                memory=None,
                pc_mode='off',
                return_aux=True,
            )
        if self.training_design == 'teacher_only':
            return self.student(
                features, image_rgb=image_rgb, pc_mode='off', return_aux=True
            )
        return self.student(
            features,
            image_rgb=image_rgb,
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
        image_rgb=None,
    ):
        if branch is not None:
            if features is None:
                raise ValueError('Precomputed DINO features are required for branch dispatch.')
            if branch == 'student_labeled':
                return self.student_labeled(
                    features,
                    memory,
                    epoch,
                    query_image_ids=query_image_ids,
                    image_rgb=image_rgb,
                )
            if branch == 'student_unlabeled':
                return self.student_unlabeled(
                    features, memory, epoch, image_rgb=image_rgb
                )
            raise ValueError(f'Unsupported TS forward branch: {branch!r}.')

        if getattr(self, 'encoder_pc_profile_v3', False):
            raise ValueError(
                'encoder_pc v3 requires explicit student_labeled or '
                'student_unlabeled branch dispatch.'
            )

        # Original combined/off API retained for the legacy SAM and pseudo trainers.
        if l_x is None or u_x is None:
            raise ValueError('Legacy TS forward requires both l_x and u_x.')
        l_x_features = self.extract_features(l_x)
        u_x_features = self.extract_features(u_x)
        l_rgb = self.prepare_rgb(l_x)
        u_rgb = self.prepare_rgb(u_x)
        with torch.no_grad():
            teacher_label = torch.sigmoid(
                self.teacher(u_x_features, image_rgb=u_rgb, pc_mode='off')[3].detach()
            )
        l_segs = list(self.student(l_x_features, image_rgb=l_rgb, pc_mode='off'))
        u_segs = list(self.student(u_x_features, image_rgb=u_rgb, pc_mode='off'))
        return l_segs, u_segs, teacher_label

    
    def inference(self, x, memory=None, epoch=None):
        if self.encoder_pc_profile_v3:
            bundle = self.extract_feature_bundle(x)
            image_rgb = self.prepare_rgb(x)
            return self.student_encoder_pc_head(
                role='inference',
                bundle=bundle,
                image_rgb=image_rgb,
                memory=memory,
                stage=self._encoder_full_stage(require_same_image_positive=False),
                epoch=epoch,
                return_aux=False,
            )
        x_features = self.extract_features(x)
        image_rgb = self.prepare_rgb(x)
        if self.training_design == 'teacher_only' or self.student.pc_hbm is None:
            return self.student(x_features, image_rgb=image_rgb, pc_mode='off')[3]
        if memory is None:
            warnings.warn(
                'PC-HBM memory is missing; using z_main logits.',
                RuntimeWarning,
                stacklevel=2,
            )
            return self.student(x_features, image_rgb=image_rgb, pc_mode='off')[3]
        _, aux = self.student(
            x_features,
            image_rgb=image_rgb,
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
        if self.encoder_pc_profile_v3:
            raise RuntimeError('encoder_pc Teacher must be loaded from a complete Base v3 artifact.')
        if path is None:
            raise ValueError('Teacher checkpoint path is required.')
        assert path.endswith('.pth'), f'Teacher parameters path should end with ".pth", but got: "{path}"'
        self._load_decoder(self.teacher, path, role='teacher')
        print(f'Successfully load teacher parameters from "{path}".')
        self.teacher.requires_grad_(False)
        print('Teacher parameters are not trainable.')

    def load_student(self, path):
        if self.encoder_pc_profile_v3:
            raise RuntimeError('encoder_pc Student must be loaded from a complete Base v3 artifact.')
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
        print('Initialized raw Student parameters from the Teacher decoder.')

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
        if self.encoder_pc_profile_v3:
            raise RuntimeError(
                'encoder_pc Student must be saved as a complete v3 adapter/decoder/refiner artifact.'
            )
        assert path.endswith('.pth'), f'Student parameters path should end with ".pth", but got: "{path}"'
        save_decoder_artifact(path, self.student, self.pc_cfg, epoch=0)
        print(f'Successfully save student parameters to {path}.')

    @torch.no_grad()
    def update_teacher(self, momentum=0.995):
        if self.encoder_pc_profile_v3:
            for student_module, teacher_module in (
                (self.student_encoder_pc_hbm, self.teacher_encoder_pc_hbm),
                (self.student, self.teacher),
                (self.student_pseudo_refiner, self.teacher_pseudo_refiner),
            ):
                update_ema_module(
                    student_module,
                    teacher_module,
                    momentum=momentum,
                    shared_only=False,
                )
            self._freeze_encoder_teacher()
            return
        update_ema_module(
            self.student,
            self.teacher,
            momentum=momentum,
            shared_only=self.training_design == 'teacher_only',
            exclude_prefixes=('pc_hbm.',),
        )

    def EMA(self, alpha=0.995):
        self.update_teacher(momentum=alpha)

    @staticmethod
    def _encoder_full_stage(*, require_same_image_positive=False):
        return EncoderPCStageFlags(
            enable_f4_f3=True,
            f4_f3_progress=1.0,
            enable_f2_f1=True,
            f2_f1_progress=1.0,
            require_same_image_positive=bool(require_same_image_positive),
        )

    def _load_encoder_pc_base(self, path):
        if path is None:
            raise ValueError('encoder_pc TS requires a Base encoder-PC v3 artifact.')
        checkpoint = load_encoder_pc_checkpoint(
            path,
            encoder_pc_hbm=self.student_encoder_pc_hbm,
            decoder=self.student,
            pseudo_refiner=self.student_pseudo_refiner,
            expected_model_role='base',
            expected_training_design='two_stage',
            expected_config=self.pc_cfg,
        )
        if int(checkpoint.get('epoch', -1)) != int(self.pc_cfg.final_epoch):
            raise RuntimeError(
                'encoder_pc TS requires the final Base v3 artifact at epoch '
                f'{self.pc_cfg.final_epoch}.'
            )
        artifact_meta = checkpoint.get('artifact_meta')
        if not isinstance(artifact_meta, Mapping):
            raise RuntimeError('Base encoder-PC v3 artifact has no metadata mapping.')
        for field in (
            'split_fingerprint',
            'producer_fingerprint',
            'dino_weight_fingerprint',
        ):
            if not isinstance(artifact_meta.get(field), str) or not artifact_meta[field]:
                raise RuntimeError(
                    f'Final Base encoder-PC v3 artifact is missing {field}.'
                )
        loaded_producer = module_fingerprint(self.student_encoder_pc_hbm)
        if artifact_meta['producer_fingerprint'] != loaded_producer:
            raise RuntimeError(
                'Base encoder-PC artifact producer fingerprint does not match '
                'its loaded Adapter.'
            )
        live_dino = module_fingerprint(self.dino)
        if artifact_meta['dino_weight_fingerprint'] != live_dino:
            raise RuntimeError(
                'Base encoder-PC artifact DINO fingerprint does not match '
                'the live frozen DINO.'
            )
        self.teacher_encoder_pc_hbm.load_state_dict(
            self.student_encoder_pc_hbm.state_dict(), strict=True
        )
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher_pseudo_refiner.load_state_dict(
            self.student_pseudo_refiner.state_dict(), strict=True
        )
        self.encoder_base_artifact_meta = dict(artifact_meta)

    def _freeze_encoder_teacher(self):
        for module in (
            self.teacher_encoder_pc_hbm,
            self.teacher,
            self.teacher_pseudo_refiner,
        ):
            module.requires_grad_(False)
            module.eval()
