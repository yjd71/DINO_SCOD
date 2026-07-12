import torch
import torch.nn as nn
from Model.decoder import Decoder

class TSModel(nn.Module):
    def __init__(self, teacher_pth=None):
        super(TSModel, self).__init__()
        
        # initialize the DINOv2 model: DINOv2-ViT-B/14 (default)
        self.dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load('./weight/dinov2_vitb14_pretrain.pth', map_location='cpu'))
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        
        self.teacher = Decoder()  # Teacher Network Decoder
        self.student = Decoder()  # Student Network Decoder

        self.load_teacher(teacher_pth)
        self.load_student(teacher_pth)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        self.teacher.eval()
        return self

    def _extract_features(self, x):
        with torch.no_grad():
            return self.dino.get_intermediate_layers(x=x, n=[2, 5, 8, 11], reshape=False, return_class_token=False, norm=True)
    
    def forward(self, l_x, u_x):

        l_x_features = self._extract_features(l_x)
        u_x_features = self._extract_features(u_x)
        
        # Teacher Network
        _, _, _, teacher_label, _ = self.teacher(features = u_x_features)
        # use sigmoid to normalize to 0-1 pseudo labels, do not backpropagate gradients
        teacher_label = torch.sigmoid(teacher_label.detach())

        # Student Network
        l_seg_4, l_seg_3, l_seg_2, l_seg_1, l_seg_g = self.student(features = l_x_features)
        u_seg_4, u_seg_3, u_seg_2, u_seg_1, u_seg_g = self.student(features = u_x_features)

        l_segs = [l_seg_4, l_seg_3, l_seg_2, l_seg_1, l_seg_g]
        u_segs = [u_seg_4, u_seg_3, u_seg_2, u_seg_1, u_seg_g]

        return l_segs, u_segs, teacher_label

    
    def inference(self, x):
        x_features = self._extract_features(x)
        _, _, _, seg_1, _ = self.student(features = x_features)
        return seg_1

    def load_teacher(self, path):
        if path is None:
            raise ValueError('Teacher checkpoint path is required.')
        assert path.endswith('.pth'), f'Teacher parameters path should end with ".pth", but got: "{path}"'
        state_dict = torch.load(path, map_location='cpu')
        self.teacher.load_state_dict(state_dict)
        print(f'Successfully load teacher parameters from "{path}".')
        for param in self.teacher.parameters():
            param.requires_grad = False
        print('Teacher parameters are not trainable.')

    def load_student(self, path):
        if path is None:
            raise ValueError('Student initialization checkpoint path is required.')
        assert path.endswith('.pth'), f'Student parameters path should end with ".pth", but got: "{path}"'
        state_dict = torch.load(path, map_location='cpu')
        self.student.load_state_dict(state_dict)
        print(f'Successfully load student parameters from "{path}".')

    def save_student(self, path):
        assert path.endswith('.pth'), f'Student parameters path should end with ".pth", but got: "{path}"'
        torch.save(self.student.state_dict(), path)
        print(f'Successfully save student parameters to {path}.')

    def EMA(self, alpha=0.995):
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(alpha).add_((1 - alpha) * param_q.detach().data)
