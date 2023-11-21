import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmdet.distillation.builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class FourierLoss(nn.Module):

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 add_conv,
                 weight_p,
                 weight_a,
                 ):
        super(FourierLoss, self).__init__()
        self.add_conv = add_conv
        self.weight_p = weight_p
        self.weight_a = weight_a
        

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        # if self.add_conv:
        #     self. = nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1, padding=0)
        

    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_fre = torch.fft.rfft2(preds_S, norm='backward')
        S_mag = torch.abs(S_fre)
        S_pha = torch.angle(S_fre)

        T_fre = torch.fft.rfft2(preds_T, norm='backward')
        T_mag = torch.abs(T_fre)
        T_pha = torch.angle(T_fre)
        # import ipdb;ipdb.set_trace()

        loss_mse = nn.MSELoss(reduction='mean')

        phase_loss = loss_mse(S_pha, T_pha)

        softmax = torch.nn.Softmax(dim=1)
        amplitude_loss = loss_mse(softmax(S_mag), softmax(T_mag))
        
        return self.weight_p * phase_loss + self.weight_a * amplitude_loss
        # return loss_mse(preds_S, preds_T)

