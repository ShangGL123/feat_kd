import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmdet.distillation.builder import DISTILL_LOSSES
from mmdet.models.builder import build_loss
from mmdet.models.losses.utils import weighted_loss
import mmcv

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def kd_focal_loss(pred,
                target,
                weight=None,
                beta=1,
                reduction='mean',
                avg_factor=None):

    target = target.detach()
    loss = F.binary_cross_entropy(pred, target, reduction='none')
    return loss

@DISTILL_LOSSES.register_module()
class GTCLLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 gt_bbox_rescale=False,
                 T=0.07,
                 loss_cl_weight=1,
                 loss_cl=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.05),
                 ):
        super(GTCLLoss, self).__init__()
        self.gt_bbox_rescale = gt_bbox_rescale
        self.T = T
        self.loss_cl = build_loss(loss_cl)
        self.loss_cl_weight = loss_cl_weight

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        
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
        loss_cl = 0
        for i in range(N):
            preds_S_perimg = preds_S[i].permute(1,2,0).reshape(-1, C)
            preds_S_perimg = torch.nn.functional.normalize(preds_S_perimg, dim=1)

            preds_T_perimg = preds_T[i].permute(1,2,0).reshape(-1, C)
            preds_T_perimg = torch.nn.functional.normalize(preds_T_perimg, dim=1)

            relation_T = preds_T_perimg @ preds_T_perimg.T
            # import ipdb;ipdb.set_trace()
            relation_weight = 1-relation_T  + torch.eye(H*W, H*W).cuda(relation_T.device)

            # compute logits
            # Einstein sum is more intuitive
            logits = torch.einsum('nc,ck->nk', [preds_S_perimg, preds_T_perimg.T])
            # apply temperature
            # logits /= self.T

            # labels: positive key indicators
            # labels = torch.arange(logits.shape[0]).cuda(logits.device)
            labels = torch.eye(logits.shape[0], logits.shape[0]).cuda(logits.device)
            # import ipdb;ipdb.set_trace()
            loss_cl += kd_focal_loss(logits.clamp(min=1e-5), relation_weight) * self.loss_cl_weight
        return loss_cl
