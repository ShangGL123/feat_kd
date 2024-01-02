import torch.nn as nn
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from mmcv.runner import load_checkpoint, load_state_dict
from mmdet.distillation.builder import DISTILLER, build_distill_loss
from collections import OrderedDict


@DISTILLER.register_module()
class DistillHeadBaseDetector(BaseDetector):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """

    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False,
                 stu_lambda=1.,
                 spatial_and_channel=False,):

        super(DistillHeadBaseDetector, self).__init__()

        self.teacher = build_detector(teacher_cfg.model,
                                      train_cfg=teacher_cfg.get('train_cfg'),
                                      test_cfg=teacher_cfg.get('test_cfg'))
        self.teacher_pretrained = teacher_pretrained
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = build_detector(student_cfg.model,
                                      train_cfg=student_cfg.get('train_cfg'),
                                      test_cfg=student_cfg.get('test_cfg'))

        self.init_student = init_student
        self.stu_lambda = stu_lambda
        self.spatial_and_channel = spatial_and_channel

        self.distill_losses = nn.ModuleDict()

        self.distill_cfg = distill_cfg
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        # self.count=0

        def regitster_hooks(student_module, teacher_module):
            def hook_teacher_forward(module, input, output):
                self.register_buffer(teacher_module, output)

            def hook_student_forward(module, input, output):
                self.register_buffer(student_module, output)

            return hook_teacher_forward, hook_student_forward

        if type(distill_cfg) is list:
            for item_loc in distill_cfg:
                assert isinstance(item_loc, dict)
                if 'output_hook' in item_loc: # featrure imitate
                    student_module = 'student_' + item_loc.student_module.replace('.', '_')
                    teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

                    self.register_buffer(student_module, None)
                    self.register_buffer(teacher_module, None)

                    hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
                    teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                    student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        self.distill_losses[loss_name] = build_distill_loss(item_loss)
                else:
                    self.loss_cls_kd = build_distill_loss(item_loc['loss_cls_kd'])
                    self.distill_losses['loss_cls_kd'] = self.loss_cls_kd
                    self.loss_reg_kd = build_distill_loss(item_loc['loss_reg_kd'])
                    self.distill_losses['loss_reg_kd'] = self.loss_reg_kd
                    

            
    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        device_id = torch.cuda.current_device()
        checkpoint = load_checkpoint(self.teacher, self.teacher_pretrained, map_location=torch.device('cuda', device_id))
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        if self.init_student:
            all_name = []
            for name, v in checkpoint["state_dict"].items():
                if not name.startswith("backbone."):
                    all_name.append((name, v))
            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
    
    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3), None

    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        tea_feats = self.teacher.extract_feat(img)
        tea_cls_scores, tea_bbox_preds, tea_centernesses = self.teacher.bbox_head(tea_feats)

        
        # vis_img = img[0].permute(1,2,0).cpu().numpy().copy()
        # import numpy as np
        # import cv2
        # vis_img =  vis_img * np.array([58.395, 57.12, 57.375]).reshape(1,1,3)
        # vis_img = vis_img + np.array([123.675, 116.28, 103.53]).reshape(1,1,3)
        # vis_img = vis_img[...,::-1].copy()
        # for i, cls_feat in enumerate(tea_cls_scores):
        #     cls_feat = cls_feat.permute(0, 2, 3, 1)
        #     cls_feat = cls_feat.max(dim=-1)[0].sigmoid()
        #     cls_feat = cls_feat.permute(1,2,0).cpu().numpy().copy()

        #     cls_feat = (cls_feat * 255).astype(np.uint8)
        #     cls_feat = cv2.applyColorMap(cls_feat, cv2.COLORMAP_PLASMA)
        #     desired_size = (vis_img.shape[1], vis_img.shape[0])
        #     cls_feat = cv2.resize(cls_feat, desired_size)
        #     # vis_img = cls_feat
        #     vis_img_out = vis_img + cls_feat
        #     print(cv2.imwrite('/home/cjh/projects/PGD/work_dirs/vis/cnn_vis{0}.jpg'.format(self.count + i), vis_img_out))
        # self.count += 10    



        # vis_img = img[0].permute(1,2,0).cpu().numpy().copy()
        # import numpy as np
        # import cv2
        # vis_img =  vis_img * np.array([58.395, 57.12, 57.375]).reshape(1,1,3)
        # vis_img = vis_img + np.array([123.675, 116.28, 103.53]).reshape(1,1,3)
        # vis_img = vis_img[...,::-1].copy()
        # out = self.teacher.bbox_head.vis_assign(tea_feats, img_metas, **kwargs)
        # for i, out_i in enumerate(out):
        #     h, w = tea_feats[i].shape[2], tea_feats[i].shape[3]
        #     out_i[out_i<80]=1
        #     out_i[out_i==80]=0
        #     out_i = out_i.reshape(h, w)[...,None].cpu().numpy().copy()

        #     cls_feat = (out_i * 255).astype(np.uint8)
        #     cls_feat = cv2.applyColorMap(cls_feat, cv2.COLORMAP_PLASMA)
        #     desired_size = (vis_img.shape[1], vis_img.shape[0])
        #     cls_feat = cv2.resize(cls_feat, desired_size)
        #     # vis_img = cls_feat
        #     vis_img_out = vis_img + cls_feat
        #     print(cv2.imwrite('/home/cjh/projects/PGD/work_dirs/vis_assign/cnn_vis{0}.jpg'.format(self.count + i), vis_img_out))
        # self.count += 10       


          
        stu_feats = self.student.extract_feat(img)
        student_loss = self.student.bbox_head.forward_train(stu_feats, img_metas, **kwargs)
        # stu_feats -> tea_head

        # mix stu_feats with tea_feats
        nums_level = len(stu_feats)
        mix_stu_feats_list = []
        for level in range(nums_level):
            N, C, H, W = tea_feats[level].shape
            device = tea_feats[level].device
            
            mat = torch.rand((N,1,H,W)).to(device)

            stu_mat = torch.where(mat>=self.stu_lambda, 0, 1).to(device)
            mix_stu_feat = torch.mul(stu_feats[level], stu_mat)
            mix_stu_feats_list.append(mix_stu_feat)
        mix_stu_feats = tuple(mix_stu_feats_list)

        stu_cls_scores, stu_bbox_preds, stu_centernesses = self.teacher.bbox_head(mix_stu_feats)

        
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            if 'output_hook' in item_loc: # featrure imitate
                student_module = 'student_' + item_loc.student_module.replace('.','_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
                
                student_feat = buffer_dict[student_module]
                teacher_feat = buffer_dict[teacher_module]

                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    
                    student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat, kwargs['gt_bboxes'], img_metas)
            else:
                loss_distill_head = self.loss_distill_head(tea_cls_scores, tea_bbox_preds, tea_centernesses, stu_cls_scores, stu_bbox_preds, stu_centernesses, img_metas, **kwargs)
                student_loss.update(loss_distill_head)
                
        return student_loss

    def loss_distill_head(self, tea_cls_scores, tea_bbox_preds, tea_centernesses,
                                stu_cls_scores, stu_bbox_preds, stu_centernesses, img_metas, **kwargs):
        gt_bboxes, gt_labels = kwargs['gt_bboxes'], kwargs['gt_labels']
        assert len(tea_cls_scores) == len(stu_cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in tea_cls_scores]
        all_level_points = self.teacher.bbox_head.get_points(featmap_sizes, tea_bbox_preds[0].dtype,
                                           tea_bbox_preds[0].device)
        labels, bbox_targets = self.teacher.bbox_head.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = tea_cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        # tea
        flatten_tea_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.teacher.bbox_head.cls_out_channels)
            for cls_score in tea_cls_scores
        ]
        flatten_tea_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in tea_bbox_preds
        ]
        flatten_tea_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in tea_centernesses
        ]
        flatten_tea_cls_scores = torch.cat(flatten_tea_cls_scores)
        flatten_tea_bbox_preds = torch.cat(flatten_tea_bbox_preds)
        flatten_tea_centerness = torch.cat(flatten_tea_centerness)

        # stu
        flatten_stu_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.teacher.bbox_head.cls_out_channels)
            for cls_score in stu_cls_scores
        ]
        flatten_stu_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in stu_bbox_preds
        ]
        flatten_stu_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in stu_centernesses
        ]
        flatten_stu_cls_scores = torch.cat(flatten_stu_cls_scores)
        flatten_stu_bbox_preds = torch.cat(flatten_stu_bbox_preds)
        flatten_stu_centerness = torch.cat(flatten_stu_centerness)


        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.teacher.bbox_head.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=tea_bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.teacher.bbox_head.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        

        losses_cls_kd = self.loss_cls_kd(flatten_stu_cls_scores, 
                                         flatten_tea_cls_scores, 
                                         avg_factor=centerness_denorm)
        
        flatten_tea_bbox_preds = distance2bbox(flatten_points, flatten_tea_bbox_preds)
        flatten_stu_bbox_preds = distance2bbox(flatten_points, flatten_stu_bbox_preds)
        
        
        reg_weights = flatten_tea_cls_scores.max(dim=1)[0].sigmoid()
        # import ipdb;ipdb.set_trace()

        losses_reg_kd = self.loss_reg_kd(flatten_stu_bbox_preds,
                                        flatten_tea_bbox_preds,
                                        weight=reg_weights, 
                                        avg_factor=centerness_denorm)


        return dict(
            losses_cls_kd=losses_cls_kd,
            losses_reg_kd=losses_reg_kd)


    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
