import torch.nn as nn
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from mmcv.runner import load_checkpoint, load_state_dict
from mmdet.distillation.builder import DISTILLER, build_distill_loss
from collections import OrderedDict


@DISTILLER.register_module()
class DistillHeadBaseDetector_DDOD(BaseDetector):
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

        super(DistillHeadBaseDetector_DDOD, self).__init__()

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

        self.cls_num_pos_samples_per_level = [0. for ii in range(5)]
        self.reg_num_pos_samples_per_level = [0. for ii in range(5)]

        self.distill_losses = nn.ModuleDict()

        self.distill_cfg = distill_cfg
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

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
                    self.loss_iou_kd = build_distill_loss(item_loc['loss_iou_kd'])
                    self.distill_losses['loss_iou_kd'] = self.loss_iou_kd
                    

            
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
        tea_cls_scores, tea_bbox_preds, tea_ious = self.teacher.bbox_head(tea_feats)
           
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

        stu_cls_scores, stu_bbox_preds, stu_ious = self.teacher.bbox_head(mix_stu_feats)

        
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
                loss_distill_head = self.loss_distill_head(tea_cls_scores, tea_bbox_preds, tea_ious, stu_cls_scores, stu_bbox_preds, stu_ious, img_metas, **kwargs)
                student_loss.update(loss_distill_head)
                
        return student_loss

    def loss_distill_head(self, tea_cls_scores, tea_bbox_preds, tea_ious,
                                stu_cls_scores, stu_bbox_preds, stu_ious, img_metas, **kwargs):
        gt_bboxes, gt_labels = kwargs['gt_bboxes'], kwargs['gt_labels']
        assert len(tea_cls_scores) == len(stu_cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in tea_cls_scores]
        assert len(featmap_sizes) == self.teacher.bbox_head.anchor_generator.num_levels

        device = tea_cls_scores[0].device
        label_channels = self.teacher.bbox_head.cls_out_channels if self.teacher.bbox_head.use_sigmoid_cls else 1

        anchor_list, valid_flag_list = self.teacher.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)
        
        # cls分支结果
        cls_targets = self.teacher.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            tea_cls_scores,
            tea_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            is_cls=True)
        if cls_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_targets
        
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        
        # get pos samples for each level
        bg_class_ind = self.teacher.bbox_head.num_classes
        for ii, each_level_label in enumerate(labels_list):
            pos_inds = ((each_level_label >= 0)
                        & (each_level_label < bg_class_ind)).nonzero().squeeze(1)
            # num_pos_samples_per_level.append(len(pos_inds))
            self.cls_num_pos_samples_per_level[ii] += len(pos_inds)
        # get reweight factor from 1 ~ 2 with bilinear interpolation
        min_pos_samples = min(self.cls_num_pos_samples_per_level)
        max_pos_samples = max(self.cls_num_pos_samples_per_level)
        interval = 1. / (max_pos_samples - min_pos_samples + 1e-10)
        reweight_factor_per_level = []
        for pos_samples in self.cls_num_pos_samples_per_level:
            factor = 2. - (pos_samples - min_pos_samples) * interval
            reweight_factor_per_level.append(factor)

        cls_losses_cls_kd, cls_losses_bbox_kd, cls_losses_iou_kd = multi_apply(
                self.loss_distill_head_single,
                anchor_list,
                labels_list,
                label_weights_list,
                tea_cls_scores,
                tea_bbox_preds,
                tea_ious,
                stu_cls_scores,
                stu_bbox_preds,
                stu_ious,
                reweight_factor_per_level,
                avg_factor=num_total_samples)
        
        # reg分支结果
        anchor_list, valid_flag_list = self.teacher.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)
        
        reg_targets = self.teacher.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            tea_cls_scores,
            tea_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            is_cls=False)
        if reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = reg_targets
        
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        
        # get pos samples for each level
        bg_class_ind = self.teacher.bbox_head.num_classes
        for ii, each_level_label in enumerate(labels_list):
            pos_inds = ((each_level_label >= 0)
                        & (each_level_label < bg_class_ind)).nonzero().squeeze(1)
            # num_pos_samples_per_level.append(len(pos_inds))
            self.reg_num_pos_samples_per_level[ii] += len(pos_inds)
        # get reweight factor from 1 ~ 2 with bilinear interpolation
        min_pos_samples = min(self.reg_num_pos_samples_per_level)
        max_pos_samples = max(self.reg_num_pos_samples_per_level)
        interval = 1. / (max_pos_samples - min_pos_samples + 1e-10)
        reweight_factor_per_level = []
        for pos_samples in self.reg_num_pos_samples_per_level:
            factor = 2. - (pos_samples - min_pos_samples) * interval
            reweight_factor_per_level.append(factor)

        reg_losses_cls_kd, reg_losses_bbox_kd, reg_losses_iou_kd = multi_apply(
                self.loss_distill_head_single,
                anchor_list,
                labels_list,
                label_weights_list,
                tea_cls_scores,
                tea_bbox_preds,
                tea_ious,
                stu_cls_scores,
                stu_bbox_preds,
                stu_ious,
                reweight_factor_per_level,
                avg_factor=num_total_samples)

        return dict(
            loss_cls_kd=cls_losses_cls_kd,
            loss_bbox_kd=reg_losses_bbox_kd,
            loss_iou_kd=reg_losses_iou_kd)
    

    def loss_distill_head_single(self, 
                                anchors,  # [4, 13600, 4]
                                labels,     # [4, 13600]
                                label_weights,   # [4, 13600]
                                tea_cls_score,   # [4, 80, 100, 136]
                                tea_bbox_pred,    # [4, 4, 100, 136]
                                tea_iou,    # [4, 1, 100, 136]
                                stu_cls_score,
                                stu_bbox_pred,
                                stu_iou,
                                reweight_factor_per_level,
                                avg_factor):
        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.teacher.bbox_head.cls_out_channels)
        stu_cls_score = stu_cls_score.permute(0, 2, 3, 1).reshape(-1, self.teacher.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        
        loss_cls_kd = self.loss_cls_kd(
            stu_cls_score,
            tea_cls_score,
            weight=label_weights,
            avg_factor=avg_factor)
        
        # regression branch distillation
        bbox_coder = self.teacher.bbox_head.bbox_coder
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        stu_bbox_pred = stu_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        anchors = anchors.reshape(-1, 4)
        tea_bbox_pred = bbox_coder.decode(anchors, tea_bbox_pred)
        stu_bbox_pred = bbox_coder.decode(anchors, stu_bbox_pred)
        
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        reg_weights[label_weights == 0] = 0

        loss_reg_kd = self.loss_reg_kd(
            stu_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights,
            avg_factor=avg_factor)
        
        # centernesses branch distillation
        labels = labels.reshape(-1)
        bg_class_ind = self.teacher.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        tea_iou = tea_iou.permute(0, 2, 3, 1).reshape(-1)
        stu_iou = stu_iou.permute(0, 2, 3, 1).reshape(-1)

        if len(pos_inds) > 0:
            loss_iou_kd = self.loss_iou_kd(
                stu_iou[pos_inds],
                tea_iou[pos_inds].sigmoid(),
                avg_factor=avg_factor)
        else:
            loss_iou_kd = stu_iou.new_tensor(0.)
        return loss_cls_kd * reweight_factor_per_level, loss_reg_kd * reweight_factor_per_level, loss_iou_kd * reweight_factor_per_level


    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
