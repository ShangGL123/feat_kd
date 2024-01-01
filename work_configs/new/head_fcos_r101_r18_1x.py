_base_ = "../base/1x_setting.py"

alpha_mgd=0
lambda_mgd=0.65

distiller = dict(
    type='DistillHeadBaseDetector',
    teacher_pretrained = 'work_dirs/fcos_r101_3x_ms/epoch_36.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       )
                                ]
                        ),
                    dict(loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=0.4),
                         loss_reg_kd=dict(type='IoULoss', loss_weight=0.75),
                        ),
                   ]
    )

student_cfg = 'work_configs/detectors/fcos_r18_1x.py'
teacher_cfg = 'work_configs/detectors/fcos_r101_3x_ms.py'
