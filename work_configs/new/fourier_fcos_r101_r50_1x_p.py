_base_ = "../base/1x_setting.py"

add_conv = False
weight_p = 0.025
weight_a = 0

distiller = dict(
    type='DistillBaseDetector',
    teacher_pretrained = 'work_dirs/fcos_r101_3x_ms/epoch_36.pth',
    init_student = True,
    # init_student = False,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FourierLoss',
                                       name='loss_fourier_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       add_conv = add_conv,
                                       weight_p = weight_p,
                                       weight_a = weight_a,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FourierLoss',
                                       name='loss_fourier_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       add_conv = add_conv,
                                       weight_p = weight_p,
                                       weight_a = weight_a,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FourierLoss',
                                       name='loss_fourier_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       add_conv = add_conv,
                                       weight_p = weight_p,
                                       weight_a = weight_a,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FourierLoss',
                                       name='loss_fourier_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       add_conv = add_conv,
                                       weight_p = weight_p,
                                       weight_a = weight_a,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FourierLoss',
                                       name='loss_fourier_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       add_conv = add_conv,
                                       weight_p = weight_p,
                                       weight_a = weight_a,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'work_configs/detectors/fcos_r50_1x.py'
teacher_cfg = 'work_configs/detectors/fcos_r101_3x_ms.py'
