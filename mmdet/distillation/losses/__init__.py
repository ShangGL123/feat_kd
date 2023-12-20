from .pgd_reg import PGDRegLoss
from .pgd_cls import PGDClsLoss
from .fgd import FGDLoss
from .mgd import MGDLoss
from .fourier import FourierLoss
from .head import KDQualityFocalLoss, CrossEntropyLoss, IoULoss, GIoULoss
from .gt_cl import GTCLLoss

__all__ = [
    'PGDRegLoss',
    'PGDClsLoss',
    'FGDLoss',
    'MGDLoss',
    'FourierLoss',
    'KDQualityFocalLoss',
    'CrossEntropyLoss',
    'IoULoss',
    'GIoULoss',
    'GTCLLoss'
]