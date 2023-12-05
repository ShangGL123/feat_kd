from .pgd_reg import PGDRegLoss
from .pgd_cls import PGDClsLoss
from .fgd import FGDLoss
from .mgd import MGDLoss
from .fourier import FourierLoss
from .head import KDQualityFocalLoss, IoULoss
from .gt_cl import GTCLLoss

__all__ = [
    'PGDRegLoss',
    'PGDClsLoss',
    'FGDLoss',
    'MGDLoss',
    'FourierLoss',
    'KDQualityFocalLoss',
    'IoULoss',
    'GTCLLoss'
]