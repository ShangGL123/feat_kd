from .pgd_reg import PGDRegLoss
from .pgd_cls import PGDClsLoss
from .fgd import FGDLoss

__all__ = [
    'PGDRegLoss',
    'PGDClsLoss',
    'FGDLoss'
]