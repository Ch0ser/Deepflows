from .module import Module
from .. import functional as F
from ...tensor import Tensor
from typing import Callable, Optional


__all__ = ['L1Loss', 'NLLLoss', 'MSELoss', 'BCELoss', 'CrossEntropyLoss']

"""
实际上的全部损失函数
__all__ = ['L1Loss', 'NLLLoss', 'NLLLoss2d', 'PoissonNLLLoss', 'GaussianNLLLoss', 'KLDivLoss',
           'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss',
           'SmoothL1Loss', 'HuberLoss', 'SoftMarginLoss', 'CrossEntropyLoss', 'MultiLabelSoftMarginLoss',
           'CosineEmbeddingLoss', 'MarginRankingLoss', 'MultiMarginLoss', 'TripletMarginLoss',
           'TripletMarginWithDistanceLoss', 'CTCLoss']
"""


class _Loss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        assert self.reduction in {'mean', 'sum'}

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class L1Loss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)


class NLLLoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, reduction=self.reduction)


class MSELoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        
    def forward(self, input: Tensor, target: Tensor):
        return F.mse_loss(input, target, reduction=self.reduction)


class BCELoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor):
        return F.binary_cross_entropy(input, target, reduction=self.reduction)


class CrossEntropyLoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, reduction=self.reduction)

