from .module import Module
from .. import functional as F
from typing import Optional, Tuple
from ...tensor import Tensor
from ..parameter import Parameter

__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'GELU', 'LeakyReLU', 'Softmax', 'LogSoftmax']

"""
实际上全部的：
__all__ = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh',
           'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU',
           'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Tanhshrink',
           'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
"""


class ReLU(Module):
    r"""

    Examples::

        > m = nn.ReLU()
        > input = randn(2)
        > output = m(input)
        > print(input)
        > print(output)

    在上面的例子中:
    假设input的初始值为[-1, 2]
    inplace=True -> 打印的input为[0, 2], output为[0, 2]
    inplace=False -> 打印的input为[-1, 2], output为[0, 2]

    总的来说，就是为True的话，原地执行操作，被操作变量与操作得到的变量的内存与值是绑定在一起的
    output的改变会引起input的改变
    False的话，output与input就是分开的，二者互不影响
    
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class Sigmoid(Module):
    r"""
    
    \frac{1}{1 + \exp(-x)}

    Examples::

        > m = nn.Sigmoid()
        > input = randn(2)
        > output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(Module):
    r"""
    
    \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Examples::

        > m = nn.Tanh()
        > input = randn(2)
        > output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class GELU(Module):
    r"""
    
    GELU(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``
            
    Examples::

        > m = nn.GELU()
        > input = randn(2)
        > output = m(input)
    """

    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'


class LeakyReLU(Module):
    r"""
    
    \max(0, x) + \text{negative\_slope} * \min(0, x)
        
    Examples::

        > m = nn.LeakyReLU(0.1)
        > input = randn(2)
        > output = m(input)
    """

    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope)


class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:
    
    Softmax(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        > m = nn.Softmax(dim=1)
        > input = randn(2, 3)
        > output = m(input)

    """

    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(Module):
    r"""
    Examples::

        > m = nn.LogSoftmax(dim=1)
        > input = torch.randn(2, 3)
        > output = m(input)
    """

    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim, )

    def extra_repr(self):
        return f'dim={self.dim}'
