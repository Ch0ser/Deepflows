from .module import Module
from ..parameter import Parameter
from .. import functional as F
from ...tensor import empty, Tensor
from ... import backend_api
from .. import init
import math


class Linear(Module):
    r""""
    y = xA^T + b`

    Examples::

        > m = nn.Linear(20,30)
        > input = randn(128, 20)
        > output = m(input)
        >print(output.size())
        size([128, 30])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device="cuda",
            dtype="float32"
    ) -> None:
        factory_kwargs = {'device': backend_api.Device(device), 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(empty((1, out_features), **factory_kwargs))
        else:
            # 这是Module类中的方法
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        对weight, bias(if not None)进行初始化
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def move(self, device):
        self.device = device
        self.weight = self.weight.to(self.device)
        self.bias = self.bias.to(self.device)
