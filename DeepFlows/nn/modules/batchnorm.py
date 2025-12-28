from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from ... import tensor
from ... import backend_api


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device: str = 'cuda', dtype=None) -> None:
        super().__init__()
        kwargs = {"device": backend_api.Device(device), "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(tensor.ones((1, num_features, 1, 1), **kwargs))
            self.bias = Parameter(tensor.zeros((1, num_features, 1, 1), **kwargs))
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.running_mean = tensor.zeros((1, num_features, 1, 1), **kwargs)
            self.running_var = tensor.ones((1, num_features, 1, 1), **kwargs)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            n, c, h, w = x.shape
            mean = tensor.sum(x, axes=0, keepdims=True)
            mean = tensor.sum(mean, axes=2, keepdims=True)
            mean = tensor.sum(mean, axes=3, keepdims=True)
            mean = mean / (n * h * w)

            diff = x - mean
            var = tensor.sum(diff * diff, axes=0, keepdims=True)
            var = tensor.sum(var, axes=2, keepdims=True)
            var = tensor.sum(var, axes=3, keepdims=True)
            var = var / (n * h * w)

            if self.track_running_stats:
                self.running_mean.data = self.running_mean.data * (1 - self.momentum) + mean.data * self.momentum
                self.running_var.data = self.running_var.data * (1 - self.momentum) + var.data * self.momentum
            x_hat = (x - mean) / (var + self.eps) ** 0.5
        else:
            if self.track_running_stats:
                x_hat = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
            else:
                x_hat = x
        if self.affine:
            return x_hat * self.weight + self.bias
        return x_hat

    def __repr__(self) -> str:
        return "{}(num_features={}, eps={}, momentum={}, affine={}, track_running_stats={})".format(
            self.__class__.__name__,
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )