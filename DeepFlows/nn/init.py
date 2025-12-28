from numpy.random import uniform, normal
import math
import numpy as np
from ..autograd import no_grad
from .. import BackendTensor
import warnings

from typing import Optional as _Optional

from ..tensor import Tensor


def _no_grad_uniform_(tensor, a, b):
    with no_grad():
        data = uniform(a, b, tensor.shape)
        tensor.data.device.from_numpy(np.ascontiguousarray(data), tensor.data._handle)
        return tensor


def _no_grad_normal_(tensor, mean, std):
    with no_grad():
        data = normal(mean, std, size=tensor.shape)
        tensor.data.device.from_numpy(np.ascontiguousarray(data), tensor.data._handle)
        return tensor


def _no_grad_fill_(tensor, val):
    with no_grad():
        tensor.data.fill(val)
        return tensor


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        > gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1

    if dimensions > 2:
        receptive_field_size = math.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    return _no_grad_normal_(tensor, mean, std)


def uniform_(tensor: Tensor, low: float = 0., high: float = 1.0) -> Tensor:
    return _no_grad_uniform_(tensor, low, high)


def fill_(tensor: Tensor, val: float) -> Tensor:
    return _no_grad_fill_(tensor, val)


def zeros_(tensor: Tensor) -> Tensor:
    return _no_grad_fill_(tensor, 0.)


def ones_(tensor) -> Tensor:
    return _no_grad_fill_(tensor, 1.)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6. / (fan_in + fan_out))
    return _no_grad_uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std=std)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out 


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _no_grad_normal_(tensor, 0, std)
