import numpy as np
from .. import tensor
from ..tensor import Tensor
from .. import backend_api
from typing import Callable, List, Optional, Tuple, Union


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    affine = input @ weight
    if bias is not None:
        affine = affine + bias
    return affine


def relu(input: Tensor) -> Tensor:
    return tensor.maximum(input, 0)


class sigmoid(tensor.UnaryOperator):
    def forward(self, input: Tensor):
        sigmoid = backend_api.zeros(input.shape, device=input.device)
        sigmoid[input.data > 0] = 1 / (1 + backend_api.exp(-input.data[input.data > 0]))
        sigmoid[input.data <= 0] = 1 - 1 / (1 + backend_api.exp(input.data[input.data <= 0]))
        return sigmoid

    def grad_fn(self, input: Tensor, grad):
        return self.data * (1 - self.data) * grad


class tanh(tensor.UnaryOperator):
    def forward(self, input: Tensor):
        return backend_api.tanh(input.data)

    def grad_fn(self, input: tensor.Tensor, grad):
        return (1 - self.data**2) * grad


class gelu:
    def __init__(self, input: Tensor, approximate: str = 'none') -> None:
        pass


def leaky_relu(input: Tensor, negative_slope: float):
    return tensor.maximum(input, input * negative_slope)


def softmax(input: Tensor, dim=None, keepdims=False):
    if dim is None:
        dim = 1
    m = tensor.max(input, dim, True)
    input_sub_max = input - m
    exp_ = tensor.exp(input_sub_max)
    denom = tensor.sum(exp_, dim, True)
    return exp_ / denom


def log_softmax(input: Tensor, dim=None, keepdims=False):
    if dim is None:
        dim = 1
    m = tensor.max(input, dim, True)
    input_sub_max = input - m
    logsumexp = tensor.log(tensor.sum(tensor.exp(input_sub_max), dim, True))
    return input_sub_max - logsumexp


def l1_loss(input: Tensor, target: Tensor, reduction: str = 'mean'):
    loss = tensor.abs(input - target)
    if reduction == 'mean':
        return tensor.mean(loss)
    elif reduction == 'sum':
        return tensor.sum(loss)
    else:
        assert 0, "reduction must be mean or sum."


def nll_loss(
    input: Tensor, 
    target: Tensor, 
    reduction: str = 'mean',
):
    nll = -input * target
    if reduction == 'mean':
        return tensor.mean(nll)
    elif reduction == 'sum':
        return tensor.sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean'):
    square_sum = tensor.square(input - target)
    if reduction == 'mean':
        return tensor.mean(square_sum)
    elif reduction == 'sum':
        return tensor.sum(square_sum),
    else:
        assert 0, "reduction must be mean or sum."


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean'):
    pass


def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean', dim: int = 1):
    m = tensor.max(input, dim, True)
    update_input = input - m
    log_sum_exp = tensor.log(tensor.sum(tensor.exp(update_input), dim, True))
    nll = -(update_input - log_sum_exp) * target
    if reduction == 'mean':
        per_sample = tensor.sum(nll, dim, True)
        return tensor.sum(per_sample) * (1.0 / input.shape[0])
    elif reduction == 'sum':
        return tensor.sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


class __im2col1d(tensor.UnaryOperator):
    def __init__(
            self,
            input: Tensor,
            kernel_size: int,
            stride: int,
    ) -> None:
        self.N, self.in_channels, self.n_features = input.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_output = (self.n_features - self.kernel_size) // stride + 1
        super().__init__(input)

    def forward(self, x: Tensor):
        col = backend_api.zeros(
            (self.N, self.in_channels, self.n_output, self.kernel_size, self.kernel_size), device=x.device)

        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            col[:, i] = x.data[:, i:i_max:self.stride]

        return col

    def grad_fn(self, input: Tensor, grad):
        grad_x = backend_api.zeros((self.N, self.in_channels, self.n_features), device=input.device)
        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            grad_x[:, i:i_max:self.stride] += grad[:, i]

        return grad_x


class __pad1d(tensor.UnaryOperator):
    def __init__(self, input: tensor.Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(input)

    def forward(self, x: tensor.Tensor):
        return backend_api.pad(x.data, [(0, 0), (0, 0),
                                    (self.pad_width, self.pad_width)])

    def grad_fn(self, x: tensor.Tensor, grad):
        if self.pad_width == 0:
            return grad
        # 修正：grad是3维（N,C,L_pad），补充通道维索引（: 表示保留所有通道）
        return grad[:, :, self.pad_width:-self.pad_width]  # 3个索引匹配3维


def conv1d(
        input: Tensor,
        kernel: Tensor,
        padding: int = 0,
        stride: int = 1,
):
    """"
    一维卷积函数

    基于im2col实现

    Parameters
    ----------
    input : Tensor
        输入数据，形状为(N, in_channels, n_features);
    kernel : Tensor
        卷积核，形状为(out_channels, in_channels, kernel_size);
    padding : int, default=0
        对输入特征两边补0数量;
    stride : int, default=1
        卷积步长.
    """
    kernel_size = kernel.shape[-1]
    pad_x = __pad1d(input, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return (col @ kernel.transpose(1, 2, 0)).sum(1).swapaxes(1, 2)


def max_pool1d(
        x: tensor.Tensor,
        kernel_size: int,
        stride: int,
        padding: int = 0,
):
    """
    一维池化函数

    基于im2col实现的一维池化.`

    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_features);
    kernel_size : int
        池化核大小;
    stride : int
        卷积步长;
    padding : int, default=0
        对输入特征两边补0数量.
    """
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return col.max(-1)


def avg_pool1d(
        x: tensor.Tensor,
        kernel_size: int,
        stride: int,
        padding: int = 0,
):
    """
    一维平均池化函数

    基于im2col实现的一维池化.`

    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_features);
    kernel_size : int
        池化核大小;
    stride : int
        卷积步长;
    padding : int, default=0
        对输入特征两边补0数量.
    """
    
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return col.mean(-1)


class __im2col2d(tensor.UnaryOperator):
    def __init__(
            self,
            x: tensor.Tensor,
            kernel_size: int,
            stride: int,
    ) -> None:
        self.N, self.in_channels, self.n_h, self.n_w = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_h, self.out_w = ((self.n_h - self.kernel_size) // self.stride + 1,
                                  (self.n_w - self.kernel_size) // self.stride + 1)
        super().__init__(x)

    def forward(self, x: tensor.Tensor):
        xp = x.data
        try:
            xnp = xp.numpy()
            sN, sC, sH, sW = xnp.strides
            shape = (self.N, self.in_channels, self.out_h, self.out_w, self.kernel_size, self.kernel_size)
            strides = (sN, sC, sH * self.stride, sW * self.stride, sH, sW)
            windows = np.lib.stride_tricks.as_strided(xnp, shape=shape, strides=strides)
            col_np = windows.transpose(0, 1, 4, 5, 2, 3)
            col = backend_api.from_numpy(np.ascontiguousarray(col_np), device=x.device)
            return col
        except Exception:
            col = backend_api.zeros((self.N, self.in_channels, self.kernel_size,
                                 self.kernel_size, self.out_h, self.out_w), device=x.device)
            for i in range(self.kernel_size):
                i_max = i + self.out_h * self.stride
                for j in range(self.kernel_size):
                    j_max = j + self.out_w * self.stride
                    col[:, :, i, j, :, :] = xp[:, :, i:i_max:self.stride,
                                                j:j_max:self.stride]
            return col

    def grad_fn(self, x: tensor.Tensor, grad):
        grad_col = grad
        grad_x = backend_api.zeros((self.N, self.in_channels, self.n_h, self.n_w), device=x.device)
        for i in range(self.kernel_size):
            i_max = i + self.out_h * self.stride
            for j in range(self.kernel_size):
                j_max = j + self.out_w * self.stride
                grad_x[:, :, i:i_max:self.stride,
                       j:j_max:self.stride] = grad_col[:, :, i, j, :, :]
        return grad_x


class __pad2d(tensor.UnaryOperator):
    def __init__(self, x: Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: Tensor):
        return backend_api.pad(x.data, ((0, 0), (0, 0),
                                        (self.pad_width, self.pad_width),
                                        (self.pad_width, self.pad_width)))

    def grad_fn(self, x: Tensor, grad):
        if self.pad_width == 0:
            return grad
        # 修正：grad是4维（N,C,H_pad,W_pad），补充通道维索引（: 表示保留所有通道）
        return grad[:, :,  # 第1维（N）全要，第2维（C）全要
                    self.pad_width:-self.pad_width,  # 第3维（H）去掉padding
                    self.pad_width:-self.pad_width]  # 第4维（W）去掉padding


def conv2d(x: Tensor,
           kernel: Tensor,
           padding: int = 0,
           stride: int = 1):
    """
    二维卷积函数

    基于im2col实现的二维卷积. 为了实现上的方便，我们不考虑长宽不同的卷积核，步长和补零。

    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_height, n_width);
    kernel : Tensor
        卷积核，形状为(out_channels, in_channels, kernel_height, kernel_width);
    padding : int, default=0
        对输入图片周围补0数量;
    stride : int, default=1
        卷积步长.
    """
    N, _, _, _ = x.shape
    out_channels, _, kernel_size, _ = kernel.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    col_filter = kernel.reshape(out_channels, -1).T
    out = col @ col_filter
    return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)


def max_pool2d(x: tensor.Tensor, kernel_size: int, stride: int, padding=0):
    """
    二维卷积函数池化

    基于im2col实现的二维卷积. 为了实现上的方便，我们不考虑长宽不同的kernel_size，步长和补零。

    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_height, n_width);
    kernel_size : int
        池化核尺寸;
    stride : int, default=1
        卷积步长;
    padding : int, default=0
        对输入图片周围补0数量;
    """
    N, in_channels, _, _ = x.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = col.max(1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


def avg_pool2d(x: tensor.Tensor, kernel_size: int, stride: int, padding=0):
    """
    二维平均池化

    基于im2col实现的二维池化. 为了实现上的方便，我们不考虑长宽不同的kernel_size，步长和补零。

    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_height, n_width);
    kernel_size : int
        池化核尺寸;
    stride : int, default=1
        卷积步长;
    padding : int, default=0
        对输入图片周围补0数量;
    """
    N, in_channels, _, _ = x.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = col.mean(1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


