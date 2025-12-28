from .module import Module
from ...tensor import Tensor
import numpy as np


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x):
        if self.training:
            # 1. 生成连续的float32掩码（CPU上），彻底避免溢出
            mask_shape = x.shape
            mask_np = np.ascontiguousarray(
                np.random.binomial(1, 1 - self.p, mask_shape).astype(np.float32)
            )

            # 2. 创建空的BackendTensor（设备上），dtype为"float32"匹配框架要求
            mask = x.device.empty(mask_shape, dtype="float32")  # 返回BackendTensor

            # 3. 关键修正：提取BackendTensor内部的底层CUDA数组（_handle属性）作为out_array
            # 原因：from_numpy需要CUDA_BACKEND.Array类型，而非封装后的BackendTensor
            mask_cuda_array = mask._handle  # 替换为你的框架中实际的底层数组属性（通常是_handle/_raw/_array）

            # 4. 调用from_numpy，参数完全匹配（numpy_array + CUDA_BACKEND.Array）
            x.device.from_numpy(mask_np, mask_cuda_array)

            # 5. 应用掩码并缩放（mask仍是BackendTensor，可正常参与运算）
            return x * mask / (1 - self.p)
        # 测试时保持原逻辑
        return x * (1 - self.p)

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)