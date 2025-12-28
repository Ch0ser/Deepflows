from math import sqrt
from ..tensor import Tensor
from typing import List, Tuple


class Optimizer:
    def __init__(self, params: List[Tensor]) -> None:
        self.params: List[Tensor] = list(params)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
