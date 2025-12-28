from .optimier import Optimizer
from typing import List
from ..tensor import Tensor
from .. import backend_api


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.v = [backend_api.zeros(param.shape, device=param.device) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.params[i].data * self.weight_decay
            if self.momentum > 0.0:
                self.v[i] = self.v[i] * self.momentum + grad
                update = grad + self.momentum * self.v[i] if self.nesterov else self.v[i]
            else:
                update = grad
            self.params[i].data -= self.lr * update
