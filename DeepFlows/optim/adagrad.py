from .optimier import Optimizer
import numpy as np


class Adagrad(Optimizer):
    def __init__(
            self,
            params,
            lr=float(1e-2),
            weight_decay=float(0),
            eps=float(1e-10),
    ):
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.s = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data
            self.s[i] += grad ** 2
            self.params[i].data -= self.lr * grad / (self.eps + self.s[i]) ** 0.5
