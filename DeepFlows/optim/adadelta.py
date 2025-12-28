from .optimier import Optimizer
import numpy as np


class Adadelta(Optimizer):
    def __init__(
            self,
            params,
            lr=float(1.0),
            rho=float(0.9),
            weight_decay=float(0),
            eps=float(1e-6),
    ):
        super().__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.s = [np.zeros(param.shape) for param in self.params]
        self.delta = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data

            self.s[i] = self.rho * self.s[i] + (1 - self.rho) * grad ** 2
            adjust_grad = ((self.delta[i] + self.eps) ** 0.5) * grad / (self.s[i] + self.eps) ** 0.5
            self.delta[i] = self.rho * self.delta[i] + (1 - self.rho) * adjust_grad ** 2
            self.params[i].data -= adjust_grad