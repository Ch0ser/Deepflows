import math


class LRScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.last_epoch = -1

    def step(self):
        self.last_epoch += 1


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        super().step()
        if self.last_epoch != 0 and self.last_epoch % self.step_size == 0:
            if hasattr(self.optimizer, 'lr'):
                self.optimizer.lr = self.optimizer.lr * self.gamma


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0) -> None:
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr if hasattr(optimizer, 'lr') else None

    def step(self):
        super().step()
        if self.base_lr is None:
            return
        t = self.last_epoch % self.T_max
        lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
        self.optimizer.lr = lr


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, T_max: int, base_lr: float = None, warmup_start_lr: float = 0.0, eta_min: float = 0.0) -> None:
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = base_lr if base_lr is not None else (optimizer.lr if hasattr(optimizer, 'lr') else None)
        self.warmup_start_lr = warmup_start_lr

    def step(self):
        super().step()
        if self.base_lr is None:
            return
        if self.last_epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            t = self.last_epoch
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (t / max(1, self.warmup_epochs))
        else:
            t = max(0, self.last_epoch - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * t / max(1, self.T_max))) / 2
        self.optimizer.lr = lr