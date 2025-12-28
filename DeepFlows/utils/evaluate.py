import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def accuracy(y_hat, y):
    # 进行softmax运算转换为0-1的概率值
    y_hat = softmax(y_hat, 1)

    # 取每行概率最大的值的索引
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 同上
    y = y.argmax(axis=1)
    # 计数预测正确的个数
    cmp = y_hat == y
    return float(np.array(cmp, dtype=y.dtype).sum())


def evaluate_accuracy(model, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(model.forward(X), y), y.numel)
    return metric[0] / metric[1]


class Accumulator:

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
