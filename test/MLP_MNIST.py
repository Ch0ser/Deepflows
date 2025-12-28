# MLP_MNIST.py
# 多层感知机（MLP）在 MNIST 数据集上的分类任务
# 功能概述
# - 加载 MNIST 手写数字数据集、预处理并拆分训练/测试集
# - 定义三层全连接 MLP 模型并在数据上训练
# - 评估分类准确率并绘制训练损失与测试准确率曲线

# 数据处理
# - 数据来源与规模：从 OpenML 获取 MNIST，取前 5000 条样本并归一化到 `[0,1]`（`test/MLP_MMNIST.py:22-24`）
# - 标签处理：独热编码（`OneHotEncoder(sparse_output=False)`，`test/MLP_MMNIST.py:24`）
# - 集合拆分：训练集 70% / 测试集 30%（`test/MLP_MMNIST.py:25-29`）
# - 批量与加载器：`batch_size=64`，构建训练/测试加载器（`test/MLP_MMNIST.py:34-37`）

# 模型结构
# - 三层 MLP：`784→100→20→10`，中间层使用 ReLU 激活（`test/MLP_MMNIST.py:41-51`）
# - 损失与优化：交叉熵损失 + SGD（`lr=0.05`）（`test/MLP_MMNIST.py:55-56`）

# 训练流程
# - 训练 50 个 epoch，逐批前向/反向/更新参数（`test/MLP_MMNIST.py:63-73`）
# - 设备：CPU-only，无需显式指定 device 参数
# - 损失汇总与日志：累计批次损失并打印每个 epoch 的平均损失（`test/MLP_MMNIST.py:65, 74-79`）

# 评估与可视化
# - 评估：测试集上逐批计算预测，统计总体准确率并打印（`test/MLP_MMNIST.py:80-91`）
# - 可视化：绘制“训练损失曲线”和“测试准确率曲线”（`test/MLP_MMNIST.py:95-110`）

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import *
from DeepFlows import optim
from DeepFlows.utils import data_loader
from DeepFlows.nn import functional as F
from DeepFlows import nn
from DeepFlows.tensor import Tensor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set()

# 数据预处理：独热化+标准化
data_x, data_y = fetch_openml('mnist_784', version=1, return_X_y=True)
data_x, data_y = data_x[:5000].values / 255, data_y[:5000].values
data_y = OneHotEncoder(sparse_output=False).fit_transform(data_y.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(
    data_x,
    data_y,
    train_size=0.7,
)
# 显式使用 float32 并保持连续，避免内部转换与溢出警告
x_train = np.ascontiguousarray(x_train.astype(np.float32))
x_test = np.ascontiguousarray(x_test.astype(np.float32))
y_train = np.ascontiguousarray(y_train.astype(np.float32))
y_test = np.ascontiguousarray(y_test.astype(np.float32))
#stder = StandardScaler()
#x_train = stder.fit_transform(x_train).astype(np.float32)
#x_test = stder.transform(x_test).astype(np.float32)

batch_size = 256

loader = data_loader(x_train, y_train, batch_size, True)
test_loader = data_loader(x_test, y_test, batch_size, False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 100, device='cpu')
        self.fc2 = nn.Linear(100, 20, device='cpu')
        self.fc3 = nn.Linear(20, 10, device='cpu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)


num_epochs = 50
train_losses = []
test_accuracies = []
t0 = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = Tensor(inputs), Tensor(labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data.numpy().item()

    train_loss = running_loss / len(loader.batch_sampler)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = Tensor(inputs), Tensor(labels)
        outputs = model(inputs)
        total += labels.shape[0]
        correct += np.sum(np.argmax(outputs.data.numpy(), 1).reshape(-1, 1) ==
                          np.argmax(labels.data.numpy(), 1).reshape(-1, 1))
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")


# 绘制训练损失和测试准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.show()
print(f"Total Training Time: {time.time()-t0:.2f}s")