
# LinearRegression.py
# 简单线性回归示例
# - 生成一维线性数据并构造回归任务
# - 使用单层线性模型拟合 y = 3x + 5 + 噪声
# - 采用均方误差损失与 Adam 优化器训练
# - 每 100 次迭代打印损失，训练完成后可视化拟合结果
# - 设备：CPU-only，模型与张量均在 CPU 上运行


import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import Tensor
import DeepFlows.nn as nn
import DeepFlows.optim as optim
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # 设置随机种子以便复现
X_np = np.random.rand(100, 1) * 1  # 随机生成 100 个输入数据 (范围: 0 ~ 10)
y_np = 3 * X_np + 5 + np.random.randn(100, 1)
X = Tensor(X_np)
y = Tensor(y_np)


# 2. 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1, device="cpu")  # 输入和输出维度都是 1

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# 3. 选择损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)  # 随机梯度下降

# 4. 训练模型
num_epochs = 100000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_numpy = loss.numpy()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_numpy.item():.4f}')

# 5. 可视化结果
predicted = model(X).numpy()
plt.scatter(X.numpy(), y.numpy(), label='Original Data', color='blue')
plt.plot(X.numpy(), predicted, label='Fitted Line', color='red')
plt.legend()
plt.show()
