import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import *
from DeepFlows import optim
from DeepFlows.utils import data_loader
from DeepFlows.nn import functional as F
from DeepFlows import nn
from DeepFlows.tensor import Tensor
from DeepFlows import backend_api

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set()

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

batch_size = 256

loader = data_loader(x_train, y_train, batch_size, True)
test_loader = data_loader(x_test, y_test, batch_size, False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 100, device='cuda')
        self.fc2 = nn.Linear(100, 20, device='cuda')
        self.fc3 = nn.Linear(20, 10, device='cuda')

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
        dev = backend_api.Device('cuda')
        inputs, labels = Tensor(inputs, device=dev), Tensor(labels, device=dev)
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
        dev = backend_api.Device('cuda')
        inputs, labels = Tensor(inputs, device=dev), Tensor(labels, device=dev)
        outputs = model(inputs)
        total += labels.shape[0]
        correct += np.sum(np.argmax(outputs.data.numpy(), 1).reshape(-1, 1) ==
                          np.argmax(labels.data.numpy(), 1).reshape(-1, 1))
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")


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