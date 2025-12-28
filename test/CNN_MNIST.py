import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import *
from DeepFlows.optim import Adam
from DeepFlows.utils import data_loader
from DeepFlows import nn
from DeepFlows.tensor import Tensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import gc
import time


# --------------------------
# 1. 加载MNIST数据集
# --------------------------
def load_mnist_data():
    import struct, os
    base = r"e:\P.A.R.A\Project\ComprehensiveDesign\codes\Deepflows\data\MNIST\raw"
    def read_idx(path):
        with open(path, 'rb') as f:
            data = f.read()
        dims = data[3]
        offset = 4
        shape = []
        for _ in range(dims):
            shape.append(struct.unpack('>I', data[offset:offset+4])[0])
            offset += 4
        arr = np.frombuffer(data, dtype=np.uint8, offset=offset)
        return arr.reshape(shape)
    train_images = read_idx(os.path.join(base, 'train-images-idx3-ubyte'))
    train_labels = read_idx(os.path.join(base, 'train-labels-idx1-ubyte'))
    test_images = read_idx(os.path.join(base, 't10k-images-idx3-ubyte'))
    test_labels = read_idx(os.path.join(base, 't10k-labels-idx1-ubyte'))
    x_train = train_images.astype(np.float32) / 255.0
    x_test = test_images.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    y_train = train_labels.astype(np.int32)
    y_test = test_labels.astype(np.int32)
    x_train = np.ascontiguousarray(x_train)
    x_test = np.ascontiguousarray(x_test)
    return x_train, y_train, x_test, y_test


# 加载MNIST数据集
x_train, y_train, x_test, y_test = load_mnist_data()

# 内存优化：创建全局编码器，避免每批次重新创建
print("创建全局独热编码器...")
encoder = OneHotEncoder(sparse_output=False)
# 预训练编码器使用所有可能的类别（0-9）
all_classes = np.arange(10).reshape(-1, 1)
encoder.fit(all_classes)

# 优化批次大小
batch_size = 64

# 数据加载器 - 使用原始标签，延迟独热编码
loader = data_loader(x_train, y_train, batch_size, shuffle=True)
test_loader = data_loader(x_test, y_test, batch_size, shuffle=False)


class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, device='cpu')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, device='cpu')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes, device='cpu')

    def forward(self, x):
        # 第1卷积块：conv→relu→pool
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第2卷积块：conv→relu→pool
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


model = MNIST_CNN(10)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10
target_acc = 95.0
train_losses = []
test_accuracies = []
test_batch_accuracies = []

t0 = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(loader):
        labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
        inputs, labels_onehot = Tensor(inputs), Tensor(labels_onehot)

        outputs = model(inputs)
        loss = criterion(outputs, labels_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data.numpy().item()  # 修复DeprecationWarning
        if batch_idx % 100 == 0:  # 每100个批次打印一次
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(loader.batch_sampler)}] 当前Loss: {loss.data.numpy().item():.4f}")

        # 内存清理
        outputs.dispose()
        loss.dispose()
        inputs.dispose()
        labels_onehot.dispose()
        del inputs, labels_onehot, outputs, loss
        if batch_idx % 50 == 0:
            gc.collect()

    train_loss = running_loss / len(loader.batch_sampler)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s")

    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
            inputs, labels_onehot = Tensor(inputs), Tensor(labels_onehot)
            outputs = model(inputs)
            total += labels_onehot.shape[0]
            _pred = np.argmax(outputs.data.numpy(), 1).reshape(-1, 1)
            _true = np.argmax(labels_onehot.data.numpy(), 1).reshape(-1, 1)
            correct += np.sum(_pred == _true)
            batch_acc = 100.0 * np.mean(_pred == _true)
            test_batch_accuracies.append(batch_acc)
            if batch_idx % 20 == 0:
                current_acc = 100 * correct / total
                print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{len(test_loader.batch_sampler)}] 当前准确率: {current_acc:.2f}%")
            outputs.dispose()
            inputs.dispose()
            labels_onehot.dispose()
            del inputs, labels_onehot, outputs
            if batch_idx % 20 == 0:
                gc.collect()
    
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
    if accuracy >= target_acc:
        print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
        break
    
    Graph.free_graph_all()
    gc.collect()

# 最终内存清理
Graph.free_graph_all()
gc.collect()

print(f"Total Training Time: {time.time()-t0:.2f}s")

# 绘制训练曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', color='orange')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('mnist_cnn_training_memory_optimized.png', dpi=150, bbox_inches='tight')
plt.show()

print("训练完成！图表已保存为 'mnist_cnn_training_memory_optimized.png'")
print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
