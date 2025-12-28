import os, sys
# 强制禁用 stdout 缓冲，确保 Nsight 下能实时看到输出
sys.stdout.reconfigure(line_buffering=True)

try:
    import nvtx
except ImportError:
    try:
        import torch.cuda.nvtx as nvtx
    except ImportError:
        # 如果没有安装nvtx或torch，尝试使用cupy或自定义的空上下文管理器
        class nvtx_dummy:
            def range_push(self, msg): pass
            def range_pop(self): pass
        nvtx = nvtx_dummy()

print("Script started. Initializing...", flush=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import *
from DeepFlows.optim import Adam
from DeepFlows.utils import data_loader
from DeepFlows import nn
from DeepFlows.tensor import Tensor
from DeepFlows import backend_api
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import gc
import time

def load_mnist_data():
    import struct, os
    # base = r"e:\P.A.R.A\Project\ComprehensiveDesign\codes\Deepflows\data\MNIST\raw"
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'MNIST', 'raw')
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

print(f"Loading MNIST data...", flush=True)
x_train, y_train, x_test, y_test = load_mnist_data()
print(f"Data loaded. Train shape: {x_train.shape}", flush=True)

encoder = OneHotEncoder(sparse_output=False)
all_classes = np.arange(10).reshape(-1, 1)
encoder.fit(all_classes)

batch_size = 64
loader = data_loader(x_train, y_train, batch_size, shuffle=True)
test_loader = data_loader(x_test, y_test, batch_size, shuffle=False)

class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, device='cuda')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, device='cuda')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes, device='cuda')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

model = MNIST_CNN(10)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 1
target_acc = 95.0
train_losses = []
test_accuracies = []
train_batch_losses = []
test_batch_accuracies = []

t0 = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(loader):
        nvtx.range_push(f"Batch {batch_idx}")
        
        nvtx.range_push("Data Transfer")
        labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
        inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
        nvtx.range_pop() # Data Transfer

        nvtx.range_push("Forward")
        outputs = model(inputs)
        loss = criterion(outputs, labels_onehot)
        nvtx.range_pop() # Forward

        nvtx.range_push("Backward")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nvtx.range_pop() # Backward
        
        running_loss += loss.data.numpy().item()
        train_batch_losses.append(loss.data.numpy().item())
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(loader.batch_sampler)}] 当前Loss: {loss.data.numpy().item():.4f}")
        outputs.dispose()
        loss.dispose()
        inputs.dispose()
        labels_onehot.dispose()
        del inputs, labels_onehot, outputs, loss
        if batch_idx % 50 == 0:
            gc.collect()
            
        nvtx.range_pop() # Batch
    train_loss = running_loss / len(loader.batch_sampler)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s")

    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
            inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
            outputs = model(inputs)
            total += labels_onehot.shape[0]
            _pred = np.argmax(outputs.data.numpy(), 1).reshape(-1, 1)
            _true = np.argmax(labels_onehot.data.numpy(), 1).reshape(-1, 1)
            correct += np.sum(_pred == _true)
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

Graph.free_graph_all()
gc.collect()

plt.figure(figsize=(12, 5))
print(f"Total Training Time: {time.time()-t0:.2f}s")
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_batch_losses) + 1), train_batch_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss (per batch)')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_batch_accuracies) + 1), test_batch_accuracies, marker='o', color='orange')
plt.title('Test Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy (%) per batch')
plt.grid(True)
plt.tight_layout()
plt.savefig('mnist_cnn_training_memory_optimized.png', dpi=150, bbox_inches='tight')
plt.show()
print("训练完成！图表已保存为 'mnist_cnn_training_memory_optimized.png'")
print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
