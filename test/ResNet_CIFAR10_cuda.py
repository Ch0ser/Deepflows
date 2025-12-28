import os, sys
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import builtins

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DeepFlows import tensor
from DeepFlows.tensor import Tensor
from DeepFlows.autograd import no_grad
from DeepFlows import nn
from DeepFlows.optim import Adam
from DeepFlows.optim.scheduler import WarmupCosineLR
from DeepFlows.utils import data_loader
from DeepFlows import backend_api

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, device="cuda"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        self.downsample = downsample
        self.stride = stride
        self.device = device

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            for layer in self.downsample:
                identity = layer(identity)
        out = out + identity
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, img_size=(32, 32), device="cuda"):
        super().__init__()
        self.device = device
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(32, device=device)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.fc = nn.Linear(256, num_classes, device=device)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            conv = nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False, device=self.device)
            bn = nn.BatchNorm2d(out_channels, device=self.device)
            downsample = [conv, bn]
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, device=self.device))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, device=self.device))
        return layers

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        for block in self.layer1:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        for block in self.layer2:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        for block in self.layer3:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        for block in self.layer4:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        x = tensor.mean(x, axis=2)
        x = tensor.mean(x, axis=2)
        x = self.fc(x)
        return x

def ResNet18(num_classes=10, img_size=(32, 32), device="cuda"):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes, img_size, device)

def load_cifar10_data():
    import pickle
    base = r"e:\P.A.R.A\Project\ComprehensiveDesign\codes\Deepflows\data\cifar-10-batches-py"
    def load_batch(fname):
        with open(os.path.join(base, fname), 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        x = d['data']
        y = np.array(d['labels'], dtype=np.int32)
        x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        return x, y
    xs = []
    ys = []
    for i in range(1, 6):
        x, y = load_batch(f'data_batch_{i}')
        xs.append(x)
        ys.append(y)
    x_train = np.ascontiguousarray(np.concatenate(xs, axis=0))
    y_train = np.ascontiguousarray(np.concatenate(ys, axis=0))
    x_test, y_test = load_batch('test_batch')
    x_test = np.ascontiguousarray(x_test)
    y_test = np.ascontiguousarray(y_test)
    m = x_train.mean(axis=(0, 2, 3), keepdims=True)
    s = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x_train = (x_train - m) / s
    x_test = (x_test - m) / s
    return x_train, y_train, x_test, y_test

def augment_batch(inputs, epoch, num_epochs):
    bs, c, h, w = inputs.shape
    pad = 4
    padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    ys = np.random.randint(0, 2 * pad + 1, size=bs)
    xs = np.random.randint(0, 2 * pad + 1, size=bs)
    out = np.empty_like(inputs)
    for i in range(bs):
        out[i] = padded[i, :, ys[i]:ys[i] + h, xs[i]:xs[i] + w]
    flip_mask = np.random.rand(bs) < 0.5
    out[flip_mask] = out[flip_mask][:, :, :, ::-1]
    if epoch < num_epochs - 5 and np.random.rand() < 0.2:
        erase_h = builtins.max(1, int(h * np.random.uniform(0.1, 0.2)))
        erase_w = builtins.max(1, int(w * np.random.uniform(0.1, 0.2)))
        ys_e = np.random.randint(0, h - erase_h + 1, size=bs)
        xs_e = np.random.randint(0, w - erase_w + 1, size=bs)
        for i in range(bs):
            out[i, :, ys_e[i]:ys_e[i]+erase_h, xs_e[i]:xs_e[i]+erase_w] = 0.0
    out = np.clip(out, -1.0, 1.0)
    return out

def train_resnet():
    batch_size = 256
    num_epochs = 20
    learning_rate = 0.001
    target_acc = 85.0
    max_train_batches = 200
    max_test_batches = 50
    x_train, y_train, x_test, y_test = load_cifar10_data()
    num_classes = 10
    encoder = OneHotEncoder(sparse_output=False)
    all_classes = np.arange(num_classes).reshape(-1, 1)
    encoder.fit(all_classes)
    loader = data_loader(x_train, y_train, batch_size, shuffle=True, prefetch_size=0, as_contiguous=True)
    test_loader = data_loader(x_test, y_test, batch_size, shuffle=False, prefetch_size=0, as_contiguous=True)
    model = ResNet18(num_classes, img_size=(32, 32), device='cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, T_max=num_epochs, eta_min=1e-5)
    train_losses = []
    test_accuracies = []
    train_batch_losses = []
    test_batch_accuracies = []
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(loader):
            if batch_idx >= max_train_batches:
                break
            inputs = augment_batch(inputs, epoch, num_epochs)
            eps = 0.05
            labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
            labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
            inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
            outputs = model(inputs)
            loss = criterion(outputs, labels_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.data.numpy().item()
            running_loss += loss_value
            train_batch_losses.append(loss_value)
            if batch_idx % 50 == 0 or batch_idx + 1 == min(len(loader.batch_sampler), max_train_batches):
                print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{min(len(loader.batch_sampler), max_train_batches)}] 当前Loss: {loss_value:.4f}")
            outputs.dispose()
            loss.dispose()
            inputs.dispose()
            labels_onehot.dispose()
            del inputs, labels_onehot, outputs, loss
            tensor.Graph.free_graph()
            if batch_idx % 50 == 0:
                gc.collect()
        train_loss = running_loss / len(loader.batch_sampler)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s")
        model.eval()
        correct = 0
        total = 0
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 开始测试...")
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                if batch_idx >= max_test_batches:
                    break
                labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
                inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
                outputs = model(inputs)
                total += labels_onehot.shape[0]
                _pred = np.argmax(outputs.data.numpy(), 1).reshape(-1, 1)
                _true = np.argmax(labels_onehot.data.numpy(), 1).reshape(-1, 1)
                correct += np.sum(_pred == _true)
                if batch_idx % 10 == 0:
                    current_acc = 100 * correct / total
                    test_batch_accuracies.append(current_acc)
                    print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{min(len(test_loader.batch_sampler), max_test_batches)}] 当前准确率: {current_acc:.2f}%")
                outputs.dispose()
                inputs.dispose()
                labels_onehot.dispose()
                del inputs, labels_onehot, outputs
                tensor.Graph.free_graph()
                if batch_idx % 20 == 0:
                    gc.collect()
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        scheduler.step()
        if accuracy >= target_acc:
            print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
            break
        gc.collect()
    gc.collect()
    total_time = time.time() - total_start_time
    print(f"Total Training Time: {total_time:.2f}s")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_batch_losses) + 1), train_batch_losses)
    plt.title('训练损失')
    plt.xlabel('批次')
    plt.ylabel('损失')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_batch_accuracies) + 1), test_batch_accuracies, color='orange')
    plt.title('测试准确率')
    plt.xlabel('批次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('resnet_cifar10_training.png')
    print("训练完成！图表已保存为 'resnet_cifar10_training.png'")
    print(f"最终测试准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    train_resnet()
