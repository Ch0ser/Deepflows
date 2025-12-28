import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import *
from DeepFlows.optim import Adam
from DeepFlows.utils import data_loader
from DeepFlows.utils.model_utils import save_checkpoint, load_checkpoint
from DeepFlows import nn
from DeepFlows.optim.scheduler import CosineAnnealingLR
from DeepFlows.tensor import Tensor
from DeepFlows import backend_api
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import gc
import time

def load_cifar10_data():
    import pickle, os
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cifar-10-batches-py')
    print(f"使用数据路径: {base}")
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
    s = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7
    x_train = (x_train - m) / s
    x_test = (x_test - m) / s
    return x_train, y_train, x_test, y_test

# 加载CIFAR-10数据（完全使用sklearn，无其他第三方库）
x_train, y_train, x_test, y_test = load_cifar10_data()
print(f"CIFAR-10数据加载完成：")
print(f"训练集: {x_train.shape} (样本数, 通道数, 高度, 宽度)")
print(f"测试集: {x_test.shape}")

# 标签独热编码（保持原有逻辑）
encoder = OneHotEncoder(sparse_output=False)
all_classes = np.arange(10).reshape(-1, 1)
encoder.fit(all_classes)
y_train_onehot = encoder.transform(y_train.reshape(-1, 1)).astype(np.float32)
y_test_onehot = encoder.transform(y_test.reshape(-1, 1)).astype(np.float32)

# 数据加载器（保持原有参数和逻辑）
batch_size = 64
loader = data_loader(x_train, y_train_onehot, batch_size, shuffle=True, prefetch_size=1)
test_loader = data_loader(x_test, y_test_onehot, batch_size, shuffle=False, prefetch_size=1)

class CIFAR10_CNN(nn.Module):
    """适配CIFAR-10的CNN模型（3通道32x32输入），保持原有模型结构风格"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, device='cuda')
        self.bn1 = nn.BatchNorm2d(32, device='cuda')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, device='cuda')
        self.bn2 = nn.BatchNorm2d(64, device='cuda')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, device='cuda')
        self.bn3 = nn.BatchNorm2d(128, device='cuda')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4 * 4, num_classes, device='cuda')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.drop(x)
        x = self.fc(x)
        return x

# 模型、损失函数、优化器（保持原有配置）
model = CIFAR10_CNN(10)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

# 模型保存路径 
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'cifar10_cnn_cuda_checkpoint.pkl')
start_epoch = 0

# 确保检查点目录存在
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"确保检查点目录 {checkpoint_dir} 存在")

# 尝试加载模型检查点
train_losses = []
test_accuracies = []
try:
    if os.path.exists(checkpoint_path):
        print(f"发现检查点 {checkpoint_path}，尝试加载...")
        checkpoint_dict = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch = checkpoint_dict.get('epoch', 0)
        for _ in range(start_epoch):
            scheduler.step()
        
        # 尝试加载训练历史数据
        if os.path.exists(checkpoint_path + '.info'):
            try:
                import dill as pickle
                with open(checkpoint_path + '.info', 'rb') as f:
                    save_info = pickle.load(f)
                    train_losses = save_info.get('train_losses', [])
                    test_accuracies = save_info.get('test_accuracies', [])
                print("成功加载训练历史数据")
            except Exception as e:
                print(f"加载训练历史数据失败: {e}，使用空列表继续")
                train_losses = []
                test_accuracies = []
        print(f"成功加载检查点！从第 {start_epoch+1} 个epoch继续训练")
    else:
        print("未发现检查点，从头开始训练")
except Exception as e:
    print(f"加载检查点失败: {e}，从头开始训练")
    start_epoch = 0
    train_losses = []
    test_accuracies = []

# 训练参数
num_epochs = 18
target_acc = 85.0  # CIFAR-10难度高于MNIST，调整目标准确率
train_batch_losses = []  # 不重复初始化train_losses和test_accuracies，避免覆盖从检查点加载的数据
print("开始正常训练，设置{}个训练轮次".format(num_epochs))

t0 = time.time()

# 训练循环（完全保持原有逻辑）
for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels_onehot) in enumerate(loader):
        aug_inputs = inputs
        if np.random.rand() < 0.5:
            aug_inputs = aug_inputs[:, :, :, ::-1]
        inputs, labels_onehot = Tensor(aug_inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
        outputs = model(inputs)
        loss = criterion(outputs, labels_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data.numpy().item()
        train_batch_losses.append(loss.data.numpy().item())
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(loader.batch_sampler)}] 当前Loss: {loss.data.numpy().item():.4f}")
        
        # 正常训练所有批次，无需提前结束训练循环
        outputs.dispose()
        loss.dispose()
        inputs.dispose()
        labels_onehot.dispose()
        del inputs, labels_onehot, outputs, loss
        Graph.free_graph()
        if batch_idx % 50 == 0:
            gc.collect()
    train_loss = running_loss / len(loader.batch_sampler)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s")
    scheduler.step()
    
    # 测试循环（完全保持原有逻辑）
    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for batch_idx, (inputs, labels_onehot) in enumerate(test_loader):
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
            Graph.free_graph()
            if batch_idx % 20 == 0:
                gc.collect()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
    
    # 保存检查点
    if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs or accuracy >= target_acc:
        print(f"保存模型检查点到 {checkpoint_path}")
        backup_path = checkpoint_path + '.backup'
        
        # 确保检查点目录存在
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            # 保存基本信息到文件
            save_info = {
                'epoch': epoch,
                'train_losses': train_losses,
                'test_accuracies': test_accuracies
            }
            
            # 先备份旧文件
            if os.path.exists(checkpoint_path + '.info'):
                os.replace(checkpoint_path + '.info', checkpoint_path + '.info.backup')
            if os.path.exists(checkpoint_path):
                os.replace(checkpoint_path, backup_path)
            
            # 保存信息文件
            with open(checkpoint_path + '.info', 'wb') as f:
                import dill as pickle
                pickle.dump(save_info, f)
            print(f"训练历史数据已保存到 {checkpoint_path + '.info'}")
            
            # 保存模型和优化器
            save_checkpoint(model, optimizer, epoch, train_losses[-1] if train_losses else None, checkpoint_path)
            print(f"模型检查点已保存到 {checkpoint_path}")
            
            # 保存成功后可以删除备份
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except Exception as e:
            print(f"保存检查点时出错: {e}")
            # 尝试恢复备份
            if os.path.exists(backup_path):
                os.replace(backup_path, checkpoint_path)
                print(f"已从备份恢复检查点")
            if os.path.exists(checkpoint_path + '.info.backup'):
                os.replace(checkpoint_path + '.info.backup', checkpoint_path + '.info')
                print(f"已从备份恢复训练历史数据")
    else:
        print(f"当前为第 {epoch+1} 轮，跳过保存")


    Graph.free_graph_all()
    gc.collect()
    
    # 提前终止条件
    if accuracy >= target_acc:
        print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
        break

Graph.free_graph_all()
gc.collect()

# 绘图逻辑
plt.figure(figsize=(12, 5))
print(f"Total Training Time: {time.time()-t0:.2f}s")
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_batch_losses) + 1), train_batch_losses, marker='o')
plt.title('CIFAR-10 Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss (per batch)')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', color='orange')  # 修正：原代码用了test_batch_accuracies（未定义）
plt.title('CIFAR-10 Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.tight_layout()
_pics_dir = os.path.join(os.path.dirname(__file__), 'pics')
os.makedirs(_pics_dir, exist_ok=True)
_fig_path = os.path.join(_pics_dir, 'cifar10_cnn_training_memory_optimized.png')
plt.savefig(_fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"训练完成！图表已保存为 '{_fig_path}'")
print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
