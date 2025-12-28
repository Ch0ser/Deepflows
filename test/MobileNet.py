import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from DeepFlows.tensor import *
from DeepFlows.optim.sgd import SGD
from DeepFlows.optim.scheduler import StepLR
from DeepFlows.utils import data_loader
from DeepFlows import nn, tensor, backend_api
from DeepFlows.nn import Module, Conv2d, BatchNorm2d, Linear
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import gc
import time
# 移除不存在的导入

class ConvBlock(Module):
    """简化的卷积块，替代深度可分离卷积"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, device="cuda"):
        super().__init__()
        self.conv = Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                          stride=stride, padding=padding, 
                          bias=False, device=device)
        self.bn = BatchNorm2d(out_ch, device=device)

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        x = self.conv(x)
        x = self.bn(x)
        # 使用maximum实现ReLU
        x = tensor.maximum(x, 0)
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x

def make_divisible(channels, divisor=8):
    """使通道数能被 divisor 整除（硬件友好）"""
    # 直接使用条件判断代替max函数
    calculated = int(channels + divisor / 2) // divisor * divisor
    new_channels = divisor if calculated < divisor else calculated
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


class MobileNetV1(Module):
    def __init__(self, num_classes=10, width_multiplier=1.0, device="cuda", img_size=(32, 32)):
        super().__init__()
        self.device = device
        self.img_size = img_size

        def c(ch):  # 通道缩放 + 对齐
            return make_divisible(ch * width_multiplier)

        # 第一层：标准卷积 - 对于32x32图像，使用stride=1避免过度压缩
        self.conv1 = Conv2d(3, c(32), kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = BatchNorm2d(c(32), device=device)
        # 使用tensor操作替代ReLU6

        # 简化的卷积块
        self.block1 = ConvBlock(c(32), c(64), stride=1, device=device)
        self.block2 = ConvBlock(c(64), c(128), stride=2, device=device)  # 32->16
        self.block3 = ConvBlock(c(128), c(128), stride=1, device=device)
        self.block4 = ConvBlock(c(128), c(256), stride=2, device=device)  # 16->8
        self.block5 = ConvBlock(c(256), c(256), stride=1, device=device)
        self.block6 = ConvBlock(c(256), c(512), stride=1, device=device)
        self.block7 = ConvBlock(c(512), c(512), stride=1, device=device)  # 修复通道不匹配问题
        self.block8 = ConvBlock(c(512), c(512), stride=1, device=device)

        # 全局平均池化将在forward中使用tensor.mean实现
        # 记录最后一层的通道数，便于正确设置全连接层
        self.last_channels = c(512)
        self.fc = Linear(self.last_channels, num_classes, device=device)

    def forward(self, x):
        from DeepFlows.tensor import Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
        # 使用tensor操作替代ReLU6
        from DeepFlows import maximum
        x = maximum(self.bn1(self.conv1(x)), 0)
        if not isinstance(x, Tensor):
            x = Tensor(x)
        # 单独调用每个卷积块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        # 全局平均池化 - 分别对每个维度进行平均，因为只支持单轴操作
        from DeepFlows import tensor
        x = tensor.mean(x, axis=2)  # 对H维度进行平均
        x = tensor.mean(x, axis=2)  # 对W维度进行平均
        # 分类层
        x = self.fc(x)
        return x

def load_animal_data(root_dir, img_size=(32, 32)):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    class_names = sorted(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    X, Y = [], []
    for cname in class_names:
        cdir = os.path.join(root_dir, cname)
        for fname in os.listdir(cdir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in exts:
                path = os.path.join(cdir, fname)
                img = Image.open(path).convert('RGB').resize(img_size, Image.BILINEAR)
                arr = np.asarray(img, dtype=np.uint8)
                arr = arr.transpose(2, 0, 1)
                X.append(arr)
                Y.append(class_to_idx[cname])
    x = np.stack(X).astype(np.float32) / 255.0
    y = np.asarray(Y, dtype=np.int32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1/7, random_state=42, stratify=y
    )
    mean_c = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std_c = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x_train = (x_train - mean_c) / std_c
    x_test = (x_test - mean_c) / std_c
    x_train = np.ascontiguousarray(x_train)
    x_test = np.ascontiguousarray(x_test)
    return x_train, y_train, x_test, y_test, class_names, mean_c.squeeze(), std_c.squeeze()

def augment_batch(inputs):
    bs = inputs.shape[0]
    out = inputs.copy()
    flip_mask = np.random.rand(bs) < 0.5
    out[flip_mask] = out[flip_mask][:, :, :, ::-1]
    return out

def train_mobilenet():
    # 加载数据
    root_dir = r"./data/Animal"
    img_size = (32, 32)
    x_train, y_train, x_test, y_test, class_names, mean_c, std_c = load_animal_data(root_dir, img_size=img_size)
    num_classes = len(class_names)
    
    # 准备数据加载器
    encoder = OneHotEncoder(sparse_output=False)
    all_classes = np.arange(num_classes).reshape(-1, 1)
    encoder.fit(all_classes)
    
    batch_size = 16
    loader = data_loader(x_train, y_train, batch_size, shuffle=True, prefetch_size=0, as_contiguous=True)
    test_loader = data_loader(x_test, y_test, batch_size, shuffle=False, prefetch_size=0, as_contiguous=True)
    
    # 创建MobileNetV1模型
    # 使用较小的width_multiplier以适应小规模数据集并减少计算量
    model = MobileNetV1(num_classes=num_classes, width_multiplier=0.25, device='cuda', img_size=img_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    num_epochs = 15
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
            inputs = augment_batch(inputs)
            # label smoothing
            eps = 0.1
            labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
            labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
            inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
            outputs = model(inputs)
            loss = criterion(outputs, labels_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.numpy().item()
            train_batch_losses.append(loss.data.numpy().item())
            if batch_idx % 2 == 0 or batch_idx + 1 == len(loader.batch_sampler):
                print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(loader.batch_sampler)}] 当前Loss: {loss.data.numpy().item():.4f}")
            # 清理显存
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
                inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
                outputs = model(inputs)
                total += labels_onehot.shape[0]
                _pred = np.argmax(outputs.data.numpy(), 1).reshape(-1, 1)
                _true = np.argmax(labels_onehot.data.numpy(), 1).reshape(-1, 1)
                correct += np.sum(_pred == _true)
                if batch_idx % 1 == 0:
                    current_acc = 100 * correct / total
                    test_batch_accuracies.append(current_acc)
                    print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{len(test_loader.batch_sampler)}] 当前准确率: {current_acc:.2f}%")
                # 清理显存
                outputs.dispose()
                inputs.dispose()
                labels_onehot.dispose()
                del inputs, labels_onehot, outputs
                if batch_idx % 20 == 0:
                    gc.collect()
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        scheduler.step()
        if accuracy >= target_acc:
            print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
            break
    
        Graph.free_graph_all()
        gc.collect()
    
    Graph.free_graph_all()
    gc.collect()
    
    # 绘制训练过程图表
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
    plt.savefig('mobilenet_animal_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练完成！图表已保存为 'mobilenet_animal_training.png'")
    print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")

def test_output_shape():
    # 创建模型（小规模，便于调试）
    print("Creating MobileNetV1 model...")
    model = MobileNetV1(num_classes=10, width_multiplier=0.25, device='cuda')  # 先用 CPU 避免 CUDA 调试复杂
    print(f"Model created successfully. Last channels: {model.last_channels}")

    # 构造输入以测试
    print("Creating input tensor...")
    x = tensor.randn(1, 3, 32, 32, device=backend_api.Device('cuda'))
    print("Input shape:", x.shape)

    # 完整前向传播测试，使用修改后的类实现
    try:
        print("Running complete forward pass...")
        output = model(x)
        print("Output shape:", output.shape)
        
        # 验证输出形状是否正确
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
        print("✅ Output shape test passed!")
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

# 运行测试
if __name__ == "__main__":
    # 先运行简单的形状测试，确保模型能正常工作
    test_output_shape()
    # 然后运行完整的训练
    print("\n开始MobileNet模型训练...")
    train_mobilenet()