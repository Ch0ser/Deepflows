import os, sys
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入DeepFlows框架组件
from DeepFlows import tensor
from DeepFlows.tensor import Tensor
from DeepFlows.autograd import no_grad
from DeepFlows import nn
from DeepFlows.optim.sgd import SGD
from DeepFlows.optim.scheduler import StepLR
from DeepFlows.utils import data_loader
from DeepFlows import backend_api

# ResNet基本组件：残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, device="cuda"):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        # 下采样模块（用于残差连接时的维度匹配）
        self.downsample = downsample
        self.stride = stride
        self.device = device
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        # 简化实现，暂时跳过激活函数以测试基本功能

        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对输入进行处理以匹配输出维度
        if self.downsample is not None:
            # 手动执行下采样步骤
            for layer in self.downsample:
                identity = layer(identity)

        # 残差连接 - 直接使用加法运算符，确保结果是tensor对象
        # 确保out和identity都是tensor类型
        out = out + identity
        # 简化实现，暂时跳过最终激活函数
        out = self.relu(out)
        return out

# ResNet主模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, img_size=(32, 32), device="cuda"):
        super().__init__()
        self.device = device
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(64, device=device)
        self.relu = nn.ReLU()
        
        # 构建残差层列表
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全连接层：输入为全局平均池化后的通道数(512)
        self.fc = nn.Linear(512, num_classes, device=device)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        # 创建下采样模块（如果需要）
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # 手动创建下采样模块
            conv = nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False, device=self.device)
            bn = nn.BatchNorm2d(out_channels, device=self.device)
            downsample = [conv, bn]  # 使用列表存储下采样组件

        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.in_channels, out_channels, stride, downsample, device=self.device))
        self.in_channels = out_channels
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, device=self.device))

        # 返回层列表，后续在forward中手动调用
        return layers

    def forward(self, x):
        # 确保输入是Tensor对象
        from DeepFlows.tensor import Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 手动依次调用每个残差块
        for block in self.layer1:
            x = block(x)
            # 确保每个残差块后的输出都是Tensor对象
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
        
        # 简化全局平均池化：由于backend只支持对单个轴进行归约，我们需要分别对每个轴求平均
        # 先获取输入形状
        batch_size, channels, height, width = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        
        # 使用更安全的方式计算全局平均池化，对每个轴分别求平均
        from DeepFlows import tensor
        # 先对height轴（轴2）求平均
        x = tensor.mean(x, axis=2)
        # 再对width轴（现在变为轴2）求平均
        x = tensor.mean(x, axis=2)
        
        # 现在x的形状是[batch, 512]，所以我们需要修改全连接层的输入维度
        
        # 全连接层分类
        x = self.fc(x)

        return x

# 创建ResNet-18模型
def ResNet18(num_classes=10, img_size=(32, 32), device="cuda"):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes, img_size, device)

# 数据加载和预处理（与CNN_Animal10_cuda.py相同）
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
                arr = arr.transpose(2, 0, 1)  # 转换为(C, H, W)
                X.append(arr)
                Y.append(class_to_idx[cname])
    x = np.stack(X).astype(np.float32) / 255.0  # 归一化到[0, 1]
    y = np.asarray(Y, dtype=np.int32)
    
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1/7, random_state=42, stratify=y
    )
    
    # 计算均值和标准差用于标准化
    mean_c = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std_c = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6  # 防止除零
    
    # 标准化数据
    x_train = (x_train - mean_c) / std_c
    x_test = (x_test - mean_c) / std_c
    
    # 确保数据是连续的，提高性能
    x_train = np.ascontiguousarray(x_train)
    x_test = np.ascontiguousarray(x_test)
    
    return x_train, y_train, x_test, y_test, class_names, mean_c.squeeze(), std_c.squeeze()

# 数据增强函数
def augment_batch(inputs):
    bs = inputs.shape[0]
    out = inputs.copy()
    # 随机水平翻转
    flip_mask = np.random.rand(bs) < 0.5
    out[flip_mask] = out[flip_mask][:, :, :, ::-1]
    return out

# 主训练函数
def train_resnet():
    # 数据路径和参数设置
    root_dir = r"./data/Animal"
    img_size = (32, 32)
    batch_size = 16
    num_epochs = 15
    learning_rate = 0.01
    target_acc = 95.0
    
    # 加载数据
    print("正在加载数据...")
    x_train, y_train, x_test, y_test, class_names, mean_c, std_c = load_animal_data(root_dir, img_size=img_size)
    num_classes = len(class_names)
    print(f"数据集加载完成，类别数: {num_classes}")
    print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")
    
    # 创建OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    all_classes = np.arange(num_classes).reshape(-1, 1)
    encoder.fit(all_classes)
    
    # 创建数据加载器
    loader = data_loader(x_train, y_train, batch_size, shuffle=True, prefetch_size=0, as_contiguous=True)
    test_loader = data_loader(x_test, y_test, batch_size, shuffle=False, prefetch_size=0, as_contiguous=True)
    
    # 创建模型
    print("正在创建ResNet-18模型...")
    model = ResNet18(num_classes, img_size=img_size, device='cuda')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练记录
    train_losses = []
    test_accuracies = []
    train_batch_losses = []
    test_batch_accuracies = []
    
    total_start_time = time.time()
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 开始训练...")
        for batch_idx, (inputs, labels) in enumerate(loader):
            # 数据增强
            inputs = augment_batch(inputs)
            
            # Label smoothing
            eps = 0.1
            labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
            labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
            
            # 转换为Tensor并移至GPU
            inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), \
                                  Tensor(labels_onehot, device=backend_api.Device('cuda'))
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels_onehot)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            loss_value = loss.data.numpy().item()
            running_loss += loss_value
            train_batch_losses.append(loss_value)
            
            # 打印进度
            if batch_idx % 2 == 0 or batch_idx + 1 == len(loader.batch_sampler):
                print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(loader.batch_sampler)}] 当前Loss: {loss_value:.4f}")
            
            # 释放内存
            outputs.dispose()
            loss.dispose()
            inputs.dispose()
            labels_onehot.dispose()
            del inputs, labels_onehot, outputs, loss
            
            # 定期回收垃圾
            if batch_idx % 50 == 0:
                gc.collect()
        
        # 计算并记录平均损失
        train_loss = running_loss / len(loader.batch_sampler)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s")
        
        # 测试模型
        model.eval()
        correct = 0
        total = 0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 开始测试...")
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # 转换标签为one-hot编码
                labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
                
                # 转换为Tensor并移至GPU
                inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), \
                                      Tensor(labels_onehot, device=backend_api.Device('cuda'))
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算准确率
                total += labels_onehot.shape[0]
                _pred = np.argmax(outputs.data.numpy(), 1).reshape(-1, 1)
                _true = np.argmax(labels_onehot.data.numpy(), 1).reshape(-1, 1)
                correct += np.sum(_pred == _true)
                
                # 打印测试进度
                if batch_idx % 1 == 0:
                    current_acc = 100 * correct / total
                    test_batch_accuracies.append(current_acc)
                    print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{len(test_loader.batch_sampler)}] 当前准确率: {current_acc:.2f}%")
                
                # 释放内存
                outputs.dispose()
                inputs.dispose()
                labels_onehot.dispose()
                del inputs, labels_onehot, outputs
                
                # 定期回收垃圾
                if batch_idx % 20 == 0:
                    gc.collect()
        
        # 计算并记录测试准确率
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        
        # 学习率调度
        scheduler.step()
        
        # 提前停止条件
        if accuracy >= target_acc:
            print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
            break
        
        # 垃圾回收
        gc.collect()
    
    # 最终垃圾回收
    gc.collect()
    
    # 打印总训练时间
    total_time = time.time() - total_start_time
    print(f"Total Training Time: {total_time:.2f}s")
    
    # 可视化训练过程
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
    plt.savefig('resnet_animal_training.png')
    print("训练完成！图表已保存为 'resnet_animal_training.png'")
    print(f"最终测试准确率: {accuracy:.2f}%")

# 如果作为主程序运行，则执行训练
if __name__ == "__main__":
    train_resnet()