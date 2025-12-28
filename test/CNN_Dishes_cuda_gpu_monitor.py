import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import *
from DeepFlows.optim import Adam
from DeepFlows.optim.scheduler import WarmupCosineLR
from DeepFlows.utils import data_loader
from DeepFlows.utils.model_utils import save_checkpoint, load_checkpoint
from DeepFlows import nn
from DeepFlows.tensor import Tensor
from DeepFlows import backend_api
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import gc
import time
import builtins
import pynvml  # Import pynvml for GPU monitoring

def load_dishes_data(root_dir, img_size=(32, 32)):
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
                try:
                    img = Image.open(path).convert('RGB').resize(img_size, Image.BILINEAR)
                    arr = np.asarray(img, dtype=np.uint8)
                    arr = arr.transpose(2, 0, 1)
                    X.append(arr)
                    Y.append(class_to_idx[cname])
                except Exception as e:
                    print(f"Skipping error file: {path}, error: {e}")
                    
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

root_dir = r"e:\P.A.R.A\Project\ComprehensiveDesign\codes\Deepflows\data\dishes"
print(f"Loading data from {root_dir}...")
x_train, y_train, x_test, y_test, class_names, mean_c, std_c = load_dishes_data(root_dir, img_size=(128, 128))
num_classes = len(class_names)
print(f"Classes: {class_names}, Num Classes: {num_classes}")

encoder = OneHotEncoder(sparse_output=False)
all_classes = np.arange(num_classes).reshape(-1, 1)
encoder.fit(all_classes)

batch_size = 32
loader = data_loader(x_train, y_train, batch_size, shuffle=True, prefetch_size=0, as_contiguous=True)
test_loader = data_loader(x_test, y_test, batch_size, shuffle=False, prefetch_size=0, as_contiguous=True)

class Dishes_CNN(nn.Module):
    def __init__(self, num_classes=10, img_size=(32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, device='cuda')
        self.bn1 = nn.BatchNorm2d(64, device='cuda')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, device='cuda')
        self.bn2 = nn.BatchNorm2d(128, device='cuda')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, device='cuda')
        self.bn3 = nn.BatchNorm2d(256, device='cuda')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        fh = img_size[0] // 8
        fw = img_size[1] // 8
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(256 * fh * fw, num_classes, device='cuda')

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
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = Dishes_CNN(num_classes, img_size=(128, 128))
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 50
scheduler = WarmupCosineLR(optimizer, warmup_epochs=10, T_max=num_epochs, eta_min=1e-5)
nn.init.kaiming_normal_(model.conv1.weight)
nn.init.kaiming_normal_(model.conv2.weight)
nn.init.kaiming_normal_(model.conv3.weight)
nn.init.kaiming_normal_(model.fc.weight)

checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'dishes_cnn_cuda_128_bs64_checkpoint.pkl')
start_epoch = 0

checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"确保检查点目录 {checkpoint_dir} 存在")

train_losses = []
test_accuracies = []
gpu_memory_usage = [] # List to store GPU memory usage

# Initialize NVML
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Monitor the first GPU
        print(f"Monitoring GPU: {pynvml.nvmlDeviceGetName(handle)}")
    else:
        handle = None
        print("No GPU found for monitoring.")
except Exception as e:
    handle = None
    print(f"Failed to initialize NVML: {e}")

try:
    if os.path.exists(checkpoint_path):
        print(f"发现检查点 {checkpoint_path}，尝试加载...")
        checkpoint_dict = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch = checkpoint_dict.get('epoch', 0)
        if hasattr(optimizer, 'v'):
            optimizer.v = [backend_api.zeros_like(p.data) for p in optimizer.params]
        if hasattr(optimizer, 's'):
            optimizer.s = [backend_api.zeros_like(p.data) for p in optimizer.params]
        if hasattr(optimizer, 't'):
            optimizer.t = 1
        for _ in range(start_epoch):
            scheduler.step()
        if os.path.exists(checkpoint_path + '.info'):
            try:
                import dill as pickle
                with open(checkpoint_path + '.info', 'rb') as f:
                    save_info = pickle.load(f)
                    train_losses = save_info.get('train_losses', [])
                    test_accuracies = save_info.get('test_accuracies', [])
                    gpu_memory_usage = save_info.get('gpu_memory_usage', []) # Load GPU history if available
                print("成功加载训练历史数据")
            except Exception as e:
                print(f"加载训练历史数据失败: {e}，使用空列表继续")
                train_losses = []
                test_accuracies = []
                gpu_memory_usage = []
        print(f"成功加载检查点！从第 {start_epoch+1} 个epoch继续训练")
    else:
        print("未发现检查点，从头开始训练")
except Exception as e:
    print(f"加载检查点失败: {e}，从头开始训练")
    start_epoch = 0
    train_losses = []
    test_accuracies = []
    gpu_memory_usage = []

target_acc = 85.0 # Adjusted target accuracy as dishes might be easier/harder, but 60 is a bit low for general purpose
# If restarting from 0 but variables were loaded, ensure they are reset if needed. 
# But here we rely on the checkpoint logic above.
# If start_epoch is 0, we should probably reset these lists unless we want to keep history?
# The original code re-initialized them after the try-catch block if loading failed, but NOT if loading succeeded.
# However, lines 160-161 in original code reset them! Wait.
# In original code lines 160-161:
# train_losses = []
# test_accuracies = []
# This bug existed in the original code? It resets the lists right after loading them!
# Let's fix this bug while we are at it, moving initialization before loading.
# Actually, looking at the original code:
# Lines 123-124 init empty lists.
# Lines 125-157 try to load.
# Lines 160-161 reset them again! 
# So the loading of history in the original code was pointless unless I misread.
# Ah, I see lines 160-161 in the read output. Yes, it resets them.
# I will fix this logic to preserve loaded history if it exists.

if start_epoch == 0:
    train_losses = []
    test_accuracies = []
    gpu_memory_usage = []

train_batch_losses = []
test_batch_accuracies = []

t0 = time.time()

def augment_batch(inputs, epoch):
    bs, c, h, w = inputs.shape
    pad = 16
    padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    ys = np.random.randint(0, 2 * pad + 1, size=bs)
    xs = np.random.randint(0, 2 * pad + 1, size=bs)
    out = np.empty_like(inputs)
    for i in range(bs):
        out[i] = padded[i, :, ys[i]:ys[i] + h, xs[i]:xs[i] + w]
    flip_mask = np.random.rand(bs) < 0.5
    out[flip_mask] = out[flip_mask][:, :, :, ::-1]
    if epoch < num_epochs - 10 and np.random.rand() < 0.2:
        erase_h = builtins.max(1, int(h * np.random.uniform(0.1, 0.2)))
        erase_w = builtins.max(1, int(w * np.random.uniform(0.1, 0.2)))
        ys_e = np.random.randint(0, h - erase_h + 1, size=bs)
        xs_e = np.random.randint(0, w - erase_w + 1, size=bs)
        for i in range(bs):
            out[i, :, ys_e[i]:ys_e[i]+erase_h, xs_e[i]:xs_e[i]+erase_w] = 0.0
    return out

for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    
    # Record max GPU memory usage for this epoch
    max_memory_used = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = augment_batch(inputs, epoch)
        eps = 0.02
        if epoch >= num_epochs - 10:
            eps = 0.0
        labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
        labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
        bs = inputs.shape[0]
        if epoch < num_epochs - 10 and np.random.rand() < 0.5:
            lam = np.random.beta(0.2, 0.2)
            idx = np.random.permutation(bs)
            inputs = lam * inputs + (1 - lam) * inputs[idx]
            labels_onehot = lam * labels_onehot + (1 - lam) * labels_onehot[idx]
        inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
        outputs = model(inputs)
        loss = criterion(outputs, labels_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data.numpy().item()
        train_batch_losses.append(loss.data.numpy().item())
        
        # Check GPU usage
        if handle:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / 1024 / 1024
            if used_mb > max_memory_used:
                max_memory_used = used_mb
                
        if batch_idx % 2 == 0 or batch_idx + 1 == len(loader.batch_sampler):
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(loader.batch_sampler)}] 当前Loss: {loss.data.numpy().item():.4f}")
        outputs.dispose()
        loss.dispose()
        inputs.dispose()
        labels_onehot.dispose()
        del inputs, labels_onehot, outputs, loss
        Graph.free_graph()
        if batch_idx % 50 == 0:
            gc.collect()
            
    gpu_memory_usage.append(max_memory_used)
    
    train_loss = running_loss / len(loader.batch_sampler)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s | Max GPU Mem: {max_memory_used:.2f}MB")

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
    scheduler.step()
    if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs or accuracy >= target_acc:
        print(f"保存模型检查点到 {checkpoint_path}")
        backup_path = checkpoint_path + '.backup'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            save_info = {
                'epoch': epoch,
                'train_losses': train_losses,
                'test_accuracies': test_accuracies,
                'gpu_memory_usage': gpu_memory_usage
            }
            if os.path.exists(checkpoint_path + '.info'):
                os.replace(checkpoint_path + '.info', checkpoint_path + '.info.backup')
            if os.path.exists(checkpoint_path):
                os.replace(checkpoint_path, backup_path)
            with open(checkpoint_path + '.info', 'wb') as f:
                import dill as pickle
                pickle.dump(save_info, f)
            print(f"训练历史数据已保存到 {checkpoint_path + '.info'}")
            save_checkpoint(model, optimizer, epoch, train_losses[-1] if train_losses else None, checkpoint_path)
            print(f"模型检查点已保存到 {checkpoint_path}")
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except Exception as e:
            print(f"保存检查点时出错: {e}")
            if os.path.exists(backup_path):
                os.replace(backup_path, checkpoint_path)
                print(f"已从备份恢复检查点")
            if os.path.exists(checkpoint_path + '.info.backup'):
                os.replace(checkpoint_path + '.info.backup', checkpoint_path + '.info')
                print(f"已从备份恢复训练历史数据")
    if accuracy >= target_acc:
        print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
        break

    Graph.free_graph_all()
    gc.collect()

Graph.free_graph_all()
gc.collect()

# Shutdown NVML
try:
    if handle:
        pynvml.nvmlShutdown()
except Exception as e:
    print(f"Error shutting down NVML: {e}")

plt.figure(figsize=(18, 5)) # Increased figure size for 3 plots
print(f"Total Training Time: {time.time()-t0:.2f}s")

plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', color='orange')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, len(gpu_memory_usage) + 1), gpu_memory_usage, marker='o', color='green')
plt.title('GPU Memory Usage')
plt.xlabel('Epoch')
plt.ylabel('Memory (MB)')
plt.grid(True)

plt.tight_layout()
plt.savefig('dishes_cnn_training_gpu.png', dpi=150, bbox_inches='tight')
plt.show()
print("训练完成！图表已保存为 'dishes_cnn_training_gpu.png'")
print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
