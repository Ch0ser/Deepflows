import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import builtins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pynvml  # Import pynvml for GPU monitoring
import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
                    arr = arr.transpose(2, 0, 1) # HWC to CHW
                    X.append(arr)
                    Y.append(class_to_idx[cname])
                except Exception as e:
                    print(f"Skipping error file: {path}, error: {e}")
                    
    x = np.stack(X).astype(np.float32) / 255.0
    y = np.asarray(Y, dtype=np.int64) # Torch uses int64 for labels usually
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1/7, random_state=42, stratify=y
    )
    mean_c = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std_c = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x_train = (x_train - mean_c) / std_c
    x_test = (x_test - mean_c) / std_c
    return x_train, y_train, x_test, y_test, class_names

# Dataset class
class DishesDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

root_dir = r"e:\P.A.R.A\Project\ComprehensiveDesign\codes\Deepflows\data\dishes"
print(f"Loading data from {root_dir}...")
# Same parameters as the DeepFlows script
img_size = (32, 32)
x_train, y_train, x_test, y_test, class_names = load_dishes_data(root_dir, img_size=img_size)
num_classes = len(class_names)
print(f"Classes: {class_names}, Num Classes: {num_classes}")

batch_size = 32 # Matched with DeepFlows script

train_dataset = DishesDataset(x_train, y_train)
test_dataset = DishesDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class Dishes_CNN(nn.Module):
    def __init__(self, num_classes=10, img_size=(32, 32)):
        super(Dishes_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        fh = img_size[0] // 8
        fw = img_size[1] // 8
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(256 * fh * fw, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = Dishes_CNN(num_classes, img_size=img_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 50

# Scheduler
# PyTorch's CosineAnnealingLR does not have built-in warmup in older versions or exact match, 
# but we can try to approximate or just use a standard scheduler. 
# For fair comparison, we'll use CosineAnnealingLR which is close to WarmupCosineLR (minus warmup)
# Or we can implement a custom lambda LR.
# DeepFlows WarmupCosineLR: warmup_epochs=10, T_max=num_epochs, eta_min=1e-5
# We will use a composed scheduler to match exactly if possible, or simplified.
# Let's use SequentialLR if available or just CosineAnnealingLR for simplicity as main phase.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 10, eta_min=1e-5)

# Init weights - Kaiming Normal
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
model.apply(init_weights)

checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'dishes_cnn_torch_checkpoint.pth')
start_epoch = 0

checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)

train_losses = []
test_accuracies = []
gpu_memory_usage = [] 

# Initialize NVML
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
        print(f"Monitoring GPU: {pynvml.nvmlDeviceGetName(handle)}")
    else:
        handle = None
        print("No GPU found for monitoring.")
except Exception as e:
    handle = None
    print(f"Failed to initialize NVML: {e}")

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        test_accuracies = checkpoint.get('test_accuracies', [])
        gpu_memory_usage = checkpoint.get('gpu_memory_usage', [])
        print(f"Resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

target_acc = 85.0 

t0 = time.time()

def augment_batch(inputs, epoch):
    # inputs: numpy array [bs, c, h, w]
    # This function is from DeepFlows script, adapted for numpy input (which we have in dataset if we didn't convert to tensor yet)
    # But in Torch DataLoader, we get tensors. Let's convert to numpy for augmentation or reimplement in torch.
    # To be strictly comparable, we should use the exact same logic.
    # The DeepFlows script applies augmentation on the batch returned by loader.
    # DeepFlows loader returns numpy arrays. Torch loader returns tensors.
    
    inputs_np = inputs.cpu().numpy()
    bs, c, h, w = inputs_np.shape
    pad = 4
    padded = np.pad(inputs_np, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    ys = np.random.randint(0, 2 * pad + 1, size=bs)
    xs = np.random.randint(0, 2 * pad + 1, size=bs)
    out = np.empty_like(inputs_np)
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
    return torch.from_numpy(out).to(device)

for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    
    max_memory_used = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Augmentation
        inputs = augment_batch(inputs, epoch)
        
        # Mixup / Smoothing logic from DeepFlows script
        # eps = 0.02 if epoch < num_epochs - 10 else 0.0
        # DeepFlows uses label smoothing + mixup manually
        # To keep it simple and comparable, we'll skip complex mixup reimplementation 
        # unless strictly required, but the user asked for "same factors".
        # Let's try to mimic basic label smoothing at least.
        
        # Label Smoothing
        # labels is long (indices). We need one-hot for manual smoothing or use CrossEntropyLoss label_smoothing (avail in new pytorch)
        # DeepFlows: labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
        
        # Mixup
        # if epoch < num_epochs - 10 and np.random.rand() < 0.5: ...
        
        # Implementing Mixup & Smoothing for strict comparison
        inputs_mixed = inputs
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
        
        eps = 0.02
        if epoch >= num_epochs - 10:
            eps = 0.0
        labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
        
        bs = inputs.shape[0]
        if epoch < num_epochs - 10 and np.random.rand() < 0.5:
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(bs).to(device)
            inputs_mixed = lam * inputs + (1 - lam) * inputs[idx]
            labels_onehot = lam * labels_onehot + (1 - lam) * labels_onehot[idx]
        
        optimizer.zero_grad()
        outputs = model(inputs_mixed)
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Check GPU usage
        if handle:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / 1024 / 1024
            if used_mb > max_memory_used:
                max_memory_used = used_mb
                
        if batch_idx % 2 == 0 or batch_idx + 1 == len(train_loader):
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{len(train_loader)}] 当前Loss: {loss.item():.4f}")

    gpu_memory_usage.append(max_memory_used)
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s | Max GPU Mem: {max_memory_used:.2f}MB")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # No augmentation/mixup for test
            
            # DeepFlows eval also uses onehot for some reason but argmax compares indices
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 1 == 0:
                current_acc = 100 * correct / total
                print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{len(test_loader)}] 当前准确率: {current_acc:.2f}%")

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
    
    # Scheduler step
    # DeepFlows uses WarmupCosineLR. PyTorch scheduler usually steps after epoch.
    # Note: DeepFlows scheduler might be slightly different implementation.
    if epoch >= 10: # Skip warmup phase for CosineAnnealingLR if we want to manually handle or just step it
        scheduler.step()
    
    # Save checkpoint
    if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs or accuracy >= target_acc:
        print(f"保存模型检查点到 {checkpoint_path}")
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'gpu_memory_usage': gpu_memory_usage
        }
        torch.save(save_dict, checkpoint_path)
        print(f"模型检查点已保存到 {checkpoint_path}")

    if accuracy >= target_acc:
        print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
        break
    
    gc.collect()
    torch.cuda.empty_cache()

# Shutdown NVML
try:
    if handle:
        pynvml.nvmlShutdown()
except Exception as e:
    print(f"Error shutting down NVML: {e}")

plt.figure(figsize=(18, 5)) 
print(f"Total Training Time: {time.time()-t0:.2f}s")

plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Training Loss (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', color='orange')
plt.title('Test Accuracy (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, len(gpu_memory_usage) + 1), gpu_memory_usage, marker='o', color='green')
plt.title('GPU Memory Usage (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Memory (MB)')
plt.grid(True)

plt.tight_layout()
plt.savefig('dishes_cnn_torch_training_gpu.png', dpi=150, bbox_inches='tight')
plt.show()
print("训练完成！图表已保存为 'dishes_cnn_torch_training_gpu.png'")
print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
