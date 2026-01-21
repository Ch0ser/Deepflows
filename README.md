# DeepFlows

[**English**](README_en.md) | [**简体中文**](README.md)

DeepFlows 是一个轻量级的深度学习框架教学/实验项目，包含张量封装、自动求导、后端抽象、常用神经网络模块与优化器，以及一些示例训练脚本与服务模块。

## 主要目录结构（简要）

- `DeepFlows/`
  - `tensor.py` — 张量与计算图核心实现，包含 `Tensor`、`Graph` 等。（参见 `DeepFlows/tensor.py`）
  - `autograd.py` — 自动求导开关（`no_grad` / `enable_grad`）与上下文管理。（参见 `DeepFlows/autograd.py`）
  - `backend/` — 后端抽象与 `BackendTensor`（CPU/CPU-Numpy/CUDA 协议），用于多后端兼容。（参见 `DeepFlows/backend/backend_tensor.py`）
  - `nn/` — 神经网络模块与函数（包含 `Module` 基类、常用层、损失函数等）。核心 `Module` 实现位于 `nn/modules/module.py`（参见 `DeepFlows/nn/modules/module.py`）。
  - `optim/` — 优化器（Adam/SGD/Adagrad/Adadelta 等）。
  - `utils/` — 数据加载、评估与可视化工具。
  - `dist/`、`MyDLFW_serving/` — 分布式与模型服务原型/示例。

## 快速开始

- 环境准备
  - 安装 `python>=3.8` 与 `numpy`。
  - 可选：使用 CUDA 后端需具备 CUDA 工具链与编译好的后端 `.pyd`（见下文），可选择自主编译或直接导入编译后的 `.pyd` 文件。

- 数据准备
  - MNIST：将原始 IDX 文件已位于 `Deepflows\data\MNIST\raw`（含 `train-images-idx3-ubyte`、`train-labels-idx1-ubyte`、`t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte`）。
  - CIFAR-10：将 Python 版批次数据已位于 `Deepflows\data\cifar-10-batches-py`（含 `data_batch_1..5`、`test_batch`）。

- 运行示例脚本 (详细说明请见 [test/scripts_description_cn.md](test/scripts_description_cn.md))
  - **基础与 MLP**
    - 线性回归 (CPU)：`python test/LinearRegression.py`
    - MLP-MNIST (CPU/CUDA)：`python test/MLP_MNIST.py` / `python test/MLP_MNIST_cuda.py`
  - **CNN (卷积神经网络)**
    - MNIST (CPU/CUDA)：`python test/CNN_MNIST.py` / `python test/CNN_MNIST_cuda.py`
    - CIFAR-10 (CPU/CUDA)：`python test/CNN_CIFAR10.py` / `python test/CNN_CIFAR10_cuda.py`
    - Animal-10 (CUDA)：`python test/CNN_Animal10_cuda.py`
    - Dishes (CUDA)：`python test/CNN_Dishes_cuda.py`
  - **高级架构 (ResNet/MobileNet)**
    - ResNet (Animal-10/CIFAR-10, CUDA)：`python test/ResNet_Animal10_cuda.py`, `python test/ResNet_CIFAR10_cuda.py`
    - MobileNet 实现：`test/MobileNet.py`
  - **功能测试**
    - CUDA 底层测试：`python test/test_cuda.py`
    - 模型保存加载测试：`python test/CNN_CIFAR10_cuda_model_save_load_test.py`

> 注意：CUDA 版脚本会显式使用 `device='cuda'`。若未启用 CUDA 后端，相关脚本将无法运行，请先完成后端编译或导入编译产物。

## 核心用法要点

- 张量与后端：使用 `Tensor(...)` 或直接 `BackendTensor(...)` 创建数据；后端自动桥接 NumPy / 自定义后端。
- 自动求导：默认开启，通过 `with DeepFlows.no_grad():` 或 `with DeepFlows.enable_grad():` 显式控制（详见 `autograd.py`）。
- 构建模型：继承 `nn.Module`，实现 `forward`，使用 `model.parameters()` 与优化器配合训练。

## 示例引用代码位置

- 包导出入口：`DeepFlows/__init__.py`
- 张量核心：`DeepFlows/tensor.py`
- 自动求导：`DeepFlows/autograd.py`
- 后端抽象：`DeepFlows/backend/backend_tensor.py`
- 示例脚本：`test/` 目录下包含多种网络架构（CNN, ResNet, MobileNet）与数据集（MNIST, CIFAR-10, Animal-10）的训练脚本。详见 [test/scripts_description_cn.md](test/scripts_description_cn.md)。

## 预训练模型与迁移学习

DeepFlows 支持加载常见视觉模型的预训练权重（源自 PyTorch），这为迁移学习提供了基础。相关核心实现位于 `DeepFlows/utils/pretrained_models.py`。

### 主要功能
1. **模型支持**: 目前支持 `resnet18`, `resnet50`, `mobilenet_v1`, `vgg16` 等模型。
2. **权重转换**: 提供了从 PyTorch 权重格式到 DeepFlows 权重格式的自动转换工具。
3. **自动管理**: 支持自动下载、缓存和加载预训练权重。

### 快速上手
可以通过以下方式快速创建一个加载了预训练权重的模型：

```python
from DeepFlows.utils.pretrained_models import create_model_with_pretrained_weights

# 自动下载(如需)、转换并创建加载好权重的模型
# 支持的模型名: 'resnet18', 'resnet50', 'mobilenet_v1', 'vgg16'
model = create_model_with_pretrained_weights('resnet18')
```

或者运行测试脚本体验完整流程：
```bash
python test/test_pretrained_models.py
```

该脚本展示了如何列出可用模型、下载权重、转换格式以及加载到模型中的完整过程。

## 常见问题与假设

- 假设你已有 Python 与 NumPy。CPU 模式无需额外编译。
- 使用 GPU 需启用 CUDA 后端：项目会尝试 `from DeepFlows.backend.backend_src.build.Release import CUDA_BACKEND`，若导入失败则视为未启用。

## CUDA 后端：编译与导入

- 自主编译（Windows）
  - 依赖：CUDA 工具链（含 NVCC）、CMake、Visual Studio Build Tools。
  - 步骤：
    - 进入 `DeepFlows/backend/backend_src`，使用 CMake 生成 VS 工程，构建 `Release`。
    - 编译成功后产物位于 `DeepFlows/backend/backend_src/build/Release/CUDA_BACKEND.pyd`。
    - 该路径与项目的导入语句匹配，无需额外配置。

- 直接导入编译产物 `.pyd`
  - 将已编译好的 `CUDA_BACKEND.pyd` 放入 `DeepFlows/backend/backend_src/build/Release/`。
  - Python 将通过 `from DeepFlows.backend.backend_src.build.Release import CUDA_BACKEND` 自动加载。

- 验证后端是否启用
  - 在 Python REPL 中执行：
    - `from DeepFlows.backend.backend_tensor import cuda`
    - `print(cuda().enabled())`
  - 若输出为 `True`，表示已正确加载 CUDA 后端；否则将回退为未启用状态。

## 可视化平台使用指南 (DeepFlows Visualization Platform)

欢迎使用 DeepFlows 可视化平台！这是一个基于 Vue 3 + FastAPI 的全栈应用，旨在通过直观的 Web 界面展示我们自研深度学习框架 `DeepFlows` 的训练过程与模型构建能力。

### 🛠️ 环境准备与安装

#### 1. 基础要求
- **Python**: 3.10 或更高版本
- **Node.js**: 16.0 或更高版本 (推荐使用 LTS)
- **CUDA (可选)**: 如果需要 GPU 加速，请确保已安装 NVIDIA CUDA Toolkit (推荐 11.x 或 12.x)

#### 2. 后端配置 (Visualization_backend)

后端负责运行深度学习框架、处理训练任务和通过 WebSocket 推送实时数据。

##### 步骤 1: 创建虚拟环境
建议使用 `venv` 或 `conda` 创建隔离环境，避免污染全局 Python 环境。

```powershell
# 在项目根目录下
python -m venv venv
# 激活环境 (Windows)
.\venv\Scripts\Activate.ps1
# 激活环境 (Linux/Mac)
source venv/bin/activate
```

##### 步骤 2: 安装依赖
我们需要安装 FastAPI 服务相关依赖以及监控工具。

```bash
cd Visualization_backend
pip install fastapi uvicorn[standard] websockets psutil nvidia-ml-py scikit-learn
```

> **注意**: 核心框架 `DeepFlows` 依赖于底层的 C++ 编译扩展 (`.pyd` 或 `.so`)。请确保 `DeepFlows/backend/backend_src/build/Release` 目录下已有编译好的文件，或者按照框架文档先行编译。

##### 步骤 3: 启动后端服务
```bash
# 确保在激活的虚拟环境中
python server.py
```
成功启动后，你会看到：
- `INFO: Uvicorn running on http://0.0.0.0:8000`
- `[System] Server starting up...`

#### 3. 前端配置 (Visualization_frontend)

前端提供可视化交互界面，包括仪表盘、模型构建器和训练配置。

##### 步骤 1: 安装依赖
```bash
cd Visualization_frontend
npm install
```

##### 步骤 2: 配置环境变量 (可选)
默认情况下，前端会尝试连接本地 `http://127.0.0.1:8000`。如果需要修改，请编辑 `.env` 文件：
```env
VITE_API_URL=http://127.0.0.1:8000/api
VITE_WS_URL=ws://127.0.0.1:8000/ws
```

##### 步骤 3: 启动前端开发服务器
```bash
npm run dev
```
浏览器访问控制台输出的地址 (通常是 `http://localhost:3000`) 即可进入系统。

### 🚀 功能使用指南

#### 1. 仪表盘 (Dashboard)
- **实时监控**: 展示当前训练的 Epoch, Batch, Loss 和 Accuracy。
- **可视化图表**:
  - **Loss & Accuracy**: 动态折线图，实时绘制训练曲线。
  - **Resource Usage**: 实时显示本机的 CPU、内存和 GPU (若有) 使用率。
- **全屏模式**: 点击右上角的全屏图标，获得沉浸式体验。

#### 2. 模型构建器 (Model Builder)
这是一个“低代码”模型搭建工具，允许你通过拖拽方式设计神经网络。

- **添加层**: 从左侧面板拖拽 `Conv2d`, `Linear`, `ReLU` 等层到中间画布。
- **编辑参数**:
  - 点击层卡片上的 **编辑图标** (✏️)。
  - 在弹出的 JSON 编辑框中修改参数（例如修改 `in_features`, `out_features`）。
  - **重要**: 请确保层与层之间的维度匹配（例如上一层输出 128，下一层输入必须是 128）。
- **删除层**: 鼠标悬停在层上，点击 **删除图标** (🗑️)。
- **保存模型**:
  - 点击右侧的 **Save Model**。
  - 输入自定义模型名称（如 "My MLP v1"）。
  - 保存成功后会自动跳转到配置页面。

#### 3. 训练配置 (Training Config)
在这里配置训练超参数并启动任务。

- **Model Selection**:
  - **Default MNIST_CNN**: 系统内置的卷积神经网络。
  - **自定义模型**: 下拉选择你在 Model Builder 中保存的模型。
- **Dataset**: 目前支持 `MNIST` (真实数据)。`CIFAR-10` 选项暂时也会回退到 MNIST（因输入通道兼容性原因）。
- **Hyperparameters**:
  - **Batch Size**: 批次大小 (推荐 32 或 64)。
  - **Epochs**: 训练轮数。
  - **Learning Rate**: 学习率 (推荐 0.001 - 0.01)。
  - **Optimizer**: 支持 `SGD`, `Adam`, `Adagrad`。
- **控制**:
  - **Start Training**: 发送配置到后端，开始训练。
  - **Stop Training**: 强制停止当前训练任务。

### 🧩 可视化平台常见问题 (FAQ)

**Q: 为什么 Resource Usage 图表中 GPU 显示为 0？**
A: 请检查是否安装了 `nvidia-ml-py` 依赖，以及本机是否拥有 NVIDIA 显卡并正确安装了驱动。如果没有 GPU，后端会自动忽略并显示 0。

**Q: 自定义模型报错 "Shape mismatch"？**
A: 这是深度学习中最常见的问题。请仔细检查 Model Builder 中每一层的参数，特别是 `Linear` 层的输入输出维度。对于 MNIST，输入图片大小为 28x28，`Flatten` 后是 784。

**Q: 页面提示 "Network Error"？**
A: 请检查后端 `server.py` 是否正在运行，且端口 `8000` 未被防火墙拦截。

## 贡献者

<a href="https://github.com/Ch0ser">
  <img src="https://github.com/Ch0ser.png" width="60px;" alt="Ch0ser"/>
</a>
<a href="https://github.com/iJcher">
  <img src="https://github.com/iJcher.png" width="60px;" alt="iJcher"/>
</a>
<a href="https://github.com/xwh12345-user">
   <img src="https://github.com/xwh12345-user.png" width="60px;" alt="xwh12345-user"/>
 </a>
 <a href="https://github.com/RegentZai">
    <img src="https://github.com/RegentZai.png" width="60px;" alt="RegentZai"/>
  </a>
  <a href="https://github.com/Lymtics-Aa">
    <img src="https://github.com/Lymtics-Aa.png" width="60px;" alt="Lymtics-Aa"/>
  </a>

## 贡献与联系

欢迎提 Issue / PR。你可以先从修复文档、补充测试用例或完善后端实现开始。

