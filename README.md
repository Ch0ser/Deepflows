# DeepFlows

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

- 运行示例脚本（CPU）
  - 线性回归：`python test/LinearRegression.py`
  - MLP-MNIST：`python test/MLP_MNIST.py`
  - CNN-MNIST（CPU）：`python test/CNN_MNIST.py`
  - CNN-CIFAR10（CPU）：`python test/CNN_CIFAR10.py`

- 运行示例脚本（CUDA，需要后端已启用）
  - MLP-MNIST（CUDA）：`python test/MLP_MNIST_cuda.py`
  - CNN-MNIST（CUDA）：`python test/CNN_MNIST_cuda.py`
  - CNN-CIFAR10（CUDA）：`python test/CNN_CIFAR10_cuda.py`

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
- 示例脚本：`test/CNN_MNIST.py`, `test/LinearRegression.py`

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

## 贡献与联系

欢迎提 Issue / PR。你可以先从修复文档、补充测试用例或完善后端实现开始。

