# DeepFlows

[**简体中文**](README.md) | [**English**](README_en.md)

DeepFlows is a lightweight deep learning framework for teaching and experimentation. It includes tensor encapsulation, automatic differentiation, backend abstraction, common neural network modules and optimizers, as well as some example training scripts and service modules.

## Directory Structure (Brief)

- `DeepFlows/`
  - `tensor.py` — Core implementation of Tensor and Computation Graph. (See `DeepFlows/tensor.py`)
  - `autograd.py` — Autograd switch (`no_grad` / `enable_grad`) and context management. (See `DeepFlows/autograd.py`)
  - `backend/` — Backend abstraction and `BackendTensor` (CPU/CPU-Numpy/CUDA protocols) for multi-backend compatibility. (See `DeepFlows/backend/backend_tensor.py`)
  - `nn/` — Neural network modules and functions (including `Module` base class, common layers, loss functions, etc.). Core `Module` implementation is in `nn/modules/module.py` (See `DeepFlows/nn/modules/module.py`).
  - `optim/` — Optimizers (Adam/SGD/Adagrad/Adadelta, etc.).
  - `utils/` — Data loading, evaluation, and visualization tools.
  - `dist/`, `MyDLFW_serving/` — Distributed training and model serving prototypes/examples.

## Quick Start

- **Environment Preparation**
  - Install `python>=3.8` and `numpy`.
  - Optional: Using the CUDA backend requires the CUDA toolkit and a compiled backend `.pyd` (see below). You can choose to compile it yourself or import the compiled `.pyd` file directly.

- **Data Preparation**
  - MNIST: Place raw IDX files in `Deepflows\data\MNIST\raw` (including `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`).
  - CIFAR-10: Place Python version batch data in `Deepflows\data\cifar-10-batches-py` (including `data_batch_1..5`, `test_batch`).

- **Run Example Scripts** (For details see [test/scripts_description_en.md](test/scripts_description_en.md))
  - **Basic & MLP**
    - Linear Regression (CPU): `python test/LinearRegression.py`
    - MLP-MNIST (CPU/CUDA): `python test/MLP_MNIST.py` / `python test/MLP_MNIST_cuda.py`
  - **CNN (Convolutional Neural Networks)**
    - MNIST (CPU/CUDA): `python test/CNN_MNIST.py` / `python test/CNN_MNIST_cuda.py`
    - CIFAR-10 (CPU/CUDA): `python test/CNN_CIFAR10.py` / `python test/CNN_CIFAR10_cuda.py`
    - Animal-10 (CUDA): `python test/CNN_Animal10_cuda.py`
    - Dishes (CUDA): `python test/CNN_Dishes_cuda.py`
  - **Advanced Architectures (ResNet/MobileNet)**
    - ResNet (Animal-10/CIFAR-10, CUDA): `python test/ResNet_Animal10_cuda.py`, `python test/ResNet_CIFAR10_cuda.py`
    - MobileNet Implementation: `test/MobileNet.py`
  - **Functional Tests**
    - CUDA Low-level Test: `python test/test_cuda.py`
    - Model Save/Load Test: `python test/CNN_CIFAR10_cuda_model_save_load_test.py`

> Note: CUDA version scripts explicitly use `device='cuda'`. If the CUDA backend is not enabled, these scripts will fail to run. Please compile the backend or import the compiled product first.

## Core Usage Points

- **Tensor & Backend**: Use `Tensor(...)` or directly `BackendTensor(...)` to create data; the backend automatically bridges NumPy / custom backends.
- **Autograd**: Enabled by default. Explicitly control via `with DeepFlows.no_grad():` or `with DeepFlows.enable_grad():` (see `autograd.py`).
- **Build Model**: Inherit from `nn.Module`, implement `forward`, and use `model.parameters()` with an optimizer for training.

## Example Code Locations

- Package Export Entry: `DeepFlows/__init__.py`
- Tensor Core: `DeepFlows/tensor.py`
- Autograd: `DeepFlows/autograd.py`
- Backend Abstraction: `DeepFlows/backend/backend_tensor.py`
- Example Scripts: The `test/` directory contains training scripts for various network architectures (CNN, ResNet, MobileNet) and datasets (MNIST, CIFAR-10, Animal-10). See [test/scripts_description_en.md](test/scripts_description_en.md).

## Pretrained Models & Transfer Learning

DeepFlows supports loading pretrained weights from common vision models (sourced from PyTorch), providing a foundation for transfer learning. Core implementation is located in `DeepFlows/utils/pretrained_models.py`.

### Key Features
1. **Model Support**: Currently supports `resnet18`, `resnet50`, `mobilenet_v1`, `vgg16`.
2. **Weight Conversion**: Provides tools to automatically convert weights from PyTorch format to DeepFlows format.
3. **Automatic Management**: Supports automatic downloading, caching, and loading of pretrained weights.

### Quick Start
You can quickly create a model loaded with pretrained weights using:

```python
from DeepFlows.utils.pretrained_models import create_model_with_pretrained_weights

# Automatically download (if needed), convert, and create a model with loaded weights
# Supported model names: 'resnet18', 'resnet50', 'mobilenet_v1', 'vgg16'
model = create_model_with_pretrained_weights('resnet18')
```

Or run the test script to experience the full process:
```bash
python test/test_pretrained_models.py
```

This script demonstrates how to list available models, download weights, convert formats, and load them into a model.

## FAQ & Assumptions

- Assumes you have Python and NumPy. CPU mode requires no extra compilation.
- Using GPU requires enabling the CUDA backend: The project tries `from DeepFlows.backend.backend_src.build.Release import CUDA_BACKEND`. If import fails, it is treated as disabled.

## CUDA Backend: Compilation & Import

- **Self-Compile (Windows)**
  - Dependencies: CUDA Toolkit (including NVCC), CMake, Visual Studio Build Tools.
  - Steps:
    - Enter `DeepFlows/backend/backend_src`, use CMake to generate VS project, build `Release`.
    - Upon success, the product is located at `DeepFlows/backend/backend_src/build/Release/CUDA_BACKEND.pyd`.
    - This path matches the project's import statement, no extra config needed.

- **Directly Import Compiled `.pyd`**
  - Place the compiled `CUDA_BACKEND.pyd` into `DeepFlows/backend/backend_src/build/Release/`.
  - Python will automatically load it via `from DeepFlows.backend.backend_src.build.Release import CUDA_BACKEND`.

- **Verify Backend Enabled**
  - Execute in Python REPL:
    - `from DeepFlows.backend.backend_tensor import cuda`
    - `print(cuda().enabled())`
  - If output is `True`, the CUDA backend is correctly loaded; otherwise, it falls back to disabled status.

## DeepFlows Visualization Platform Guide

Welcome to the DeepFlows Visualization Platform! This is a full-stack application based on Vue 3 + FastAPI, designed to demonstrate the training process and model building capabilities of our self-developed deep learning framework `DeepFlows` via an intuitive Web interface.

### 🛠️ Environment Setup & Installation

#### 1. Basic Requirements
- **Python**: 3.10 or higher
- **Node.js**: 16.0 or higher (LTS recommended)
- **CUDA (Optional)**: If GPU acceleration is needed, ensure NVIDIA CUDA Toolkit is installed (11.x or 12.x recommended).

#### 2. Backend Configuration (Visualization_backend)

The backend is responsible for running the deep learning framework, handling training tasks, and pushing real-time data via WebSockets.

##### Step 1: Create Virtual Environment
It is recommended to use `venv` or `conda` to create an isolated environment to avoid polluting the global Python environment.

```powershell
# In project root
python -m venv venv
# Activate environment (Windows)
.\venv\Scripts\Activate.ps1
# Activate environment (Linux/Mac)
source venv/bin/activate
```

##### Step 2: Install Dependencies
We need to install FastAPI service dependencies and monitoring tools.

```bash
cd Visualization_backend
pip install fastapi uvicorn[standard] websockets psutil nvidia-ml-py scikit-learn
```

> **Note**: The core framework `DeepFlows` depends on the underlying C++ compiled extension (`.pyd` or `.so`). Please ensure that the compiled file exists in the `DeepFlows/backend/backend_src/build/Release` directory, or compile it first according to the framework documentation.

##### Step 3: Start Backend Service
```bash
# Ensure you are in the activated virtual environment
python server.py
```
After successful startup, you will see:
- `INFO: Uvicorn running on http://0.0.0.0:8000`
- `[System] Server starting up...`

#### 3. Frontend Configuration (Visualization_frontend)

The frontend provides a visual interactive interface, including dashboards, model builders, and training configurations.

##### Step 1: Install Dependencies
```bash
cd Visualization_frontend
npm install
```

##### Step 2: Configure Environment Variables (Optional)
By default, the frontend tries to connect to local `http://127.0.0.1:8000`. To modify, edit the `.env` file:
```env
VITE_API_URL=http://127.0.0.1:8000/api
VITE_WS_URL=ws://127.0.0.1:8000/ws
```

##### Step 3: Start Frontend Dev Server
```bash
npm run dev
```
Visit the address output in the console (usually `http://localhost:3000`) to enter the system.

### 🚀 Usage Guide

#### 1. Dashboard
- **Real-time Monitoring**: Shows current Epoch, Batch, Loss, and Accuracy.
- **Visual Charts**:
  - **Loss & Accuracy**: Dynamic line charts, plotting training curves in real-time.
  - **Resource Usage**: Real-time display of CPU, Memory, and GPU (if available) usage.
- **Full Screen Mode**: Click the full-screen icon in the top right for an immersive experience.

#### 2. Model Builder
This is a "Low-Code" model building tool allowing you to design neural networks via drag-and-drop.

- **Add Layer**: Drag `Conv2d`, `Linear`, `ReLU`, etc., from the left panel to the center canvas.
- **Edit Parameters**:
  - Click the **Edit Icon** (✏️) on the layer card.
  - Modify parameters in the JSON editor popup (e.g., change `in_features`, `out_features`).
  - **Important**: Ensure dimensions match between layers (e.g., if previous layer outputs 128, next layer input must be 128).
- **Delete Layer**: Hover over a layer and click the **Delete Icon** (🗑️).
- **Save Model**:
  - Click **Save Model** on the right.
  - Enter a custom model name (e.g., "My MLP v1").
  - Automatically jumps to the configuration page upon success.

#### 3. Training Config
Configure training hyperparameters and start tasks here.

- **Model Selection**:
  - **Default MNIST_CNN**: System built-in CNN.
  - **Custom Models**: Select models you saved in Model Builder from the dropdown.
- **Dataset**: Currently supports `MNIST` (Real data). `CIFAR-10` option will temporarily fallback to MNIST (due to input channel compatibility reasons).
- **Hyperparameters**:
  - **Batch Size**: Batch size (recommended 32 or 64).
  - **Epochs**: Training epochs.
  - **Learning Rate**: Learning rate (recommended 0.001 - 0.01).
  - **Optimizer**: Supports `SGD`, `Adam`, `Adagrad`.
- **Controls**:
  - **Start Training**: Send config to backend and start training.
  - **Stop Training**: Force stop the current training task.

### 🧩 Visualization Platform FAQ

**Q: Why does the GPU show 0 in Resource Usage?**
A: Check if `nvidia-ml-py` dependency is installed, and if the machine has an NVIDIA GPU with drivers correctly installed. If no GPU, the backend automatically ignores it and displays 0.

**Q: Custom model reports "Shape mismatch"?**
A: This is the most common issue in deep learning. Carefully check the parameters of each layer in the Model Builder, especially input/output dimensions of `Linear` layers. For MNIST, input image size is 28x28, which is 784 after `Flatten`.

**Q: Page says "Network Error"?**
A: Check if the backend `server.py` is running and port `8000` is not blocked by a firewall.

## Contributors

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
  <a href="https://github.com/feudalDragon">
    <img src="https://github.com/feudalDragon.png" width="60px;" alt="feudalDragon"/>
  </a>

## Contribution and Contact

Welcome to submit Issues / PRs. You can start by fixing documentation, adding test cases, or improving backend implementation.
