# 测试脚本说明

本文档简要介绍了 `test` 目录下的脚本功能。这些脚本主要用于测试和展示 DeepFlows 框架的能力，涵盖了多种模型架构和数据集。

## CUDA 后端测试
- [test_cuda.py](test_cuda.py): CUDA 后端扩展 (`CUDA_BACKEND`) 的底层测试脚本。验证基本操作，如数组创建、内存传输和内核执行（例如 `fill`）。

## 基础机器学习示例
- [LinearRegression.py](LinearRegression.py): 使用 CPU 的简单线性回归示例。拟合直线 $y = 3x + 5$ 到合成数据。

## MLP (多层感知机)
- [MLP_MNIST.py](MLP_MNIST.py): 在 MNIST 数据集上训练的 3 层 MLP (CPU)。
- [MLP_MNIST_cuda.py](MLP_MNIST_cuda.py): 在 MNIST 数据集上训练的 3 层 MLP (CUDA 加速)。

## CNN (卷积神经网络)
### MNIST 数据集
- [CNN_MNIST.py](CNN_MNIST.py): 在 MNIST 数据集上训练的 CNN 模型 (CPU)。
- [CNN_MNIST_cuda.py](CNN_MNIST_cuda.py): 在 MNIST 数据集上训练的 CNN 模型 (CUDA)。

### CIFAR-10 数据集
- [CNN_CIFAR10.py](CNN_CIFAR10.py): 在 CIFAR-10 数据集上训练的 CNN 模型 (CPU)。
- [CNN_CIFAR10_cuda.py](CNN_CIFAR10_cuda.py): 在 CIFAR-10 数据集上训练的 CNN 模型 (CUDA)。
- [CNN_CIFAR10_cuda_model_save_load_test.py](CNN_CIFAR10_cuda_model_save_load_test.py): 测试 CUDA 上 CIFAR-10 CNN 模型的保存和加载功能。

### Animal-10 数据集
- [CNN_Animal10_cuda.py](CNN_Animal10_cuda.py): 在 Animal-10 数据集上训练的 CNN 模型 (CUDA)。
- [CNN_Animal10_cudacopy.py](CNN_Animal10_cudacopy.py): Animal-10 训练脚本的备份/副本。

### Dishes (菜品) 数据集
- [CNN_Dishes_cuda.py](CNN_Dishes_cuda.py): 在 Dishes 数据集上训练的 CNN 模型 (CUDA)。改编自 Animal-10 脚本。

## 高级架构
### ResNet (残差网络)
- [ResNet.py](ResNet.py): ResNet 组件（残差块）的实现。
- [ResNet_Animal10_cuda.py](ResNet_Animal10_cuda.py): 在 Animal-10 数据集上训练的 ResNet 模型 (CUDA)。
- [ResNet_CIFAR10_cuda.py](ResNet_CIFAR10_cuda.py): 在 CIFAR-10 数据集上训练的 ResNet 模型 (CUDA)。

### MobileNet
- [MobileNet.py](MobileNet.py): MobileNetV1 架构（深度可分离卷积）的实现。

---
*注意：带有 `_cuda` 后缀的脚本需要 GPU 以及正确构建和配置 DeepFlows CUDA 后端。*
