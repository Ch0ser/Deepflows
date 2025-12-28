# Test Scripts Description

This document provides a brief overview of the scripts located in the `test` directory. These scripts are primarily used for testing and demonstrating the capabilities of the DeepFlows framework, including various model architectures and datasets.

## CUDA Backend Tests
- [test_cuda.py](test_cuda.py): Low-level testing script for the CUDA backend extension (`CUDA_BACKEND`). It verifies basic operations like array creation, memory transfer, and kernel execution (e.g., `fill`).

## Basic Machine Learning Examples
- [LinearRegression.py](LinearRegression.py): A simple linear regression example using CPU. Fits a line $y = 3x + 5$ to synthetic data.

## MLP (Multi-Layer Perceptron)
- [MLP_MNIST.py](MLP_MNIST.py): 3-layer MLP trained on MNIST dataset using CPU.
- [MLP_MNIST_cuda.py](MLP_MNIST_cuda.py): 3-layer MLP trained on MNIST dataset using CUDA acceleration.

## CNN (Convolutional Neural Networks)
### MNIST Dataset
- [CNN_MNIST.py](CNN_MNIST.py): CNN model trained on MNIST dataset using CPU.
- [CNN_MNIST_cuda.py](CNN_MNIST_cuda.py): CNN model trained on MNIST dataset using CUDA.

### CIFAR-10 Dataset
- [CNN_CIFAR10.py](CNN_CIFAR10.py): CNN model trained on CIFAR-10 dataset (CPU).
- [CNN_CIFAR10_cuda.py](CNN_CIFAR10_cuda.py): CNN model trained on CIFAR-10 dataset (CUDA).
- [CNN_CIFAR10_cuda_model_save_load_test.py](CNN_CIFAR10_cuda_model_save_load_test.py): Tests the model saving and loading functionality for the CIFAR-10 CNN model on CUDA.

### Animal-10 Dataset
- [CNN_Animal10_cuda.py](CNN_Animal10_cuda.py): CNN model trained on the Animal-10 dataset using CUDA.
- [CNN_Animal10_cudacopy.py](CNN_Animal10_cudacopy.py): Backup/Duplicate of the Animal-10 training script.

### Dishes Dataset
- [CNN_Dishes_cuda.py](CNN_Dishes_cuda.py): CNN model trained on the Dishes dataset using CUDA. Adapted from the Animal-10 script.

## Advanced Architectures
### ResNet
- [ResNet.py](ResNet.py): Implementation of ResNet components (Residual Blocks).
- [ResNet_Animal10_cuda.py](ResNet_Animal10_cuda.py): ResNet model trained on Animal-10 dataset using CUDA.
- [ResNet_CIFAR10_cuda.py](ResNet_CIFAR10_cuda.py): ResNet model trained on CIFAR-10 dataset using CUDA.

### MobileNet
- [MobileNet.py](MobileNet.py): Implementation of MobileNetV1 architecture (Depthwise Separable Convolutions).

---
*Note: Scripts with `_cuda` suffix require a GPU and the DeepFlows CUDA backend to be properly built and configured.*
