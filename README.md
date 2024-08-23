# MLP from Scratch in Python using Numpy

This repository contains an implementation of a Multilayer Perceptron (MLP) built entirely from scratch using Python and Numpy. The implementation covers fundamental components such as tensor operations, a simple automatic differentiation system, and the core architecture of a neural network, all without relying on external machine learning libraries.

## Features

- **Custom Tensor Operations**: Implements essential operations (addition, subtraction, multiplication, division, and exponentiation) within a custom `Tensor` class, including support for gradient tracking.
- **Automatic Differentiation**: Features a basic auto-diff system for gradient computation via backward propagation, using a Directed Acyclic Graph (DAG) and topological sorting for efficient gradient flow, similar to PyTorch's autograd.
- **Flexible MLP Architecture**: Design and train your own MLPs with customizable input sizes, layer configurations, and activation functions.
- **Gradient Descent Optimizer**: A simple yet effective gradient descent optimizer to train the neural network.
- **Iris Dataset Example**: A Jupyter Notebook is included to demonstrate how the MLP can be applied to the classic Iris classification problem.

## Implementation Overview

### Tensor Class
The `Tensor` class is the core data structure of the implementation. It represents both the data and its associated gradient and supports basic arithmetic operations with automatic gradient computation. The key methods include:
- **Addition, Subtraction, Multiplication, Division, and Exponentiation**: All basic operations are overloaded to support tensors.
- **Backward Propagation**: Gradients are accumulated via the `_backward` method, and a topological sorting is used to ensure the correct order of operations during backpropagation.

### Neuron Class
The `Neuron` class simulates a single neuron, including:
- **Weights and Bias**: Initialized randomly and adjusted during training.
- **Activation Function**: Uses the `tanh` activation function for non-linearity.
- **Forward Propagation**: Computes the weighted sum of inputs and applies the activation function.

### Layer Class
The `Layer` class consists of multiple neurons and handles:
- **Forward Propagation**: Processes the input through each neuron in the layer.
- **Parameter Management**: Collects the parameters (weights and biases) from all neurons for optimization.

### MLP Class
The `MLP` class represents a fully connected feedforward neural network composed of multiple layers. Key functionalities include:
- **Network Construction**: Layers are sequentially connected to form the network.
- **Forward Propagation**: The input is passed through each layer to produce the output.
- **Parameter Aggregation**: Collects all parameters from all layers for optimization.

### Optimizer Class
The `Optimizer` class implements gradient descent, enabling the network to learn by updating weights and biases:
- **Zero Gradients**: Resets gradients before backpropagation.
- **Step Function**: Updates parameters based on the computed gradients and learning rate.

## Computational Graph and Auto-Diff

![Computational Graph Example](https://github.com/user-attachments/assets/be6b0791-8b99-4b9f-814f-796aa87b4f76)

![Topological Sorting in Computational Graph](https://github.com/user-attachments/assets/b6b791ad-4060-45fa-b79f-d4fe14f62bc5)

The auto-differentiation system in this project closely mirrors PyTorch's dynamic computation graph. By dynamically constructing a graph of operations as tensors interact, this system allows for efficient and automatic computation of gradients via backpropagation.

## Comparison to PyTorch Auto-Diff
While the automatic differentiation system in this project is conceptually similar to PyTorch’s autograd, it is intentionally simplified for educational purposes. The system builds a computation graph on the fly as operations are executed, which allows for efficient gradient calculation during backpropagation.

## Example: Training on the Iris Dataset
To illustrate the practical application of this MLP implementation, a Jupyter Notebook (`MLP.ipynb`) is provided. The notebook demonstrates how to train the network on the Iris dataset, showing step-by-step how to configure the MLP, optimize its parameters, and evaluate its performance.
