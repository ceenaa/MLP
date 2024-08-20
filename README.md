# MLP from Scratch in Python using Numpy

This project is an implementation of a Multilayer Perceptron (MLP) from scratch using only Python and Numpy. The implementation includes core components such as tensor operations, a simple automatic differentiation system, and a basic neural network structure.


## Features

- **Tensor Operations**: Supports basic operations like addition, subtraction, multiplication, division, and exponentiation.
- **Automatic Differentiation**: Implements a basic system to calculate gradients using backward propagation, using Computational Graph(DAG), topological sort to implement a pytorch-like auto grad tool..
- **MLP Architecture**: Build your own multilayer perceptron with customizable input sizes and layer configurations.
- **Optimization**: Basic gradient descent optimizer implemented to train the network.
- **Example with Iris Dataset**: A Jupyter Notebook is provided to demonstrate how the MLP can be trained on the Iris dataset.

## Implementation Details

* Tensor Class
The Tensor class is the core data structure, representing both the data and its gradient. It supports basic operations like addition, subtraction, multiplication, division, and exponentiation, each with backward methods to accumulate gradients.

* Neuron Class
The Neuron class represents a single neuron in the network, with a set of weights and a bias. It supports forward propagation and uses the tanh activation function.

* Layer Class
The Layer class is a collection of neurons, performing forward propagation through all neurons in the layer.

* MLP Class
The MLP (Multilayer Perceptron) class is a simple feedforward neural network made up of multiple layers. It can be customized with various layer sizes.

* Optimizer Class
The Optimizer class implements a basic gradient descent algorithm to update the parameters of the network based on the gradients.

## Computational Graph examples for auto diff
![image](https://github.com/user-attachments/assets/be6b0791-8b99-4b9f-814f-796aa87b4f76)

![image](https://github.com/user-attachments/assets/b6b791ad-4060-45fa-b79f-d4fe14f62bc5)

## Comparison to PyTorch Auto-diff
The automatic differentiation system implemented in this project is conceptually similar to PyTorch's auto-diff feature. Like PyTorch, this implementation builds a computation graph dynamically as operations are applied, allowing for efficient backpropagation of gradients through the network. However, this implementation is much simpler and is designed for educational purposes to illustrate the core concepts of auto-differentiation.

## Example with Iris Dataset
An example of training this MLP on the Iris dataset is provided in the end of the MLP.ipynb Jupyter Notebook.

