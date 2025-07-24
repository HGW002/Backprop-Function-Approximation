# Backpropagation Neural Network for Function Approximation

This project implements a simple **feedforward neural network** trained using **backpropagation** to approximate the function:

\[
g(p) = 1 + \sin\left(\frac{\pi p}{2}\right), \quad -2 \leq p \leq 2
\]

##  Problem Source
> *Neural Network Design*  
> by Martin T. Hagan, Howard B. Demuth, and Mark H. Beale

## Network Architecture

- **Type**: 1–S₁–1 Feedforward Neural Network
- **Activation Functions**:
  - Hidden layer: `logsig` (sigmoid)
  - Output layer: `purelin` (linear)

## Features

- Matrix-based forward and backward pass
- Customizable:
  - Learning rate (α)
  - Number of hidden neurons (S₁)
  - Weight/bias initialization


### Example Convergence Behavior

| Learning Rate | Hidden Units (S₁) | Converges? | Comment           |
|---------------|-------------------|------------|--------------------|
| 0.1           | 2                 | ✅         | Slow, stable       |
| 0.2           | 10                | ✅         | Fast convergence   |
| 0.4           | 10                | ❌         | Oscillates         |

