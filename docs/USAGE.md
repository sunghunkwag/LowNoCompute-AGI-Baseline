# Usage Guide

## LowNoCompute-AI-Baseline

This guide provides detailed instructions on how to use the LowNoCompute-AI-Baseline neural network implementation.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Training a Model](#training-a-model)
5. [Testing](#testing)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependency is `numpy` for numerical computations.

---

## Quick Start

Here's a minimal example to get you started:

```python
import numpy as np
from main import initialize_weights, train_step, forward_pass

# Initialize weights
weights = initialize_weights(input_size=10, hidden_size=5, output_size=3)

# Generate dummy data
X = np.random.randn(32, 10)  # 32 samples, 10 features
y = np.random.randn(32, 3)   # 32 samples, 3 outputs

# Train for one step
loss = train_step(X, y, weights, learning_rate=0.01)
print(f"Training loss: {loss}")

# Make predictions
predictions, _ = forward_pass(X, weights)
print(f"Predictions shape: {predictions.shape}")
```

---

## Configuration

The project includes a configuration file at `configs/config.yaml` that you can customize:

```yaml
model:
  input_size: 10
  hidden_size: 5
  output_size: 3
  learning_rate: 0.01

training:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
```

### Loading Configuration

To use the configuration in your code:

```python
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_size = config['model']['input_size']
hidden_size = config['model']['hidden_size']
output_size = config['model']['output_size']
```

---

## Training a Model

### Full Training Loop

For a complete training example, see `examples/basic_usage.py`:

```bash
python examples/basic_usage.py
```

### Custom Training

```python
import numpy as np
from main import initialize_weights, train_step

# Prepare data
X_train = np.random.randn(100, 10)
y_train = np.random.randn(100, 3)

# Initialize model
weights = initialize_weights(10, 5, 3)

# Training loop
for epoch in range(100):
    loss = train_step(X_train, y_train, weights, learning_rate=0.01)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
```

---

## Testing

### Running Unit Tests

The project includes comprehensive unit tests in the `tests/` directory:

```bash
python -m pytest tests/
```

Or run the test file directly:

```bash
python tests/test_main.py
```

### Test Coverage

The tests cover:
- Weight initialization
- Forward pass computation
- Loss calculation
- Backward pass (gradients)
- Weight updates
- Numerical stability

---

## API Reference

### Core Functions

#### `initialize_weights(input_size, hidden_size, output_size)`

Initializes network weights with random values.

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_size` (int): Number of hidden units
- `output_size` (int): Number of output units

**Returns:**
- `weights` (dict): Dictionary containing 'W1', 'b1', 'W2', 'b2'

#### `forward_pass(X, weights)`

Performs forward propagation through the network.

**Parameters:**
- `X` (np.ndarray): Input data of shape (batch_size, input_size)
- `weights` (dict): Network weights

**Returns:**
- `output` (np.ndarray): Network predictions of shape (batch_size, output_size)
- `cache` (dict): Intermediate values for backward pass

#### `compute_loss(predictions, targets)`

Computes mean squared error loss.

**Parameters:**
- `predictions` (np.ndarray): Model predictions
- `targets` (np.ndarray): Ground truth targets

**Returns:**
- `loss` (float): MSE loss value

#### `train_step(X, y, weights, learning_rate)`

Performs one complete training step (forward, loss, backward, update).

**Parameters:**
- `X` (np.ndarray): Input batch
- `y` (np.ndarray): Target batch
- `weights` (dict): Network weights (updated in-place)
- `learning_rate` (float): Learning rate

**Returns:**
- `loss` (float): Training loss for this step

---

## Examples

### Example 1: Basic Training

See `examples/basic_usage.py` for a complete training example.

### Example 2: Using Dummy Environment

The `environments/dummy_env.py` provides a simple environment for testing:

```python
from environments.dummy_env import DummyEnvironment

env = DummyEnvironment(observation_dim=10, action_dim=3)
obs = env.reset()

for _ in range(5):
    action = np.random.randint(0, env.action_dim)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()
```

---

## Troubleshooting

### Common Issues

#### Import Error

**Problem:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

#### NaN Loss Values

**Problem:** Loss becomes NaN during training

**Solution:** Try reducing the learning rate:
```python
learning_rate = 0.001  # Instead of 0.01
```

#### Slow Training

**Problem:** Training is very slow

**Solution:** Reduce the number of epochs or use smaller batch sizes.

---

## Additional Resources

- [Main README](../README.md)
- [Example Code](../examples/)
- [Test Suite](../tests/)
- [Configuration Files](../configs/)

---

## Contributing

Feel free to open issues or submit pull requests to improve this project!

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
