# Usage Guide

## LowNoCompute-AI-Baseline

This guide provides detailed instructions on how to use the experience-based reasoning components in the LowNoCompute-AI-Baseline framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install numpy
```

The framework is designed to be minimal and only requires NumPy for numerical computations.

---

## Quick Start

Here's a minimal example to get you started with experience-based reasoning:

```python
import numpy as np
from main import LightweightSSM, ExperienceBuffer, MinimalMAML, generate_simple_task

# 1. Create components
ssm = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
maml = MinimalMAML(model=ssm, inner_lr=0.01)
buffer = ExperienceBuffer(max_size=100)

# 2. Generate and adapt to a task
support_set, query_set = generate_simple_task('sine')
buffer.add(support_set + query_set)  # Store as experience

# 3. Adapt to new task using experiences
new_support, new_query = generate_simple_task('sine')
adapted_params = maml.inner_update(
    new_support, 
    steps=3, 
    experience_buffer=buffer,
    experience_batch_size=10
)

# 4. Make predictions
test_input = np.array([0.5])
ssm.reset_state()
prediction = ssm.forward(test_input)
print(f"Prediction: {prediction[0]:.4f}")
```

---

## Core Components

### LightweightSSM

A minimal State Space Model for efficient sequential processing.

**Key Features:**
- Linear time complexity for sequences
- Stateful processing with reset capability
- Get/set parameter interface for meta-learning
- Pure NumPy implementation for CPU efficiency

### ExperienceBuffer

A memory buffer that stores past task experiences for retrieval during adaptation.

**Key Features:**
- Circular buffer with automatic size management
- Random sampling for experience retrieval
- Integration with meta-learning inner loops
- Minimal memory footprint

### MinimalMAML

A simplified MAML (Model-Agnostic Meta-Learning) implementation with experience buffer support.

**Key Features:**
- Gradient-based fast adaptation
- Optional experience-enhanced adaptation
- Meta-learning outer loop optimization
- Finite difference gradients (CPU-friendly)

---

## Basic Usage

### Working with LightweightSSM

```python
from main import LightweightSSM

# Create SSM
ssm = LightweightSSM(input_dim=2, hidden_dim=4, output_dim=1)

# Process a sequence
ssm.reset_state()  # Always reset before new sequence
inputs = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
for x in inputs:
    output = ssm.forward(x)
    print(f"Input: {x}, Output: {output[0]:.4f}")

# Parameter management
params = ssm.get_params()  # Returns dict with 'A', 'B', 'C', 'D'
modified_params = {k: v + 0.01 for k, v in params.items()}
ssm.set_params(modified_params)
```

### Using ExperienceBuffer

```python
from main import ExperienceBuffer

# Create buffer
buffer = ExperienceBuffer(max_size=50)

# Add experiences (list of (input, output) tuples)
experiences = [
    (np.array([0.1]), np.array([0.2])),
    (np.array([0.3]), np.array([0.6])),
    (np.array([0.5]), np.array([1.0]))
]
buffer.add(experiences)

# Sample from buffer
batch = buffer.get_batch(batch_size=2)
print(f"Sampled {len(batch)} experiences")
for x, y in batch:
    print(f"  x: {x[0]:.3f}, y: {y[0]:.3f}")
```

### Meta-Learning with MinimalMAML

```python
from main import MinimalMAML, generate_simple_task

# Setup
ssm = LightweightSSM(input_dim=1, hidden_dim=6, output_dim=1)
maml = MinimalMAML(model=ssm, inner_lr=0.02, outer_lr=0.001)

# Generate training tasks and meta-train
task_batch = []
for _ in range(3):  # 3 tasks per batch
    support, query = generate_simple_task('sine')
    task_batch.append({'support': support, 'query': query})

# Meta-update (outer loop)
maml.meta_update(task_batch)

# Fast adaptation (inner loop) on new task
new_support, _ = generate_simple_task('linear')
adapted_params = maml.inner_update(new_support, steps=2)
```

---

## Advanced Features

### Experience-Enhanced Adaptation

The key innovation is combining meta-learning with experience retrieval:

```python
# 1. Build experience buffer during meta-training
buffer = ExperienceBuffer(max_size=200)

for episode in range(10):  # Meta-training episodes
    task_batch = []
    for _ in range(4):  # Tasks per batch
        support, query = generate_simple_task('sine')
        task_batch.append({'support': support, 'query': query})
        
        # Accumulate experiences
        buffer.add(support + query)
    
    maml.meta_update(task_batch)

# 2. Test-time adaptation with experience retrieval
new_task_support, new_task_query = generate_simple_task('sine')

# Adaptation WITHOUT experience buffer
adapted_no_buffer = maml.inner_update(
    new_task_support, 
    steps=3, 
    experience_buffer=None
)

# Adaptation WITH experience buffer
adapted_with_buffer = maml.inner_update(
    new_task_support,
    steps=3,
    experience_buffer=buffer,
    experience_batch_size=15
)

# Compare performance on query set...
```

### Task Generation

```python
from main import generate_simple_task

# Generate sine wave task
support, query = generate_simple_task('sine')
# Returns: 5 support samples, 10 query samples
# Each sample is (x, y) where x, y are 1D numpy arrays

# Generate linear function task
support, query = generate_simple_task('linear')
# Same format, different underlying function

# Example of examining generated data
print("Support set:")
for i, (x, y) in enumerate(support):
    print(f"  Sample {i+1}: x={x[0]:.3f}, y={y[0]:.3f}")
```

---

## API Reference

### LightweightSSM Class

#### Constructor
```python
LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
```

#### Methods

**`forward(x: np.ndarray) -> np.ndarray`**
- Process one time step
- **Parameters:** `x` - input vector of shape `(input_dim,)`
- **Returns:** output vector of shape `(output_dim,)`

**`reset_state()`**
- Reset hidden state to zeros
- Call before processing new sequences

**`get_params() -> Dict[str, np.ndarray]`**
- **Returns:** dictionary with keys 'A', 'B', 'C', 'D'

**`set_params(params: Dict[str, np.ndarray])`**
- **Parameters:** `params` - parameter dictionary from `get_params()`

### ExperienceBuffer Class

#### Constructor
```python
ExperienceBuffer(max_size=100)
```

#### Methods

**`add(experience_batch: List[Tuple[np.ndarray, np.ndarray]])`**
- Add experiences to buffer
- **Parameters:** `experience_batch` - list of (input, output) tuples

**`get_batch(batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]`**
- Sample random batch from buffer
- **Parameters:** `batch_size` - number of experiences to sample
- **Returns:** list of (input, output) tuples

**`__len__() -> int`**
- **Returns:** current number of experiences in buffer

### MinimalMAML Class

#### Constructor
```python
MinimalMAML(model, inner_lr=0.01, outer_lr=0.001)
```

#### Methods

**`inner_update(support_data, steps=1, eps=1e-5, experience_buffer=None, experience_batch_size=10) -> Dict[str, np.ndarray]`**
- Perform fast adaptation (inner loop)
- **Parameters:**
  - `support_data` - list of (input, output) tuples
  - `steps` - number of gradient steps
  - `experience_buffer` - optional ExperienceBuffer for experience-enhanced adaptation
  - `experience_batch_size` - number of experiences to sample if buffer provided
- **Returns:** adapted parameter dictionary

**`meta_update(task_batch, eps=1e-5)`**
- Perform meta-learning update (outer loop)
- **Parameters:**
  - `task_batch` - list of task dictionaries with 'support' and 'query' keys

### Utility Functions

**`mse(y_pred: np.ndarray, y_true: np.ndarray) -> float`**
- Compute mean squared error

**`generate_simple_task(task_type='sine') -> Tuple[List, List]`**
- Generate synthetic tasks
- **Parameters:** `task_type` - 'sine' or 'linear'
- **Returns:** (support_set, query_set) tuple

---

## Examples

### Example 1: Basic SSM Usage

See `examples/basic_usage.py` for comprehensive demonstrations.

```bash
python examples/basic_usage.py
```

### Example 2: Full Meta-Training Demo

Run the complete experience-based reasoning demonstration:

```bash
python main.py
```

This demonstrates:
- Meta-training on synthetic tasks
- Experience buffer accumulation
- Test-time adaptation comparison
- Performance analysis

### Example 3: Custom Task Domain

```python
def custom_task_generator():
    """Create custom task for your domain."""
    # Your task generation logic here
    support_set = [...]  # (input, output) tuples
    query_set = [...]    # (input, output) tuples
    return support_set, query_set

# Use with existing framework
ssm = LightweightSSM(input_dim=your_input_dim, hidden_dim=8, output_dim=your_output_dim)
maml = MinimalMAML(model=ssm)
buffer = ExperienceBuffer()

support, query = custom_task_generator()
buffer.add(support + query)
# ... continue with adaptation
```

---

## Troubleshooting

### Common Issues

#### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:** Install NumPy:
```bash
pip install numpy
```

#### NaN Values During Training

**Problem:** Loss or predictions become NaN

**Solutions:**
- Reduce learning rates: `inner_lr=0.001, outer_lr=0.0001`
- Check input data ranges (normalize if necessary)
- Reduce finite difference epsilon: `eps=1e-6`

#### Poor Adaptation Performance

**Problem:** Model doesn't adapt well to new tasks

**Solutions:**
- Increase adaptation steps: `steps=5`
- Tune learning rates
- Increase hidden dimensions: `hidden_dim=16`
- Populate experience buffer with more diverse experiences

#### Memory Usage

**Problem:** Experience buffer uses too much memory

**Solutions:**
- Reduce buffer size: `max_size=50`
- Use smaller batch sizes: `experience_batch_size=5`

#### Slow Training

**Problem:** Training is very slow

**Solutions:**
- The finite difference approach is inherently slower but more stable
- Reduce model dimensions
- Reduce number of adaptation steps
- Consider JAX implementation for auto-differentiation (future work)

### Performance Tips

1. **Start small**: Begin with small dimensions and grow as needed
2. **Monitor gradients**: Check that finite difference gradients are reasonable
3. **Experience quality**: Ensure experience buffer contains relevant, diverse examples
4. **Task similarity**: The method works best when test tasks are similar to training tasks

---

## Contributing

Feel free to open issues or submit pull requests to improve this project!

### Future Enhancements

- JAX-based auto-differentiation for speed
- Hierarchical experience buffers
- Attention-based experience retrieval
- More sophisticated task generators

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.