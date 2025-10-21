# LowNoCompute-AI-Baseline

A minimal, modular AI baseline framework for meta-learning and policy orchestration under strict low-compute or no-compute constraints. Inspired by, and extending, the SSM-MetaRL-TestCompute repository: this project explores methods for effective generalization and adaptation with State Space Models, meta-RL, and automated test-time adaptation—targeting the most resource-limited environments.

## Key Features

### Experience-Based Reasoning Architecture

This framework introduces a novel **experience-based reasoning** approach that goes beyond traditional meta-learning by incorporating a dynamic **ExperienceBuffer** component for test-time adaptation. The architecture combines three powerful paradigms:

1. **Minimal Meta-Learning (MAML-style)**: Fast adaptation with gradient-based meta-learning
2. **Sequential State Space Models (SSM)**: Efficient temporal modeling with linear complexity
3. **ExperienceBuffer**: A core component that stores and retrieves relevant past experiences for robust, experience-driven adaptation

### Why Experience-Based Reasoning?

While pure meta-learning provides a good initialization, **experience-based reasoning** enables more stable and effective adaptation in ultra-low-compute settings by:

- **Leveraging past experiences**: The ExperienceBuffer dynamically stores task episodes and retrieves similar experiences during test time
- **Reducing adaptation variance**: Experience-driven reasoning provides more stable gradients and reduces overfitting to limited samples
- **Enabling continual learning**: The buffer accumulates knowledge across tasks, creating a growing knowledge base
- **Improving sample efficiency**: By referencing similar past scenarios, the model adapts faster with fewer samples

### Core Component: ExperienceBuffer

The `ExperienceBuffer` is the heart of experience-based reasoning:

- Stores task experiences with states, actions, rewards, and context embeddings
- Retrieves k-nearest experiences based on similarity matching
- Integrates seamlessly with meta-learning and SSM components
- Operates efficiently even under extreme compute constraints

## Architecture Overview

```
Input → SSM Encoder → Meta-Learner (MAML) ⟷ ExperienceBuffer → Policy → Output
                            ↓                        ↑
                      Fast Adaptation        Experience Retrieval
```

The system combines:
- **SSM layers** for efficient sequence processing
- **Meta-learning** for rapid task adaptation
- **ExperienceBuffer** for experience-driven reasoning and stability

## Usage Example

### Basic Setup with ExperienceBuffer

```python
import numpy as np
from main import LightweightSSM, ExperienceBuffer, MinimalMAML

# Initialize components
input_dim = 1
hidden_dim = 8
output_dim = 1

# SSM for efficient sequence encoding
ssm_model = LightweightSSM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# ExperienceBuffer for experience-based reasoning
experience_buffer = ExperienceBuffer(max_size=200)

# Meta-learner for fast adaptation
maml = MinimalMAML(model=ssm_model, inner_lr=0.01, outer_lr=0.001)

# Training loop with experience accumulation
def train_meta_learning():
    num_episodes = 10
    batch_size = 4
    
    for episode in range(num_episodes):
        task_batch = []
        
        for _ in range(batch_size):
            # Generate a task (sine wave or linear function)
            support_set, query_set = generate_simple_task('sine')  # from main.py
            task_batch.append({'support': support_set, 'query': query_set})
            
            # Store experiences for future retrieval
            experience_buffer.add(support_set)
            experience_buffer.add(query_set)
        
        # Meta-update using task batch
        maml.meta_update(task_batch)
        print(f"Episode {episode + 1} completed. Buffer size: {len(experience_buffer)}")

# Test-time adaptation with experience retrieval
def test_adaptation(new_task_data):
    """Adapt to a new task using both meta-learning and past experiences."""
    
    # Method 1: Adaptation without experience buffer
    ssm_model.reset_state()
    adapted_params_no_buffer = maml.inner_update(
        support_data=new_task_data, 
        steps=3, 
        experience_buffer=None
    )
    
    # Method 2: Adaptation WITH experience buffer
    ssm_model.reset_state()
    adapted_params_with_buffer = maml.inner_update(
        support_data=new_task_data,
        steps=3,
        experience_buffer=experience_buffer,
        experience_batch_size=10
    )
    
    # Test predictions
    test_input = np.array([0.8])
    
    ssm_model.set_params(adapted_params_no_buffer)
    ssm_model.reset_state()
    pred_no_buffer = ssm_model.forward(test_input)
    
    ssm_model.set_params(adapted_params_with_buffer)
    ssm_model.reset_state()
    pred_with_buffer = ssm_model.forward(test_input)
    
    return pred_no_buffer, pred_with_buffer

# Example usage - run the full demo
if __name__ == "__main__":
    from main import main
    main()  # Runs complete meta-training + test-time adaptation demo
```

### Key Integration Points

1. **During Training**: 
   - SSM processes sequences efficiently with `LightweightSSM`
   - Meta-learner performs fast adaptation with `MinimalMAML`
   - ExperienceBuffer accumulates task experiences with `add()`

2. **During Testing**:
   - Retrieve k-nearest experiences from buffer with `get_batch()`
   - Use experiences to stabilize and improve adaptation
   - Combine meta-learned initialization with experience-driven refinement

## Why This Approach Works

The combination of meta-learning, SSM, and experience-based reasoning creates a synergistic effect:

- **SSM**: Provides efficient, linear-complexity sequence modeling
- **Meta-Learning**: Enables fast adaptation from few samples
- **ExperienceBuffer**: Adds memory and stability, preventing catastrophic forgetting and reducing adaptation variance

Together, these components enable robust performance even when:
- Compute resources are severely limited
- Training data is scarce
- Test-time adaptation must be fast and stable
- Continual learning is required

## Installation

```bash
git clone https://github.com/sunghunkwag/LowNoCompute-AI-Baseline.git
cd LowNoCompute-AI-Baseline
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- NumPy (>=1.21.0)
- PyYAML (>=5.4.0) - for configuration management
- (Optional) JAX for potential auto-differentiation improvements

## Running the Demo

```bash
python main.py
```

This will run a complete demonstration including:
1. Meta-training on synthetic sine/linear tasks
2. Building an experience buffer during training
3. Test-time adaptation comparison (with vs without experience buffer)
4. Performance analysis showing the benefits of experience-based reasoning

## Additional Examples

### Basic Component Usage

```bash
# Run comprehensive examples of all components
python examples/basic_usage.py
```

### Using Configuration Files

```bash
# Run with YAML configuration
python examples/main_with_config.py
```

This example demonstrates how to use `config.yaml` to configure all framework parameters.

### Running Tests

```bash
# Run comprehensive test suite
python tests/test_main.py
```

## Implementation Notes

### Design Philosophy

- **CPU-First**: Optimized for CPU-only environments with minimal dependencies
- **Minimal Footprint**: Small memory usage suitable for edge devices
- **Educational**: Simple, readable code with extensive documentation
- **Extensible**: Modular design for easy experimentation and extension

### Performance Considerations

- Uses finite difference gradients for simplicity and stability
- Float64 precision for numerical stability in gradient computation
- Circular buffer with automatic memory management
- Linear time complexity for SSM operations

### Potential Improvements

- **JAX Implementation**: For auto-differentiation and GPU acceleration
- **Attention Mechanisms**: For more sophisticated experience retrieval
- **Hierarchical Buffers**: For multi-scale reasoning
- **Distributed Buffers**: For multi-agent experience sharing

## Future Directions

- Hierarchical experience buffers for multi-scale reasoning
- Attention-based experience retrieval mechanisms
- Integration with other efficient architectures (RetNet, RWKV)
- Distributed experience sharing across agents
- JAX-based implementation for auto-differentiation

## Acknowledgments

Inspired by SSM-MetaRL-TestCompute and research in meta-learning, state space models, and memory-augmented neural networks.

## License

MIT License - See LICENSE file for details