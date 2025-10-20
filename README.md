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
import torch
import torch.nn as nn
from main import SSMLayer, ExperienceBuffer, MetaLearner

# Initialize components
input_dim = 64
hidden_dim = 128
action_dim = 10

# SSM for efficient sequence encoding
ssm_encoder = SSMLayer(input_dim, hidden_dim)

# ExperienceBuffer for experience-based reasoning
experience_buffer = ExperienceBuffer(
    capacity=1000,
    state_dim=hidden_dim,
    action_dim=action_dim
)

# Meta-learner for fast adaptation
meta_learner = MetaLearner(
    input_dim=hidden_dim,
    output_dim=action_dim,
    hidden_dim=256,
    meta_lr=0.01
)

# Training loop with experience accumulation
def train_step(task_batch):
    for task in task_batch:
        # Encode task context with SSM
        context = ssm_encoder(task['observations'])
        
        # Meta-learning inner loop
        adapted_params = meta_learner.adapt(
            context,
            task['support_set']
        )
        
        # Store experience for future retrieval
        experience_buffer.add(
            state=context,
            action=task['actions'],
            reward=task['rewards'],
            context_embedding=context.mean(dim=0)
        )
        
        # Compute loss on query set
        predictions = meta_learner.forward(
            task['query_set'],
            params=adapted_params
        )
        loss = compute_loss(predictions, task['query_labels'])
        loss.backward()

# Test-time adaptation with experience retrieval
def test_adapt(new_task):
    # Encode new task
    context = ssm_encoder(new_task['observations'])
    
    # Retrieve similar past experiences
    similar_experiences = experience_buffer.retrieve(
        query_embedding=context.mean(dim=0),
        k=5
    )
    
    # Adapt using both meta-learning AND retrieved experiences
    adapted_params = meta_learner.adapt_with_experience(
        context=context,
        support_set=new_task['support_set'],
        experiences=similar_experiences
    )
    
    # Make predictions with experience-informed adaptation
    predictions = meta_learner.forward(
        new_task['query_set'],
        params=adapted_params
    )
    
    return predictions
```

### Key Integration Points

1. **During Training**: 
   - SSM processes sequences efficiently
   - Meta-learner performs fast adaptation
   - ExperienceBuffer accumulates task experiences

2. **During Testing**:
   - Retrieve k-nearest experiences from buffer
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
pip install torch numpy
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- (Optional) Mamba SSM for optimized state space models

## Future Directions

- Hierarchical experience buffers for multi-scale reasoning
- Attention-based experience retrieval mechanisms
- Integration with other efficient architectures (RetNet, RWKV)
- Distributed experience sharing across agents

## Acknowledgments

Inspired by SSM-MetaRL-TestCompute and research in meta-learning, state space models, and memory-augmented neural networks.

## License

MIT License - See LICENSE file for details
