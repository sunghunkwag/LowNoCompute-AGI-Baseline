# LowNoCompute-AI-Baseline

[![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen)](tests/test_main.py)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code Status](https://img.shields.io/badge/status-production%20ready-brightgreen)]()

A minimal, modular AI baseline framework for meta-learning and policy orchestration under strict low-compute or no-compute constraints. Inspired by, and extending, the SSM-MetaRL-TestCompute repository: this project explores methods for effective generalization and adaptation with State Space Models, meta-RL, and automated test-time adaptation—targeting the most resource-limited environments.

## 🎯 Key Features

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

## 🏗️ Architecture Overview

```
Input → SSM Encoder → Meta-Learner (MAML) ⟷ ExperienceBuffer → Policy → Output
                            ↓                        ↑
                      Fast Adaptation        Experience Retrieval
```

The system combines:
- **SSM layers** for efficient sequence processing
- **Meta-learning** for rapid task adaptation
- **ExperienceBuffer** for experience-driven reasoning and stability

## ✅ Verified & Tested

All code has been **live tested and verified** to work correctly:

- ✅ **19/19 unit tests passing** (100% pass rate)
- ✅ **All 6 scripts execute successfully**
- ✅ **Zero runtime errors**
- ✅ **Complete integration testing**
- ✅ **Production ready**

See [LIVE_TEST_RESULTS.md](LIVE_TEST_RESULTS.md) for detailed test execution results.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/sunghunkwag/LowNoCompute-AI-Baseline.git
cd LowNoCompute-AI-Baseline
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy (>=1.21.0)
- PyYAML (>=5.4.0) - for configuration management
- (Optional) JAX for potential auto-differentiation improvements

### Run the Demo

```bash
python main.py
```

**Expected Output**:
```
Minimal AI/Meta-RL Baseline with Experience-Based Reasoning
============================================================
Starting meta-training for 10 episodes...
Episode 5/10 completed. Buffer size: 200
Episode 10/10 completed. Buffer size: 200
Meta-training completed!

=== Test-Time Adaptation Example ===
...
Result: Experience-based reasoning led to a more accurate adaptation. ✓
```

## 📚 Usage Examples

### Basic Setup with ExperienceBuffer

```python
import numpy as np
from main import LightweightSSM, ExperienceBuffer, MinimalMAML, generate_simple_task

# Initialize components
ssm_model = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
experience_buffer = ExperienceBuffer(max_size=200)
maml = MinimalMAML(model=ssm_model, inner_lr=0.01, outer_lr=0.001)

# Training loop with experience accumulation
for episode in range(10):
    task_batch = []
    for _ in range(4):
        support_set, query_set = generate_simple_task('sine')
        task_batch.append({'support': support_set, 'query': query_set})
        
        # Store experiences for future retrieval
        experience_buffer.add(support_set)
        experience_buffer.add(query_set)
    
    # Meta-update using task batch
    maml.meta_update(task_batch)
    print(f"Episode {episode + 1} completed. Buffer size: {len(experience_buffer)}")

# Test-time adaptation with experience retrieval
test_support, test_query = generate_simple_task('sine')

# Adapt with experience buffer
maml.inner_update(
    support_data=test_support,
    steps=3,
    experience_buffer=experience_buffer,
    experience_batch_size=10
)
```

### Using Configuration Files

```python
from configs.config_loader import load_config, get_ssm_config, get_maml_config
from main import LightweightSSM, MinimalMAML

# Load configuration from YAML
config = load_config('configs/config.yaml')

# Initialize components with config
ssm_cfg = get_ssm_config(config)
maml_cfg = get_maml_config(config)

model = LightweightSSM(**ssm_cfg)
maml = MinimalMAML(model=model, inner_lr=maml_cfg['inner_lr'], outer_lr=maml_cfg['outer_lr'])
```

## 🧪 Running Examples and Tests

### Basic Component Usage

```bash
# Run comprehensive examples of all components
python examples/basic_usage.py
```

**Output**: Demonstrates LightweightSSM, ExperienceBuffer, MinimalMAML, and experience-enhanced adaptation.

### Config-Based Execution

```bash
# Run with YAML configuration
python examples/main_with_config.py
```

**Output**: Shows how to use `config.yaml` to configure all framework parameters.

### Running Tests

```bash
# Run comprehensive test suite (19 unit tests)
python tests/test_main.py
```

**Expected Result**:
```
Ran 19 tests in 0.068s
OK
```

## 📊 Test Results Summary

| Component | Tests | Status |
|-----------|-------|--------|
| LightweightSSM | 6 | ✅ All Passed |
| ExperienceBuffer | 5 | ✅ All Passed |
| MinimalMAML | 4 | ✅ All Passed |
| Utility Functions | 3 | ✅ All Passed |
| Integration | 1 | ✅ All Passed |
| **Total** | **19** | **✅ 100% Pass Rate** |

## 📖 Documentation

- **[LIVE_TEST_RESULTS.md](LIVE_TEST_RESULTS.md)** - Actual test execution results with output
- **[COMPREHENSIVE_TEST_REPORT.md](COMPREHENSIVE_TEST_REPORT.md)** - Complete testing and verification report
- **[BUGFIX.md](BUGFIX.md)** - Bug fixes and solutions documentation
- **[CHANGELOG.md](CHANGELOG.md)** - Complete change history
- **[docs/USAGE.md](docs/USAGE.md)** - Detailed API reference and usage guide

## 🔧 Implementation Notes

### Design Philosophy

- **CPU-First**: Optimized for CPU-only environments with minimal dependencies
- **Minimal Footprint**: Small memory usage suitable for edge devices
- **Educational**: Simple, readable code with extensive documentation
- **Extensible**: Modular design for easy experimentation and extension
- **Production Ready**: Fully tested and verified to work correctly

### Performance Considerations

- Uses finite difference gradients for simplicity and stability
- Float64 precision for numerical stability in gradient computation
- Circular buffer with automatic memory management
- Linear time complexity for SSM operations

### Key Integration Points

1. **During Training**: 
   - SSM processes sequences efficiently with `LightweightSSM`
   - Meta-learner performs fast adaptation with `MinimalMAML`
   - ExperienceBuffer accumulates task experiences with `add()`

2. **During Testing**:
   - Retrieve k-nearest experiences from buffer with `get_batch()`
   - Use experiences to stabilize and improve adaptation
   - Combine meta-learned initialization with experience-driven refinement

## 🎓 Why This Approach Works

The combination of meta-learning, SSM, and experience-based reasoning creates a synergistic effect:

- **SSM**: Provides efficient, linear-complexity sequence modeling
- **Meta-Learning**: Enables fast adaptation from few samples
- **ExperienceBuffer**: Adds memory and stability, preventing catastrophic forgetting and reducing adaptation variance

Together, these components enable robust performance even when:
- Compute resources are severely limited
- Training data is scarce
- Test-time adaptation must be fast and stable
- Continual learning is required

## 🔮 Future Directions

- **JAX Implementation**: For auto-differentiation and GPU acceleration
- **Attention Mechanisms**: For more sophisticated experience retrieval
- **Hierarchical Buffers**: For multi-scale reasoning
- **Distributed Buffers**: For multi-agent experience sharing
- Integration with other efficient architectures (RetNet, RWKV)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by SSM-MetaRL-TestCompute and research in meta-learning, state space models, and memory-augmented neural networks.

## 💬 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Status**: ✅ Production Ready | **Tests**: 19/19 Passing | **Python**: 3.8+ | **License**: MIT

