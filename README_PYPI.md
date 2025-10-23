# README.md Update for PyPI Badge

Add this section at the top of README.md after the package is uploaded to PyPI:

```markdown
[![PyPI version](https://badge.fury.io/py/lownocompute-ai-baseline.svg)](https://badge.fury.io/py/lownocompute-ai-baseline)
[![Downloads](https://pepy.tech/badge/lownocompute-ai-baseline)](https://pepy.tech/project/lownocompute-ai-baseline)
```

## Installation Section Update

Replace the current installation section with:

```markdown
## 🚀 Quick Start

### Installation

#### From PyPI (Recommended)

```bash
pip install lownocompute-ai-baseline
```

#### From Source

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
```

## Usage Section Update

Add this after installation:

```markdown
### Basic Usage

```python
# Install the package
# pip install lownocompute-ai-baseline

import numpy as np
from lownocompute_ai_baseline import LightweightSSM, ExperienceBuffer, MinimalMAML

# Initialize components
ssm = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
buffer = ExperienceBuffer(max_size=200)
maml = MinimalMAML(model=ssm, inner_lr=0.01, outer_lr=0.001)

# Use the model
output = ssm.forward(np.array([0.5]))
print(f"Output: {output}")
```

### With Configuration

```python
from lownocompute_ai_baseline.configs import load_config, get_ssm_config

config = load_config()
ssm_cfg = get_ssm_config(config)
ssm = LightweightSSM(**ssm_cfg)
```
```

