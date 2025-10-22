"""
LowNoCompute-AI-Baseline: A minimal AI framework for meta-learning under low-compute constraints.

This package provides:
- LightweightSSM: Efficient state space model for sequence processing
- ExperienceBuffer: Memory buffer for experience-based reasoning
- MinimalMAML: Minimal meta-learning implementation
- Configuration utilities for easy setup
"""

__version__ = '1.0.0'
__author__ = 'Sunghun Kwag'
__license__ = 'MIT'

from .core import (
    LightweightSSM,
    ExperienceBuffer,
    MinimalMAML,
    mse,
    generate_simple_task,
)

__all__ = [
    'LightweightSSM',
    'ExperienceBuffer',
    'MinimalMAML',
    'mse',
    'generate_simple_task',
    '__version__',
]

