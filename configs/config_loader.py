"""Configuration loader for LowNoCompute-AI-Baseline.

This module provides utilities to load and validate configuration from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        # Default to config.yaml in the same directory
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_ssm_config(config: Dict[str, Any]) -> Dict[str, int]:
    """Extract LightweightSSM configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with SSM parameters (input_dim, hidden_dim, output_dim)
    """
    ssm_config = config.get('ssm', {})
    return {
        'input_dim': ssm_config.get('input_dim', 1),
        'hidden_dim': ssm_config.get('hidden_dim', 8),
        'output_dim': ssm_config.get('output_dim', 1)
    }


def get_maml_config(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract MinimalMAML configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with MAML parameters (inner_lr, outer_lr, etc.)
    """
    maml_config = config.get('maml', {})
    return {
        'inner_lr': maml_config.get('inner_lr', 0.01),
        'outer_lr': maml_config.get('outer_lr', 0.001),
        'inner_steps': maml_config.get('inner_steps', 3),
        'finite_diff_eps': maml_config.get('finite_diff_eps', 1e-5)
    }


def get_buffer_config(config: Dict[str, Any]) -> Dict[str, int]:
    """Extract ExperienceBuffer configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with buffer parameters (max_size, experience_batch_size)
    """
    buffer_config = config.get('experience_buffer', {})
    return {
        'max_size': buffer_config.get('max_size', 200),
        'experience_batch_size': buffer_config.get('experience_batch_size', 10)
    }


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract meta-training configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with training parameters
    """
    training_config = config.get('meta_training', {})
    return {
        'num_episodes': training_config.get('num_episodes', 10),
        'tasks_per_batch': training_config.get('tasks_per_batch', 4),
        'task_types': training_config.get('task_types', ['sine', 'linear'])
    }


def get_task_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract task generation configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with task generation parameters
    """
    task_config = config.get('task_generation', {})
    return {
        'support_samples': task_config.get('support_samples', 5),
        'query_samples': task_config.get('query_samples', 10),
        'input_range': task_config.get('input_range', [-2, 2]),
        'noise_std': task_config.get('noise_std', 0.0)
    }


def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration.
    
    Args:
        config: Configuration dictionary to print
    """
    print("=" * 60)
    print("Configuration Loaded:")
    print("=" * 60)
    
    print("\n[LightweightSSM]")
    ssm_cfg = get_ssm_config(config)
    for key, value in ssm_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[MinimalMAML]")
    maml_cfg = get_maml_config(config)
    for key, value in maml_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[ExperienceBuffer]")
    buffer_cfg = get_buffer_config(config)
    for key, value in buffer_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[Meta-Training]")
    training_cfg = get_training_config(config)
    for key, value in training_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[Task Generation]")
    task_cfg = get_task_generation_config(config)
    for key, value in task_cfg.items():
        print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print_config(config)
        print("\n✓ Configuration loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")

