#!/usr/bin/env python3
"""
Example: Running main.py with configuration from YAML file.

This demonstrates how to use the config_loader to run the experience-based
reasoning framework with parameters loaded from config.yaml.
"""

import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import LightweightSSM, ExperienceBuffer, MinimalMAML, generate_simple_task
from configs.config_loader import load_config, get_ssm_config, get_maml_config, get_buffer_config, get_training_config


def main_with_config(config_path=None):
    """Run the experience-based reasoning framework using configuration file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default config.yaml
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Extract component configurations
    ssm_cfg = get_ssm_config(config)
    maml_cfg = get_maml_config(config)
    buffer_cfg = get_buffer_config(config)
    training_cfg = get_training_config(config)
    
    print("\n" + "=" * 60)
    print("Experience-Based Reasoning with Config")
    print("=" * 60)
    print(f"SSM: input_dim={ssm_cfg['input_dim']}, hidden_dim={ssm_cfg['hidden_dim']}, output_dim={ssm_cfg['output_dim']}")
    print(f"MAML: inner_lr={maml_cfg['inner_lr']}, outer_lr={maml_cfg['outer_lr']}")
    print(f"Buffer: max_size={buffer_cfg['max_size']}")
    print(f"Training: {training_cfg['num_episodes']} episodes, {training_cfg['tasks_per_batch']} tasks/batch")
    print("=" * 60)
    
    # Initialize components with config
    model = LightweightSSM(
        input_dim=ssm_cfg['input_dim'],
        hidden_dim=ssm_cfg['hidden_dim'],
        output_dim=ssm_cfg['output_dim']
    )
    
    maml = MinimalMAML(
        model=model,
        inner_lr=maml_cfg['inner_lr'],
        outer_lr=maml_cfg['outer_lr']
    )
    
    experience_buffer = ExperienceBuffer(max_size=buffer_cfg['max_size'])
    
    # Meta-training
    print(f"\nStarting meta-training for {training_cfg['num_episodes']} episodes...")
    
    for episode in range(training_cfg['num_episodes']):
        task_batch = []
        
        for _ in range(training_cfg['tasks_per_batch']):
            # Generate task using configured task types
            task_type = random.choice(training_cfg['task_types'])
            support_set, query_set = generate_simple_task(task_type)
            task_batch.append({'support': support_set, 'query': query_set})
            
            # Store experiences
            experience_buffer.add(support_set)
            experience_buffer.add(query_set)
        
        # Meta-update
        maml.meta_update(task_batch, eps=maml_cfg['finite_diff_eps'])
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{training_cfg['num_episodes']} completed. Buffer size: {len(experience_buffer)}")
    
    print("\nMeta-training completed!")
    print(f"Final experience buffer size: {len(experience_buffer)}")
    
    # Test-time adaptation demonstration
    print("\n" + "=" * 60)
    print("Test-Time Adaptation with Experience Buffer")
    print("=" * 60)
    
    # Save meta-learned parameters
    meta_learned_params = model.get_params()
    
    # Generate a new test task
    print("\nGenerating new test task...")
    test_support, test_query = generate_simple_task('sine')
    
    # Adaptation without buffer
    print("\n[Without Experience Buffer]")
    model.set_params(meta_learned_params)
    model.reset_state()
    
    # Compute initial loss
    initial_loss = 0.0
    for x, y in test_query:
        model.reset_state()
        pred = model.forward(x)
        initial_loss += (pred[0] - y[0]) ** 2
    initial_loss /= len(test_query)
    
    # Adapt
    maml.inner_update(
        test_support, 
        steps=maml_cfg['inner_steps'],
        eps=maml_cfg['finite_diff_eps'],
        experience_buffer=None
    )
    
    # Compute adapted loss
    adapted_loss_no_buffer = 0.0
    for x, y in test_query:
        model.reset_state()
        pred = model.forward(x)
        adapted_loss_no_buffer += (pred[0] - y[0]) ** 2
    adapted_loss_no_buffer /= len(test_query)
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Adapted loss: {adapted_loss_no_buffer:.4f}")
    print(f"  Improvement:  {initial_loss - adapted_loss_no_buffer:.4f}")
    
    # Adaptation with buffer
    print("\n[With Experience Buffer]")
    model.set_params(meta_learned_params)
    model.reset_state()
    
    maml.inner_update(
        test_support,
        steps=maml_cfg['inner_steps'],
        eps=maml_cfg['finite_diff_eps'],
        experience_buffer=experience_buffer,
        experience_batch_size=buffer_cfg['experience_batch_size']
    )
    
    # Compute adapted loss with buffer
    adapted_loss_with_buffer = 0.0
    for x, y in test_query:
        model.reset_state()
        pred = model.forward(x)
        adapted_loss_with_buffer += (pred[0] - y[0]) ** 2
    adapted_loss_with_buffer /= len(test_query)
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Adapted loss: {adapted_loss_with_buffer:.4f}")
    print(f"  Improvement:  {initial_loss - adapted_loss_with_buffer:.4f}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Loss without buffer: {adapted_loss_no_buffer:.4f}")
    print(f"Loss with buffer:    {adapted_loss_with_buffer:.4f}")
    
    if adapted_loss_with_buffer < adapted_loss_no_buffer:
        improvement = adapted_loss_no_buffer - adapted_loss_with_buffer
        print(f"✓ Experience buffer improved adaptation by {improvement:.4f}")
    else:
        print("✗ Experience buffer did not improve adaptation in this run")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main_with_config()

