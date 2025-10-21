"""Basic usage example for LowNoCompute-AI-Baseline.

This example demonstrates how to use the experience-based reasoning components
implemented in main.py: LightweightSSM, ExperienceBuffer, and MinimalMAML.
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path to import main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import (
    LightweightSSM,
    ExperienceBuffer, 
    MinimalMAML,
    generate_simple_task,
    mse
)


def demonstrate_ssm_basic():
    """Demonstrate basic LightweightSSM functionality."""
    print("\n=== LightweightSSM Basic Demo ===")
    
    # Create a simple SSM
    ssm = LightweightSSM(input_dim=1, hidden_dim=4, output_dim=1)
    
    print(f"SSM created: input_dim={ssm.input_dim}, hidden_dim={ssm.hidden_dim}, output_dim={ssm.output_dim}")
    
    # Test forward pass
    test_inputs = [0.1, 0.5, -0.3, 0.8, -0.2]
    outputs = []
    
    print("\nSequential processing:")
    ssm.reset_state()
    for i, x in enumerate(test_inputs):
        y = ssm.forward(np.array([x]))
        outputs.append(y[0])
        print(f"  Step {i+1}: input={x:6.3f} -> output={y[0]:6.3f}")
    
    # Demonstrate parameter get/set
    params = ssm.get_params()
    print(f"\nParameters extracted. A matrix shape: {params['A'].shape}")
    
    # Modify parameters slightly and set them back
    modified_params = {k: v + np.random.normal(0, 0.01, v.shape) for k, v in params.items()}
    ssm.set_params(modified_params)
    print("Parameters modified and set back successfully.")


def demonstrate_experience_buffer():
    """Demonstrate ExperienceBuffer functionality."""
    print("\n=== ExperienceBuffer Demo ===")
    
    # Create buffer
    buffer = ExperienceBuffer(max_size=20)
    print(f"Experience buffer created with max_size=20")
    
    # Add some dummy experiences
    experiences = []
    for i in range(15):
        x = np.random.uniform(-1, 1, (1,))
        y = np.sin(x) + np.random.normal(0, 0.1, (1,))
        experiences.append((x, y))
    
    buffer.add(experiences)
    print(f"Added {len(experiences)} experiences. Buffer size: {len(buffer)}")
    
    # Sample from buffer
    sample_batch = buffer.get_batch(5)
    print(f"\nSampled batch of size {len(sample_batch)}:")
    for i, (x, y) in enumerate(sample_batch):
        print(f"  Sample {i+1}: x={x[0]:6.3f}, y={y[0]:6.3f}")
    
    # Add more to test max_size limit
    more_experiences = []
    for i in range(10):
        x = np.random.uniform(-1, 1, (1,))
        y = np.cos(x) + np.random.normal(0, 0.1, (1,))
        more_experiences.append((x, y))
    
    buffer.add(more_experiences)
    print(f"\nAdded {len(more_experiences)} more experiences. Buffer size: {len(buffer)}")
    print("(Note: Buffer maintains max_size=20, oldest entries were automatically removed)")


def demonstrate_maml_basic():
    """Demonstrate basic MinimalMAML functionality."""
    print("\n=== MinimalMAML Basic Demo ===")
    
    # Setup
    ssm = LightweightSSM(input_dim=1, hidden_dim=6, output_dim=1)
    maml = MinimalMAML(model=ssm, inner_lr=0.05, outer_lr=0.001)
    
    print(f"MAML setup: inner_lr={maml.inner_lr}, outer_lr={maml.outer_lr}")
    
    # Generate a simple task
    support_set, query_set = generate_simple_task('sine')
    print(f"\nGenerated sine task: {len(support_set)} support samples, {len(query_set)} query samples")
    
    # Test prediction before adaptation
    original_params = ssm.get_params()
    test_x = np.array([0.5])
    
    ssm.reset_state()
    pred_before = ssm.forward(test_x)
    print(f"\nPrediction before adaptation: {pred_before[0]:6.3f}")
    
    # Perform inner update (adaptation)
    print("\nPerforming inner adaptation (3 steps)...")
    adapted_params = maml.inner_update(support_set, steps=3)
    
    # Test prediction after adaptation
    ssm.reset_state()
    pred_after = ssm.forward(test_x)
    print(f"Prediction after adaptation:  {pred_after[0]:6.3f}")
    
    # Compute loss on support set to show learning
    ssm.set_params(original_params)
    loss_before = 0.0
    for x, y_true in support_set:
        ssm.reset_state()
        y_pred = ssm.forward(x)
        loss_before += mse(y_pred, y_true)
    loss_before /= len(support_set)
    
    ssm.set_params(adapted_params)
    loss_after = 0.0
    for x, y_true in support_set:
        ssm.reset_state()
        y_pred = ssm.forward(x)
        loss_after += mse(y_pred, y_true)
    loss_after /= len(support_set)
    
    print(f"\nSupport set loss before adaptation: {loss_before:6.4f}")
    print(f"Support set loss after adaptation:  {loss_after:6.4f}")
    print(f"Loss reduction: {loss_before - loss_after:6.4f}")


def demonstrate_experience_enhanced_adaptation():
    """Demonstrate adaptation with vs without experience buffer."""
    print("\n=== Experience-Enhanced Adaptation Demo ===")
    
    # Setup
    ssm = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
    maml = MinimalMAML(model=ssm, inner_lr=0.03, outer_lr=0.001)
    buffer = ExperienceBuffer(max_size=50)
    
    # Populate buffer with past experiences (various sine waves)
    print("Populating experience buffer with past sine wave experiences...")
    for _ in range(10):
        support, query = generate_simple_task('sine')
        buffer.add(support + query)
    
    print(f"Experience buffer populated with {len(buffer)} experiences")
    
    # Generate a new test task
    new_support, new_query = generate_simple_task('sine')
    print(f"\nNew task generated with {len(new_support)} support samples")
    
    # Save original parameters
    original_params = ssm.get_params()
    
    # Test case 1: Adaptation WITHOUT experience buffer
    print("\n[Case 1] Adaptation without experience buffer...")
    ssm.set_params(original_params)
    adapted_no_buffer = maml.inner_update(new_support, steps=2, experience_buffer=None)
    
    # Evaluate on query set
    loss_no_buffer = 0.0
    for x, y_true in new_query:
        ssm.reset_state()
        y_pred = ssm.forward(x)
        loss_no_buffer += mse(y_pred, y_true)
    loss_no_buffer /= len(new_query)
    
    # Test case 2: Adaptation WITH experience buffer
    print("[Case 2] Adaptation with experience buffer...")
    ssm.set_params(original_params)
    adapted_with_buffer = maml.inner_update(
        new_support, 
        steps=2, 
        experience_buffer=buffer, 
        experience_batch_size=15
    )
    
    # Evaluate on query set
    loss_with_buffer = 0.0
    for x, y_true in new_query:
        ssm.reset_state()
        y_pred = ssm.forward(x)
        loss_with_buffer += mse(y_pred, y_true)
    loss_with_buffer /= len(new_query)
    
    # Compare results
    print(f"\nResults:")
    print(f"  Query loss without buffer: {loss_no_buffer:6.4f}")
    print(f"  Query loss with buffer:    {loss_with_buffer:6.4f}")
    improvement = loss_no_buffer - loss_with_buffer
    print(f"  Improvement from buffer:   {improvement:6.4f}")
    
    if improvement > 0:
        print("  -> Experience buffer helped adaptation! \u2713")
    else:
        print("  -> Experience buffer didn't help in this case.")


def run_mini_meta_training():
    """Demonstrate a mini meta-training loop."""
    print("\n=== Mini Meta-Training Demo ===")
    
    # Setup
    ssm = LightweightSSM(input_dim=1, hidden_dim=6, output_dim=1)
    maml = MinimalMAML(model=ssm, inner_lr=0.02, outer_lr=0.005)
    buffer = ExperienceBuffer(max_size=100)
    
    print("Running 5 episodes of meta-training...")
    
    for episode in range(5):
        # Generate task batch
        task_batch = []
        for _ in range(3):  # 3 tasks per batch
            task_type = np.random.choice(['sine', 'linear'])
            support, query = generate_simple_task(task_type)
            task_batch.append({'support': support, 'query': query})
            
            # Add to experience buffer
            buffer.add(support + query)
        
        # Meta update
        maml.meta_update(task_batch)
        
        print(f"  Episode {episode + 1}/5 completed. Buffer size: {len(buffer)}")
    
    print("\nMini meta-training completed!")
    print(f"Final buffer contains {len(buffer)} experiences from training.")
    
    return ssm, maml, buffer


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("LowNoCompute-AI-Baseline: Basic Usage Examples")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all demonstrations
    demonstrate_ssm_basic()
    demonstrate_experience_buffer()
    demonstrate_maml_basic()
    demonstrate_experience_enhanced_adaptation()
    
    # Run mini meta-training and final test
    ssm, maml, buffer = run_mini_meta_training()
    
    print("\n" + "=" * 60)
    print("All basic usage examples completed successfully!")
    print("\nTo see the full meta-training + test-time adaptation demo,")
    print("run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()