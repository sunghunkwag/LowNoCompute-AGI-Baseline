"""Basic usage example for LowNoCompute-AI-Baseline.

This example demonstrates how to use the neural network baseline
implemented in main.py for a simple training task.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import (
    initialize_weights,
    forward_pass,
    compute_loss,
    train_step
)


def generate_dummy_data(n_samples=100, input_dim=10, output_dim=3):
    """Generate dummy training data.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Input dimension
        output_dim: Output dimension
    
    Returns:
        X: Input data of shape (n_samples, input_dim)
        y: Target data of shape (n_samples, output_dim)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randn(n_samples, output_dim)
    return X, y


def train_model(X, y, input_size, hidden_size, output_size, 
                learning_rate=0.01, epochs=100, verbose=True):
    """Train the neural network model.
    
    Args:
        X: Training input data
        y: Training target data
        input_size: Input layer size
        hidden_size: Hidden layer size
        output_size: Output layer size
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        verbose: Whether to print progress
    
    Returns:
        weights: Trained model weights
        loss_history: List of loss values during training
    """
    # Initialize weights
    weights = initialize_weights(input_size, hidden_size, output_size)
    
    loss_history = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    # Training loop
    for epoch in range(epochs):
        # Perform one training step
        loss = train_step(X, y, weights, learning_rate)
        loss_history.append(loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    print("\nTraining completed!")
    return weights, loss_history


def evaluate_model(X, y, weights):
    """Evaluate the trained model.
    
    Args:
        X: Test input data
        y: Test target data
        weights: Trained model weights
    
    Returns:
        loss: Test loss
        predictions: Model predictions
    """
    predictions, _ = forward_pass(X, weights)
    loss = compute_loss(predictions, y)
    return loss, predictions


def main():
    """Main function to run the basic usage example."""
    print("=" * 60)
    print("LowNoCompute-AI-Baseline: Basic Usage Example")
    print("=" * 60)
    print()
    
    # Configuration
    n_samples = 100
    input_dim = 10
    hidden_dim = 5
    output_dim = 3
    learning_rate = 0.01
    epochs = 100
    
    # Generate training data
    print("1. Generating dummy training data...")
    X_train, y_train = generate_dummy_data(n_samples, input_dim, output_dim)
    print(f"   Training data shape: X={X_train.shape}, y={y_train.shape}\n")
    
    # Train the model
    print("2. Training the neural network...")
    weights, loss_history = train_model(
        X_train, y_train,
        input_size=input_dim,
        hidden_size=hidden_dim,
        output_size=output_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=True
    )
    
    # Evaluate the model
    print("\n3. Evaluating the trained model...")
    test_loss, predictions = evaluate_model(X_train, y_train, weights)
    print(f"   Final test loss: {test_loss:.6f}")
    print(f"   Predictions shape: {predictions.shape}")
    
    # Show some statistics
    print("\n4. Training statistics:")
    print(f"   Initial loss: {loss_history[0]:.6f}")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    print(f"   Loss reduction: {(loss_history[0] - loss_history[-1]):.6f}")
    print(f"   Loss reduction %: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
