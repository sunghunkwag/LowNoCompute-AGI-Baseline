"""Unit tests for the main module.

This module contains unit tests for the neural network components
in the main.py file.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from main import (
        initialize_weights,
        forward_pass,
        compute_loss,
        backward_pass,
        update_weights,
        train_step
    )
except ImportError as e:
    print(f"Warning: Could not import from main: {e}")
    print("Some tests may be skipped.")


class TestNeuralNetworkComponents(unittest.TestCase):
    """Test cases for neural network components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.input_size = 10
        self.hidden_size = 5
        self.output_size = 3
        self.batch_size = 4
    
    def test_weight_initialization(self):
        """Test weight initialization function."""
        try:
            weights = initialize_weights(self.input_size, self.hidden_size, self.output_size)
            
            # Check that weights dictionary contains expected keys
            self.assertIn('W1', weights)
            self.assertIn('b1', weights)
            self.assertIn('W2', weights)
            self.assertIn('b2', weights)
            
            # Check shapes
            self.assertEqual(weights['W1'].shape, (self.input_size, self.hidden_size))
            self.assertEqual(weights['b1'].shape, (self.hidden_size,))
            self.assertEqual(weights['W2'].shape, (self.hidden_size, self.output_size))
            self.assertEqual(weights['b2'].shape, (self.output_size,))
            
            # Check that weights are not all zeros
            self.assertFalse(np.allclose(weights['W1'], 0))
            self.assertFalse(np.allclose(weights['W2'], 0))
            
        except NameError:
            self.skipTest("initialize_weights not available")
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        try:
            weights = initialize_weights(self.input_size, self.hidden_size, self.output_size)
            X = np.random.randn(self.batch_size, self.input_size)
            
            output, cache = forward_pass(X, weights)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, self.output_size))
            
            # Check that cache contains necessary components
            self.assertIsInstance(cache, dict)
            
            # Check output values are finite
            self.assertTrue(np.all(np.isfinite(output)))
            
        except NameError:
            self.skipTest("forward_pass not available")
    
    def test_loss_computation(self):
        """Test loss computation."""
        try:
            predictions = np.random.randn(self.batch_size, self.output_size)
            targets = np.random.randn(self.batch_size, self.output_size)
            
            loss = compute_loss(predictions, targets)
            
            # Check that loss is a scalar
            self.assertIsInstance(loss, (float, np.floating))
            
            # Check that loss is non-negative
            self.assertGreaterEqual(loss, 0)
            
            # Check that identical predictions and targets give zero loss
            zero_loss = compute_loss(predictions, predictions)
            self.assertAlmostEqual(zero_loss, 0, places=6)
            
        except NameError:
            self.skipTest("compute_loss not available")
    
    def test_backward_pass(self):
        """Test backward pass gradient computation."""
        try:
            weights = initialize_weights(self.input_size, self.hidden_size, self.output_size)
            X = np.random.randn(self.batch_size, self.input_size)
            y = np.random.randn(self.batch_size, self.output_size)
            
            output, cache = forward_pass(X, weights)
            gradients = backward_pass(output, y, cache, weights)
            
            # Check that gradients dictionary contains expected keys
            self.assertIn('dW1', gradients)
            self.assertIn('db1', gradients)
            self.assertIn('dW2', gradients)
            self.assertIn('db2', gradients)
            
            # Check gradient shapes match weight shapes
            self.assertEqual(gradients['dW1'].shape, weights['W1'].shape)
            self.assertEqual(gradients['db1'].shape, weights['b1'].shape)
            self.assertEqual(gradients['dW2'].shape, weights['W2'].shape)
            self.assertEqual(gradients['db2'].shape, weights['b2'].shape)
            
            # Check that gradients are finite
            for grad in gradients.values():
                self.assertTrue(np.all(np.isfinite(grad)))
            
        except NameError:
            self.skipTest("backward_pass not available")
    
    def test_weight_update(self):
        """Test weight update function."""
        try:
            weights = initialize_weights(self.input_size, self.hidden_size, self.output_size)
            initial_weights = {k: v.copy() for k, v in weights.items()}
            
            # Create dummy gradients
            gradients = {k: np.random.randn(*v.shape) for k, v in weights.items()}
            gradients = {k.replace('W', 'dW').replace('b', 'db'): v for k, v in gradients.items()}
            
            learning_rate = 0.01
            update_weights(weights, gradients, learning_rate)
            
            # Check that weights have changed
            for key in weights.keys():
                self.assertFalse(np.allclose(weights[key], initial_weights[key]))
            
        except NameError:
            self.skipTest("update_weights not available")
    
    def test_train_step(self):
        """Test complete training step."""
        try:
            weights = initialize_weights(self.input_size, self.hidden_size, self.output_size)
            X = np.random.randn(self.batch_size, self.input_size)
            y = np.random.randn(self.batch_size, self.output_size)
            
            loss = train_step(X, y, weights, learning_rate=0.01)
            
            # Check that loss is returned
            self.assertIsInstance(loss, (float, np.floating))
            self.assertGreaterEqual(loss, 0)
            
        except NameError:
            self.skipTest("train_step not available")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of implementations."""
    
    def test_large_values(self):
        """Test handling of large values."""
        try:
            # Test with large input values
            weights = initialize_weights(10, 5, 3)
            X = np.random.randn(4, 10) * 100  # Large values
            
            output, _ = forward_pass(X, weights)
            
            # Check for NaN or Inf
            self.assertTrue(np.all(np.isfinite(output)))
            
        except NameError:
            self.skipTest("Required functions not available")
    
    def test_zero_input(self):
        """Test handling of zero input."""
        try:
            weights = initialize_weights(10, 5, 3)
            X = np.zeros((4, 10))
            
            output, _ = forward_pass(X, weights)
            
            # Should still produce finite output
            self.assertTrue(np.all(np.isfinite(output)))
            
        except NameError:
            self.skipTest("Required functions not available")


if __name__ == '__main__':
    print("Running unit tests for main.py...")
    unittest.main(verbosity=2)
