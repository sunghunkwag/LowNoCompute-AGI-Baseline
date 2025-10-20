"""Dummy Environment for LowNoCompute-AI-Baseline.

This module provides a simple environment interface for testing and development.
It can be extended for more complex RL or simulation environments.
"""

import numpy as np
from typing import Tuple, Optional


class DummyEnvironment:
    """A simple dummy environment for testing the neural network baseline.
    
    This environment generates random observations and rewards,
    useful for testing without complex dependencies.
    
    Attributes:
        observation_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        max_steps (int): Maximum steps per episode
        current_step (int): Current step counter
    """
    
    def __init__(self, observation_dim: int = 10, action_dim: int = 3, max_steps: int = 100):
        """Initialize the dummy environment.
        
        Args:
            observation_dim: Size of observation vector
            action_dim: Number of possible actions
            max_steps: Maximum steps before episode termination
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self._state = None
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self._state = np.random.randn(self.observation_dim)
        return self._state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (index)
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        # Generate random next state
        self._state = np.random.randn(self.observation_dim)
        
        # Generate random reward (simulate task)
        reward = np.random.randn()
        
        # Increment step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'step': self.current_step,
            'action': action
        }
        
        return self._state.copy(), reward, done, info
    
    def render(self) -> None:
        """Render the environment (dummy implementation)."""
        if self._state is not None:
            print(f"Step {self.current_step}: State = {self._state[:3]}...")  # Show first 3 dims
    
    def close(self) -> None:
        """Clean up resources."""
        self._state = None
        self.current_step = 0


if __name__ == "__main__":
    # Simple test
    env = DummyEnvironment(observation_dim=10, action_dim=3)
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for _ in range(5):
        action = np.random.randint(0, env.action_dim)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    
    env.close()
    print("Environment test completed!")
