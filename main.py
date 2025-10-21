#!/usr/bin/env python3
"""
Minimal AI/Meta-RL Baseline with Experience-Based Reasoning
- CPU-only, tiny memory footprint
- Minimal meta-learning with test-time adaptation
- Adds ExperienceBuffer for experience-based reasoning at test time

Note: This implementation uses finite difference gradients for simplicity and 
CPU compatibility. For production use, consider JAX-based auto-differentiation
for significantly improved performance.
"""
from __future__ import annotations

import time
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from collections import deque


# ===============================================================
# Lightweight State Space Model (SSM)
# ===============================================================
class LightweightSSM:
    """Tiny State Space Model (SSM) for sequential processing on CPU.

    Design goals: extremely small, deterministic CPU-friendly math, and simple
    get/set parameter API to support meta-learning style updates.

    Dynamics:
      h_{t+1} = tanh(A h_t + B x_t)
      y_t     = C h_t + D x_t
    
    This implementation prioritizes simplicity and CPU efficiency over speed.
    For performance-critical applications, consider JAX-based implementations
    with auto-differentiation.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 8, output_dim: int = 1):
        # Shapes kept tiny for CPU friendliness
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        rng = np.random.default_rng()
        # Parameters kept as float64 for numerical stability in finite differences
        self.A = (rng.standard_normal((hidden_dim, hidden_dim)) * 0.1).astype(np.float64)  # state transition
        self.B = (rng.standard_normal((hidden_dim, input_dim)) * 0.1).astype(np.float64)   # input map
        self.C = (rng.standard_normal((output_dim, hidden_dim)) * 0.1).astype(np.float64)  # readout
        self.D = (rng.standard_normal((output_dim, input_dim)) * 0.1).astype(np.float64)   # skip/readout

        self.h = np.zeros(hidden_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward one step. x shape: (input_dim,)"""
        x = np.asarray(x, dtype=np.float64).reshape(self.input_dim)
        self.h = np.tanh(self.A @ self.h + self.B @ x)
        y = self.C @ self.h + self.D @ x
        return y

    def reset_state(self) -> None:
        """Reset hidden state to zeros. Call before processing new sequences."""
        self.h = np.zeros(self.hidden_dim, dtype=np.float64)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get a copy of all model parameters."""
        return {"A": self.A.copy(), "B": self.B.copy(), "C": self.C.copy(), "D": self.D.copy()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters with shape validation."""
        # Defensive shape checks
        assert params["A"].shape == self.A.shape, f"A shape mismatch: expected {self.A.shape}, got {params['A'].shape}"
        assert params["B"].shape == self.B.shape, f"B shape mismatch: expected {self.B.shape}, got {params['B'].shape}"
        assert params["C"].shape == self.C.shape, f"C shape mismatch: expected {self.C.shape}, got {params['C'].shape}"
        assert params["D"].shape == self.D.shape, f"D shape mismatch: expected {self.D.shape}, got {params['D'].shape}"
        
        self.A = params["A"].astype(np.float64, copy=True)
        self.B = params["B"].astype(np.float64, copy=True)
        self.C = params["C"].astype(np.float64, copy=True)
        self.D = params["D"].astype(np.float64, copy=True)


# ===============================================================
# Loss Function
# ===============================================================

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean squared error between predictions and targets."""
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(np.mean((y_pred - y_true) ** 2))


# ===============================================================
# Experience-Based Reasoning: ExperienceBuffer
# ===============================================================
class ExperienceBuffer:
    """A simple circular memory buffer to store past experiences (data samples).
    
    This buffer enables experience-based reasoning by storing and retrieving
    relevant past experiences during test-time adaptation.

    - add(batch): appends a batch of (x, y) samples
    - get_batch(k): randomly samples up to k past samples
    """

    def __init__(self, max_size: int = 100):
        """Initialize experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store. When full,
                     oldest experiences are automatically removed.
        """
        # deque with maxlen automatically drops oldest items when full
        self.buffer = deque(maxlen=max_size)

    def add(self, experience_batch: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Add a batch of experiences to the buffer.
        
        Args:
            experience_batch: List of (input, output) tuples to store.
        """
        if not experience_batch:
            return
        self.buffer.extend(experience_batch)

    def get_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get a random batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            List of (input, output) tuples. If batch_size exceeds the current 
            buffer length, returns as many as available.
        """
        if len(self.buffer) == 0:
            return []
        actual = min(max(0, int(batch_size)), len(self.buffer))
        return random.sample(list(self.buffer), actual)

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)


# ===============================================================
# Minimal MAML with optional ExperienceBuffer for inner updates
# ===============================================================
class MinimalMAML:
    """Minimal MAML-style meta-learner with experience buffer support.
    
    This implementation uses finite difference gradients for simplicity and
    CPU compatibility. While slower than auto-differentiation, it's more
    stable and doesn't require additional dependencies.
    
    Key innovation: Optional integration with ExperienceBuffer for 
    experience-enhanced adaptation during test time.
    """

    def __init__(self, model: LightweightSSM, inner_lr: float = 0.02, outer_lr: float = 0.001):
        """Initialize MinimalMAML.
        
        Args:
            model: LightweightSSM to train
            inner_lr: Learning rate for inner loop (fast adaptation)
            outer_lr: Learning rate for outer loop (meta-learning)
        """
        self.model = model
        self.inner_lr = float(inner_lr)
        self.outer_lr = float(outer_lr)

    def _finite_diff_grad(
        self,
        params: Dict[str, np.ndarray],
        batch: List[Tuple[np.ndarray, np.ndarray]],
        eps: float = 1e-5,
    ) -> Dict[str, np.ndarray]:
        """Compute finite-difference gradients of loss over the provided batch.
        
        Note: This is O(num_parameters) which can be slow for large models.
        For performance-critical applications, consider JAX auto-differentiation.
        """
        grads: Dict[str, np.ndarray] = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.model.set_params(params)
        base_loss = 0.0
        for x, y_true in batch:
            self.model.reset_state()
            y_pred = self.model.forward(x)
            base_loss += mse(y_pred, y_true)

        for k in params:
            w = params[k]
            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig = w[idx]
                w[idx] = orig + eps
                self.model.set_params(params)
                loss_eps = 0.0
                for x, y_true in batch:
                    self.model.reset_state()
                    y_pred = self.model.forward(x)
                    loss_eps += mse(y_pred, y_true)
                grads[k][idx] = (loss_eps - base_loss) / eps
                w[idx] = orig
                it.iternext()
        return grads

    def inner_update(
        self,
        support_data: List[Tuple[np.ndarray, np.ndarray]],
        steps: int = 1,
        eps: float = 1e-5,
        experience_buffer: Optional[ExperienceBuffer] = None,
        experience_batch_size: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Perform inner-loop adaptation (fast adaptation).

        If an ExperienceBuffer is provided and non-empty, its sampled experiences
        are concatenated with the provided support_data to form the adaptation batch.
        This is the core of experience-based reasoning.
        
        Args:
            support_data: List of (input, output) tuples for adaptation
            steps: Number of gradient descent steps
            eps: Epsilon for finite difference gradients
            experience_buffer: Optional buffer for experience retrieval
            experience_batch_size: Number of past experiences to retrieve
            
        Returns:
            Updated parameter dictionary
        """
        params = self.model.get_params()

        combined_support = list(support_data)
        if experience_buffer and len(experience_buffer) > 0:
            past = experience_buffer.get_batch(experience_batch_size)
            combined_support = combined_support + past

        n_steps = max(1, int(steps))
        for _ in range(n_steps):
            grads = self._finite_diff_grad(params, combined_support, eps=eps)
            for k in params:
                params[k] = params[k] - self.inner_lr * grads[k] / max(1, len(combined_support))

        self.model.set_params(params)
        return params

    def meta_update(self, task_batch: List[Dict[str, Any]], eps: float = 1e-5) -> None:
        """Perform outer-loop meta-update using finite differences.

        Note: For educational simplicity, this retains a basic MAML-like loop.
        During meta-training we do NOT use the experience buffer to avoid leakage.
        
        Args:
            task_batch: List of task dictionaries with 'support' and 'query' keys
            eps: Epsilon for finite difference gradients
        """
        original = self.model.get_params()
        meta_grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in original.items()}
        T = max(1, len(task_batch))

        for task in task_batch:
            support = task['support']
            query = task['query']

            # Adapt from the original parameters on support data only
            self.model.set_params(original)
            adapted = self.inner_update(support, steps=1, eps=eps, experience_buffer=None)

            # Evaluate query loss at adapted params
            q_loss_base = 0.0
            self.model.set_params(adapted)
            for x, y_true in query:
                self.model.reset_state()
                q_loss_base += mse(self.model.forward(x), y_true)

            # Finite-difference wrt adapted parameters for meta-gradient
            for k in adapted:
                w = adapted[k]
                it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig = w[idx]
                    w[idx] = orig + eps
                    self.model.set_params(adapted)
                    q_loss_eps = 0.0
                    for x, y_true in query:
                        self.model.reset_state()
                        q_loss_eps += mse(self.model.forward(x), y_true)
                    meta_grads[k][idx] += (q_loss_eps - q_loss_base) / eps
                    w[idx] = orig
                    it.iternext()

        updated = {k: original[k] - self.outer_lr * (meta_grads[k] / T) for k in original}
        self.model.set_params(updated)


# ===============================================================
# Task generation (tiny synthetic tasks)
# ===============================================================

def generate_simple_task(task_type: str = 'sine') -> Tuple[List[Tuple], List[Tuple]]:
    """Generate small synthetic tasks: sine or linear.

    Args:
        task_type: Either 'sine' for sine wave tasks or 'linear' for linear function tasks
        
    Returns:
        Tuple of (support, query) where each is a list of (x, y) pairs with shapes
        matching model input/output (1D).
    """
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32)

    if task_type == 'sine':
        phase = rng.uniform(0.0, 2 * np.pi)
        amplitude = rng.uniform(0.5, 2.0)
        frequency = rng.uniform(0.5, 2.0)
        support_x = rng.uniform(-2, 2, (5, 1))
        support_y = amplitude * np.sin(frequency * support_x + phase)
        query_x = rng.uniform(-2, 2, (10, 1))
        query_y = amplitude * np.sin(frequency * query_x + phase)
    else:  # linear
        slope = rng.uniform(-2, 2)
        intercept = rng.uniform(-1, 1)
        support_x = rng.uniform(-2, 2, (5, 1))
        support_y = slope * support_x + intercept
        query_x = rng.uniform(-2, 2, (10, 1))
        query_y = slope * query_x + intercept

    support = [(x.flatten(), y.flatten()) for x, y in zip(support_x, support_y)]
    query = [(x.flatten(), y.flatten()) for x, y in zip(query_x, query_y)]
    return support, query


# ===============================================================
# Test-time adaptation demo: with and without ExperienceBuffer
# ===============================================================

def test_time_adaptation_example(initial_params: Dict[str, np.ndarray], experience_buffer: ExperienceBuffer) -> None:
    """Demonstrate test-time adaptation with and without experience buffer.
    
    This function showcases the core benefit of experience-based reasoning:
    improved adaptation performance by leveraging past experiences.
    """
    print("\n=== Test-Time Adaptation Example ===")

    model = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
    maml = MinimalMAML(model, inner_lr=0.05)

    print("A new task is encountered: y = 0.5*sin(3*x + 1.2)")
    print(f"The agent has access to an experience buffer with {len(experience_buffer)} past data points.")

    test_x = np.array([[-1.5], [-0.5], [0.3], [1.2]])
    test_y = 0.5 * np.sin(3 * test_x + 1.2)
    support_data = [(x.flatten(), y.flatten()) for x, y in zip(test_x, test_y)]

    test_input = np.array([0.8])
    true_value = float(0.5 * np.sin(3 * 0.8 + 1.2))

    # --- Case 1: Adaptation WITHOUT experience buffer ---
    print("\n[Case 1] Adapting with only 4 new examples...")
    model.set_params(initial_params)  # reset to meta-learned params
    model.reset_state()
    pred_before = model.forward(test_input)

    maml.inner_update(support_data, steps=3, experience_buffer=None)  # no buffer

    model.reset_state()
    pred_after_no_buffer = model.forward(test_input)
    error_no_buffer = abs(pred_after_no_buffer[0] - true_value)
    print(f"  - Before adaptation error: {abs(pred_before[0] - true_value):.4f}")
    print(f"  - After adaptation error:  {error_no_buffer:.4f}")

    # --- Case 2: Adaptation WITH experience buffer ---
    print("\n[Case 2] Adapting with 4 new examples + 10 past experiences...")
    model.set_params(initial_params)  # reset to meta-learned params

    maml.inner_update(support_data, steps=3, experience_buffer=experience_buffer, experience_batch_size=10)

    model.reset_state()
    pred_after_with_buffer = model.forward(test_input)
    error_with_buffer = abs(pred_after_with_buffer[0] - true_value)
    print(f"  - Before adaptation error: {abs(pred_before[0] - true_value):.4f}")
    print(f"  - After adaptation error:  {error_with_buffer:.4f}")

    print("\n--- Comparison ---")
    improvement = error_no_buffer - error_with_buffer
    print(f"Error reduction by using experience buffer: {improvement:.4f}")
    if improvement > 0:
        print("Result: Experience-based reasoning led to a more accurate adaptation. ✓")
    else:
        print("Result: Experience-based reasoning did not improve adaptation in this case.")


# ===============================================================
# Main: meta-train then demonstrate test-time adaptation with buffer
# ===============================================================

def main() -> None:
    """Main demonstration of experience-based reasoning framework.
    
    This function runs the complete pipeline:
    1. Meta-training on synthetic tasks
    2. Experience buffer accumulation
    3. Test-time adaptation comparison
    """
    print("Minimal AI/Meta-RL Baseline with Experience-Based Reasoning")
    print("=" * 60)

    # 1) Meta-train to acquire fast adaptation capability
    model = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
    maml = MinimalMAML(model, inner_lr=0.01, outer_lr=0.001)

    num_episodes = 10
    batch_size = 4
    experience_buffer = ExperienceBuffer(max_size=200)

    print(f"Starting meta-training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        task_batch = []
        for _ in range(batch_size):
            task_type = random.choice(['sine', 'linear'])
            support_set, query_set = generate_simple_task(task_type)
            task_batch.append({'support': support_set, 'query': query_set})

            # Store seen data as experiences (memory) for test-time reasoning later
            experience_buffer.add(support_set)
            experience_buffer.add(query_set)

        maml.meta_update(task_batch)
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. Buffer size: {len(experience_buffer)}")

    print("\nMeta-training completed!")
    meta_learned_params = model.get_params()

    # 2) Demonstrate test-time adaptation with and without experience buffer
    test_time_adaptation_example(meta_learned_params, experience_buffer)


if __name__ == "__main__":
    main()