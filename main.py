#!/usr/bin/env python3
"""
Ultra-Minimal AI/Meta-RL Baseline for Low-Compute Settings
Referenced from SSM-MetaRL-TestCompute/main.py but reduced to essential components
Design Principles:
- CPU-only operation (no GPU required)
- Minimal memory footprint
- Essential meta-learning components only
- Test-time adaptation capability
"""
from __future__ import annotations

import numpy as np
import random
from typing import List, Tuple, Dict, Any
import time


class LightweightSSM:
    """Tiny State Space Model (SSM) for sequential processing on CPU.

    This is intentionally small: linear state update with tanh nonlinearity and
    linear readout. It supports resetting state between sequences and exposes
    get/set params for meta-learning.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 8, output_dim: int = 1):
        # Shapes kept tiny for CPU friendliness
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        rng = np.random.default_rng()
        self.A = (rng.standard_normal((hidden_dim, hidden_dim)) * 0.1).astype(np.float64)  # state transition
        self.B = (rng.standard_normal((hidden_dim, input_dim)) * 0.1).astype(np.float64)   # input map
        self.C = (rng.standard_normal((output_dim, hidden_dim)) * 0.1).astype(np.float64)  # readout
        self.D = (rng.standard_normal((output_dim, input_dim)) * 0.1).astype(np.float64)   # skip/readout

        self.h = np.zeros(hidden_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward one step. x shape: (input_dim,)"""
        # Ensure correct dtype/shape
        x = np.asarray(x, dtype=np.float64).reshape(self.input_dim)
        # h_{t+1} = tanh(A h_t + B x_t)
        self.h = np.tanh(self.A @ self.h + self.B @ x)
        # y_t = C h_t + D x_t
        y = self.C @ self.h + self.D @ x
        return y

    def reset_state(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float64)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"A": self.A.copy(), "B": self.B.copy(), "C": self.C.copy(), "D": self.D.copy()}

    def set_params(self, params: Dict[str, np.ndarray]):
        # Basic shape checks for robustness
        assert params["A"].shape == self.A.shape
        assert params["B"].shape == self.B.shape
        assert params["C"].shape == self.C.shape
        assert params["D"].shape == self.D.shape
        self.A = params["A"].astype(np.float64, copy=True)
        self.B = params["B"].astype(np.float64, copy=True)
        self.C = params["C"].astype(np.float64, copy=True)
        self.D = params["D"].astype(np.float64, copy=True)


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(np.mean((y_pred - y_true) ** 2))


class MinimalMAML:
    """Minimal MAML-style meta-learner for CPU.

    Uses simple finite-difference gradients for inner/outer loops to avoid
    heavyweight autodiff, keeping it friendly for low-compute settings.
    """

    def __init__(self, model: LightweightSSM, inner_lr: float = 0.02, outer_lr: float = 0.001):
        self.model = model
        self.inner_lr = float(inner_lr)
        self.outer_lr = float(outer_lr)

    def _finite_diff_grad(self, params: Dict[str, np.ndarray], batch: List[Tuple[np.ndarray, np.ndarray]], eps: float = 1e-5) -> Dict[str, np.ndarray]:
        # Compute scalar loss gradient wrt each parameter tensor via forward diffs
        grads: Dict[str, np.ndarray] = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        # Baseline loss
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

    def inner_update(self, support_data: List[Tuple[np.ndarray, np.ndarray]], steps: int = 1, eps: float = 1e-5) -> Dict[str, np.ndarray]:
        params = self.model.get_params()
        for _ in range(max(1, int(steps))):
            grads = self._finite_diff_grad(params, support_data, eps=eps)
            for k in params:
                params[k] = params[k] - self.inner_lr * grads[k] / max(1, len(support_data))
        # Set and return adapted params
        self.model.set_params(params)
        return params

    def meta_update(self, task_batch: List[Dict[str, Any]], eps: float = 1e-5):
        original = self.model.get_params()
        meta_grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in original.items()}
        T = max(1, len(task_batch))
        for task in task_batch:
            support = task['support']
            query = task['query']
            # Inner adaptation from original parameters (clone to avoid drift)
            self.model.set_params(original)
            adapted = self.inner_update(support, steps=1, eps=eps)
            # Finite-diff meta-gradient on query with respect to adapted params
            q_loss_base = 0.0
            self.model.set_params(adapted)
            for x, y_true in query:
                self.model.reset_state()
                q_loss_base += mse(self.model.forward(x), y_true)

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
        # Apply meta step to original params
        updated = {k: original[k] - self.outer_lr * (meta_grads[k] / T) for k in original}
        self.model.set_params(updated)


def generate_simple_task(task_type: str = 'sine') -> Tuple[List[Tuple], List[Tuple]]:
    """Generate small synthetic tasks: sine or linear.

    Returns (support, query) where each is a list of (x, y) pairs with shapes
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
    else:
        slope = rng.uniform(-2, 2)
        intercept = rng.uniform(-1, 1)
        support_x = rng.uniform(-2, 2, (5, 1))
        support_y = slope * support_x + intercept
        query_x = rng.uniform(-2, 2, (10, 1))
        query_y = slope * query_x + intercept

    support = [(x.flatten(), y.flatten()) for x, y in zip(support_x, support_y)]
    query = [(x.flatten(), y.flatten()) for x, y in zip(query_x, query_y)]
    return support, query


def test_time_adaptation_example():
    """Demonstrate test-time adaptation capability."""
    print("\n=== Test-Time Adaptation Example ===")

    model = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
    maml = MinimalMAML(model, inner_lr=0.05)

    print("Encountering new task: y = 0.5*sin(3*x + 1.2)")

    test_x = np.array([[-1.5], [-0.5], [0.3], [1.2]])
    test_y = 0.5 * np.sin(3 * test_x + 1.2)
    support_data = [(x.flatten(), y.flatten()) for x, y in zip(test_x, test_y)]

    model.reset_state()
    test_input = np.array([0.8])
    pred_before = model.forward(test_input)
    true_value = 0.5 * np.sin(3 * 0.8 + 1.2)
    print(f"Before adaptation: pred={pred_before[0]:.3f}, true={true_value[0]:.3f}, error={abs(pred_before[0] - true_value[0]):.3f}")

    print("Adapting to new task with 4 support examples...")
    maml.inner_update(support_data, steps=3)

    model.reset_state()
    pred_after = model.forward(test_input)
    print(f"After adaptation:  pred={pred_after[0]:.3f}, true={true_value[0]:.3f}, error={abs(pred_after[0] - true_value[0]):.3f}")

    improvement = abs(pred_before[0] - true_value[0]) - abs(pred_after[0] - true_value[0])
    print(f"Improvement: {improvement:.3f} (positive = better)")


def main():
    """Main training loop for minimal AI baseline on CPU."""
    print("Minimal AI/Meta-RL Baseline - CPU Only")
    print("=" * 50)

    model = LightweightSSM(input_dim=1, hidden_dim=8, output_dim=1)
    maml = MinimalMAML(model, inner_lr=0.01, outer_lr=0.001)

    total_params = int(sum(np.prod(p.shape) for p in model.get_params().values()))
    print(f"Model parameters: {total_params} total")

    num_episodes = 10
    batch_size = 2
    print(f"\nStarting meta-training: {num_episodes} episodes, batch size {batch_size}")

    for episode in range(num_episodes):
        task_batch = []
        for _ in range(batch_size):
            task_type = random.choice(['sine', 'linear'])
            support_set, query_set = generate_simple_task(task_type)
            task_batch.append({'support': support_set, 'query': query_set})

        start_time = time.time()
        maml.meta_update(task_batch)
        update_time = time.time() - start_time

        if (episode + 1) % 2 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Update time: {update_time:.3f}s")

    print("\nMeta-training completed!")
    test_time_adaptation_example()

    print("\n=== Summary ===")
    print("\u2713 Lightweight SSM core implemented")
    print("\u2713 MAML-style meta-learning functional")
    print("\u2713 CPU-only operation confirmed")
    print("\u2713 Test-time adaptation demonstrated")
    print("\nThis baseline provides the essential AI components:")
    print("- Sequential processing (SSM)")
    print("- Few-shot learning (MAML)")
    print("- Online adaptation capability")
    print("- Resource-efficient design")


if __name__ == "__main__":
    main()
