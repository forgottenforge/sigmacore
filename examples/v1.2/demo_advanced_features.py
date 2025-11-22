"""
Demo: Advanced Features v1.2.3
==============================
Verifies Callbacks, Serialization, Visualization, and CLI concepts.
"""

import os
import sys
import numpy as np
from sigma_c.optimization.universal import UniversalOptimizer
from sigma_c.core.callbacks import LoggingCallback, EarlyStopping, CheckpointCallback
from sigma_c.visualization import plot_convergence, plot_landscape, plot_pareto_frontier

# 1. Define a Mock System and Optimizer
class SimpleSystem:
    def __init__(self):
        self.state = 0.0

class SimpleOptimizer(UniversalOptimizer):
    def _evaluate_performance(self, system, params):
        # Simple quadratic function: -(x-2)^2 + 10
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        return -(x - 2)**2 - (y - 3)**2 + 10

    def _evaluate_stability(self, system, params):
        # Stability decreases as we move away from 0
        x = params.get('x', 0.0)
        return 1.0 / (1.0 + abs(x))

    def _apply_params(self, system, params):
        return system

def run_demo():
    print("=== Sigma-C v1.2.3 Advanced Features Demo ===")
    
    # Setup
    system = SimpleSystem()
    optimizer = SimpleOptimizer(performance_weight=0.8, stability_weight=0.2)
    
    param_space = {
        'x': np.linspace(0, 5, 20).tolist(),
        'y': np.linspace(0, 5, 20).tolist()
    }
    
    # 1. Test Callbacks
    print("\n[1] Testing Callbacks...")
    callbacks = [
        LoggingCallback(interval=50),
        EarlyStopping(monitor='score', patience=10),
        CheckpointCallback('demo_checkpoint', interval=100)
    ]
    
    result = optimizer.optimize(
        system, 
        param_space, 
        strategy='brute_force',
        callbacks=callbacks
    )
    
    print(f"Optimization finished. Best Score: {result.score:.4f}")
    print(f"Optimal Params: {result.optimal_params}")
    
    # 2. Test Serialization
    print("\n[2] Testing Serialization...")
    optimizer.save("demo_optimizer_state.json")
    print("Saved state.")
    
    new_optimizer = SimpleOptimizer()
    new_optimizer.load("demo_optimizer_state.json")
    print(f"Loaded state. History length: {len(new_optimizer.history)}")
    assert len(new_optimizer.history) == len(optimizer.history)
    
    # 3. Test Visualization
    print("\n[3] Testing Visualization...")
    plot_convergence(result, save_path="demo_convergence.png")
    plot_pareto_frontier(result, save_path="demo_pareto.png")
    plot_landscape(optimizer, param_space, system, save_path="demo_landscape.png")
    print("Saved plots: demo_convergence.png, demo_pareto.png, demo_landscape.png")
    
    print("\n=== Demo Complete: SUCCESS ===")

if __name__ == "__main__":
    run_demo()
