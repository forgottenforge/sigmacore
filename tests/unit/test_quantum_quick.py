#!/usr/bin/env python3
"""Quick test of quantum optimizer performance"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c.adapters.quantum import QuantumAdapter
from sigma_c.optimization.quantum import BalancedQuantumOptimizer

adapter = QuantumAdapter()

def grover_factory(epsilon=0.0, idle_frac=0.0):
    return adapter.create_grover_with_noise(epsilon=epsilon, idle_frac=idle_frac)

optimizer = BalancedQuantumOptimizer(adapter, 0.5, 0.5)

print("Testing quantum optimizer...")
result = optimizer.optimize_circuit(
    grover_factory,
    param_space={
        'epsilon': [0.0, 0.05],
        'idle_frac': [0.0, 0.1]
    }
)

print(f"\nPerformance: {result.performance_after:.4f}")
print(f"Sigma_c: {result.sigma_c_after:.4f}")
print(f"Score: {result.score:.4f}")
print(f"Optimal params: {result.optimal_params}")
