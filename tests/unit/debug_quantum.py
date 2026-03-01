#!/usr/bin/env python3
"""Debug quantum performance calculation"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from sigma_c.adapters.quantum import QuantumAdapter
import numpy as np

adapter = QuantumAdapter()

# Test simulation with different epsilon values
print("Testing quantum simulation fidelity:")
print("-" * 50)

for eps in [0.0, 0.05, 0.1, 0.15]:
    circuit = adapter.create_grover_with_noise(epsilon=eps, idle_frac=0.0)
    result = adapter.run_optimization(
        circuit_type='custom',
        epsilon_values=[eps],
        shots=100,
        custom_circuit=circuit
    )
    fidelity = result['observable'][0]
    print(f"Epsilon={eps:.2f}: Fidelity={fidelity:.4f}")

print("\nExpected: eps=0.0 should give ~0.95 fidelity")
