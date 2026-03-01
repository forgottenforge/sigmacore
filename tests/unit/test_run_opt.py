#!/usr/bin/env python3
import sys, os
sys.path.append('.')
from sigma_c.adapters.quantum import QuantumAdapter

adapter = QuantumAdapter()

# Test run_optimization directly
circuit = adapter.create_grover_with_noise(epsilon=0.0, idle_frac=0.0)

print("Testing run_optimization with custom circuit (eps=0.0 in params):")
result = adapter.run_optimization(
    circuit_type='custom',
    epsilon_values=[0.0],
    shots=100,
    custom_circuit=circuit
)

print(f"Result keys: {result.keys()}")
print(f"Observable: {result['observable']}")
print(f"Sigma_c: {result['sigma_c']}")
print(f"Raw counts: {result['raw_counts']}")
