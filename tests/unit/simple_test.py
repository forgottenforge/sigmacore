#!/usr/bin/env python3
import sys, os
sys.path.append('.')
from sigma_c.adapters.quantum import QuantumAdapter
import numpy as np

adapter = QuantumAdapter()

# Direct test of simulation
eps_values = [0.0, 0.05, 0.1]
print("Direct simulation test:")
for eps in eps_values:
    base_success = 0.95
    degradation_rate = 8.0
    success_prob = base_success * np.exp(-degradation_rate * eps)
    success_prob = max(0.05, min(0.95, success_prob))
    print(f"eps={eps:.2f} -> success_prob={success_prob:.4f}")
