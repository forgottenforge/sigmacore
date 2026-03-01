#!/usr/bin/env python3
"""Test fixed Grover implementation"""
import sys, os
sys.path.append('.')

try:
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    import numpy as np
    
    # Test correct Grover with CZ
    circuit = Circuit()
    circuit.h(0)
    circuit.h(1)
    
    # Oracle with CZ
    circuit.cz(0, 1)
    
    # Diffusion with CZ
    circuit.h(0)
    circuit.h(1)
    circuit.x(0)
    circuit.x(1)
    circuit.cz(0, 1)
    circuit.x(0)
    circuit.x(1)
    circuit.h(0)
    circuit.h(1)
    
    device = LocalSimulator("braket_dm")
    task = device.run(circuit, shots=100)
    counts = task.result().measurement_counts
    
    print("Fixed Grover with CZ:")
    print(f"Counts: {counts}")
    success_rate = counts.get('11', 0) / 100
    print(f"Success rate: {success_rate:.2f}")
    print(f"Expected: ~1.00 for ideal Grover")
    
except ImportError:
    print("Braket not installed")
