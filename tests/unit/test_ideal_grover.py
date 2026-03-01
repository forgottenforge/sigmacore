#!/usr/bin/env python3
"""Test ideal Grover circuit without any noise"""
import sys, os
sys.path.append('.')

try:
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    import numpy as np
    
    # Create ideal Grover circuit manually
    circuit = Circuit()
    
    # Initial Hadamards
    circuit.h(0)
    circuit.h(1)
    
    # Oracle (mark |11>)
    circuit.cnot(0, 1)
    circuit.rz(1, np.pi)
    circuit.cnot(0, 1)
    
    # Diffusion
    circuit.h(0)
    circuit.h(1)
    circuit.x(0)
    circuit.x(1)
    circuit.cnot(0, 1)
    circuit.rz(1, np.pi)
    circuit.cnot(0, 1)
    circuit.x(0)
    circuit.x(1)
    circuit.h(0)
    circuit.h(1)
    
    # Run
    device = LocalSimulator("braket_dm")
    task = device.run(circuit, shots=100)
    counts = task.result().measurement_counts
    
    print("Ideal Grover circuit results:")
    print(f"Counts: {counts}")
    success_rate = counts.get('11', 0) / 100
    print(f"Success rate: {success_rate:.2f}")
    
except ImportError:
    print("Braket not installed, skipping test")
