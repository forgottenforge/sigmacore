#!/usr/bin/env python3
"""Test RZ decomposition"""
import sys, os
sys.path.append('.')

try:
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    import numpy as np
    
    # Test 1: Direct RZ
    circuit1 = Circuit()
    circuit1.h(0)
    circuit1.h(1)
    circuit1.cnot(0, 1)
    circuit1.rz(1, np.pi)  # Direct RZ
    circuit1.cnot(0, 1)
    
    # Test 2: Decomposed RZ (current implementation)
    circuit2 = Circuit()
    circuit2.h(0)
    circuit2.h(1)
    circuit2.cnot(0, 1)
    # _rz_physical decomposition
    circuit2.rx(1, np.pi/2)
    circuit2.ry(1, np.pi)
    circuit2.rx(1, -np.pi/2)
    circuit2.cnot(0, 1)
    
    device = LocalSimulator("braket_dm")
    
    task1 = device.run(circuit1, shots=100)
    counts1 = task1.result().measurement_counts
    
    task2 = device.run(circuit2, shots=100)
    counts2 = task2.result().measurement_counts
    
    print("Direct RZ:", counts1)
    print("Decomposed RZ:", counts2)
    
except ImportError:
    print("Braket not installed")
