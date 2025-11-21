# Quantum Optimization Guide

## Overview
The Quantum domain in Sigma-C focuses on optimizing quantum circuits to balance **Fidelity** (Performance) against **Noise Resilience** (Stability).

## Hardware-Aware Compilation
Sigma-C v1.2.3 introduces hardware-aware compilation, allowing you to target specific quantum processors (QPUs).

### Supported Backends
- **Rigetti**: Native CZ gates.
- **IQM**: Native CZ gates.
- **IBM**: CNOT basis (via transpilation).
- **Simulators**: AWS Braket SV1, DM1.

### Example: Optimizing for Rigetti
```python
from sigma_c.adapters.quantum import QuantumAdapter

# Configure for Rigetti Ankaa-3
config = {
    'device': 'rigetti',
    'auto_compile': True,
    'shots': 1000
}
adapter = QuantumAdapter(config)

# Create a circuit (e.g., Grover's Algorithm)
circuit = adapter.create_grover_with_noise(n_qubits=2, epsilon=0.0)
```

## Noise Models
Sigma-C simulates realistic noise to calculate $\sigma_c$.

- **Depolarizing Noise**: Random errors applied to all qubits.
- **Dephasing Noise**: Loss of quantum information over time (T2).
- **Readout Error**: Probability of measuring 0 as 1 (and vice-versa).

You can control these parameters in the `QuantumAdapter`:
```python
adapter.set_noise_model(
    depolarizing_prob=0.001,
    dephasing_prob=0.002,
    readout_error=0.01
)
```

## Algorithms
The framework includes built-in factories for common algorithms:
- **Grover's Algorithm**: Search unstructured databases.
- **QAOA**: Quantum Approximate Optimization Algorithm.
- **VQE**: Variational Quantum Eigensolver (via custom factories).
