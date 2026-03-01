# Sigma-C Framework v2.0.3

**Universal Criticality Analysis & Active Control System**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-2.0.3-green.svg)](https://pypi.org/project/sigma-c-framework/)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

## Overview

Sigma-C is a framework for detecting and analyzing **critical phase transitions** across physical, computational, and data-driven systems. It provides a unified susceptibility-based approach: sweep a control parameter, compute the response function (susceptibility), and locate the critical point where the system transitions between qualitatively different regimes.

The core idea is simple: for any system with a tunable parameter and a measurable observable, the susceptibility `chi = dO/d(epsilon)` peaks at the critical point `sigma_c`. The sharpness of that peak (`kappa`) quantifies how pronounced the transition is.

### Peer-Reviewed Application

The methodology behind Sigma-C has been validated in a peer-reviewed publication:

> **"Operational scale detection in quantum magnetism"**
> AVS Quantum Science, Volume 8, Issue 1, Article 013804 (2026)
> [https://doi.org/10.1116/5.0254846](https://doi.org/10.1116/5.0254846)

This paper demonstrates the framework's application to **quantum computing on real hardware** (Rigetti Ankaa-3), where Sigma-C successfully identifies the critical noise threshold at which quantum algorithms lose their advantage over classical computation. The detected critical point (`sigma_c = 0.070 +/- 0.009`) and correlation length (`xi_c = 8.00 +/- 0.50 qubits`) are consistent with theoretical predictions from quantum error correction theory.

## Core Capabilities

- **Susceptibility Analysis**: Detect critical points via `chi = dO/d(epsilon)` with Gaussian kernel smoothing
- **Active Control**: PID controller to maintain systems at or near critical points
- **Streaming Computation**: O(1) real-time susceptibility updates using Welford's algorithm
- **Observable Discovery**: Automatic identification of optimal order parameters
- **Multi-Scale Analysis**: Wavelet-based criticality detection across scales
- **Statistical Rigor**: Jonckheere-Terpstra trend tests, isotonic regression with bootstrap CI
- **High-Performance Core**: Optional C++ backend via pybind11

## Domain Adapters

| Domain | Adapter | Key Methods |
|--------|---------|-------------|
| Quantum | `QuantumAdapter` | Noise sweep, depth scaling, idle sensitivity, Fisher information |
| GPU/HPC | `GPUAdapter` | Cache transition detection, roofline analysis, thermal throttling |
| Finance | `FinancialAdapter` | Hurst exponent, GARCH(1,1) volatility, order flow imbalance |
| Climate | `ClimateAdapter` | Mesoscale boundary detection, vertical stability analysis |
| Seismic | `SeismicAdapter` | Gutenberg-Richter b-value, Omori aftershock scaling |
| Magnetic | `MagneticAdapter` | Critical exponents (beta, gamma, alpha), finite size scaling |
| ML | `MLAdapter` | Training robustness, learning rate sensitivity |
| Edge/IoT | `EdgeAdapter` | Power efficiency phase transitions |
| LLM Cost | `LLMCostAdapter` | Cost-quality Pareto frontier analysis |

## Installation

```bash
# Core framework
pip install sigma-c-framework

# With quantum integrations
pip install sigma-c-framework[quantum]

# With ML integrations
pip install sigma-c-framework[ml]
```

## Quick Start

### Detecting a Phase Transition (Ising Model)

```python
import numpy as np
from sigma_c import Universe

# Generate synthetic magnetization data across temperatures
temperatures = np.linspace(1.5, 3.5, 50)
# Simulate mean-field magnetization: M ~ (Tc - T)^0.125
Tc = 2.269  # Exact 2D Ising critical temperature
magnetization = np.where(
    temperatures < Tc,
    np.abs(Tc - temperatures)**0.125,
    0.01 * np.random.randn(np.sum(temperatures >= Tc))
)

# Find the critical point using susceptibility analysis
mag = Universe.magnetic()
result = mag.compute_susceptibility(temperatures, magnetization)

print(f"Detected Tc:    {result['sigma_c']:.3f}")
print(f"Theoretical Tc: {Tc}")
print(f"Peak sharpness: {result['kappa']:.1f}")
```

### Quantum Noise Threshold Detection

```python
import numpy as np
from sigma_c import Universe

qpu = Universe.quantum(device='simulator')
result = qpu.run_optimization(
    circuit_type='grover',
    epsilon_values=np.linspace(0.0, 0.25, 20),
    shots=1000
)

print(f"Critical noise level: {result['sigma_c']:.4f}")
print(f"Peak clarity (kappa): {result['kappa']:.1f}")
# Above sigma_c, Grover's algorithm loses quantum advantage
```

### Financial Volatility Regime Detection

```python
import numpy as np
from sigma_c import Universe

fin = Universe.finance()
returns = np.random.randn(1000) * 0.02  # Simulated daily returns

# GARCH(1,1) volatility clustering analysis
garch = fin.analyze_volatility_clustering(returns)
print(f"Persistence: {garch['persistence']:.3f}")
print(f"Regime:      {'Critical' if garch['persistence'] > 0.95 else 'Stable'}")
```

## Integrations

- **Quantum**: Qiskit, PennyLane, Cirq, AWS Braket
- **ML**: PyTorch, JAX, TensorFlow
- **Monitoring**: Grafana, Kubernetes
- **Reporting**: LaTeX, publication-quality plots

## Documentation

- [API Reference](docs/API_REFERENCE_v2.0.md)
- [Release Notes](docs/releases/RELEASE_NOTES_v2.0.2.md)
- [Examples](examples/v4/)

## License

**Open Source**: AGPL-3.0-or-later
**Commercial**: Contact [nfo@forgottenforge.xyz](mailto:nfo@forgottenforge.xyz)

Copyright (c) 2025 ForgottenForge.xyz
