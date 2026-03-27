# Sigma-C Framework v3.0.0

**Universal Criticality Analysis & Active Control System**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)](https://pypi.org/project/sigma-c-framework/)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

## What's New in v3.0

- **Contraction Geometry**: D (contraction diameter) and gamma (contraction ratio) as first-class metrics
- **Four-Type Classification**: Systems classified as D (Divergent), O (Oscillatory), S (Stable), or R (Resonant)
- **3 New Domains**: Number Theory, Protein Stability, and Linguistics adapters
- **Extended Derivative Estimation**: Configurable derivative methods for susceptibility computation
- **Formal Validation**: Rigorous mathematical validation of criticality claims
- **Information Theory**: Shannon entropy and mutual information analysis in `beyond/information.py`
- **12 Domain Adapters** (was 9) with full backward compatibility
- **85+ New Tests** across 9 new test modules
- **7 New Demos** showcasing all v3.0 features

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
- **High-Performance Core**: Optional C++ backend via pybind11, CUDA acceleration via CuPy

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
| Number Theory | `NumberTheoryAdapter` | 12-map verification, prime distribution analysis |
| Protein | `ProteinAdapter` | TTR/LYZ/GSN/SOD1/PRNP mutation stability |
| Linguistics | `LinguisticsAdapter` | Cross-language correlation analysis |

## Integrations

| Category | Integration | Description |
|----------|------------|-------------|
| **Quantum** | Qiskit | Circuit noise sensitivity analysis |
| | PennyLane | VQA criticality tracking device |
| | Cirq | Circuit optimization for stability |
| | AWS Braket | Native quantum hardware adapter |
| **ML Frameworks** | PyTorch | `CriticalModule` with activation tracking |
| | TensorFlow | `SigmaCCallback` for Keras training |
| | JAX | `critical_jit` decorator, `CriticalOptimizer` |
| | CUDA/CuPy | GPU-accelerated susceptibility computation |
| **APIs** | REST | FastAPI endpoint (`SigmaCAPI`) |
| | GraphQL | Strawberry + built-in zero-dep resolver |
| | WASM | Browser-native JS module generator |
| **Monitoring** | Grafana | Prometheus metrics export (push + pull) |
| | Kubernetes | Pod criticality monitoring + autoscaling |
| | GitHub Actions | AST-based code complexity CI gate |
| **Finance** | QuantLib | Black-Scholes with criticality adjustment |
| | Zipline | Crash avoidance trading strategy |
| **Platforms** | Home Assistant | Smart home criticality sensor |
| | VSCode | Real-time code complexity status bar |
| **Reporting** | LaTeX | Publication-ready tables, figures, reports |
| **Bindings** | Julia | `SigmaC.jl` native binding |
| | Mathematica | `SigmaC.m` Wolfram Language binding |
| | Lean 4 | `SigmaC.lean` theorem prover binding |

## Installation

```bash
# Core framework
pip install sigma-c-framework

# With quantum integrations
pip install sigma-c-framework[quantum]

# With GPU acceleration
pip install sigma-c-framework[gpu]
```

## Quick Start

### Detecting a Phase Transition (Ising Model)

```python
import numpy as np
from sigma_c import Universe

# Generate synthetic magnetization data across temperatures
temperatures = np.linspace(1.5, 3.5, 50)
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

## Examples

The `examples/v4/` directory contains 12 demo files covering every module:

| Demo | Covers |
|------|--------|
| `demo_quantum.py` | Quantum noise threshold detection |
| `demo_finance.py` | GARCH volatility, Hurst exponent |
| `demo_climate.py` | Atmospheric mesoscale boundary |
| `demo_magnetic.py` | 2D Ising Curie temperature |
| `demo_seismic.py` | Gutenberg-Richter b-value |
| `demo_gpu.py` | GPU cache transition, roofline |
| `demo_diagnostics.py` | Universal diagnostics system |
| `demo_integrations.py` | GraphQL, CI, REST, WASM, HA, TF, LaTeX, Bridge |
| `demo_ml_frameworks.py` | PyTorch, JAX, CUDA, TensorFlow |
| `demo_quantum_connectors.py` | Qiskit, PennyLane, Cirq |
| `demo_edge_llm.py` | Edge IoT, ML hyperparameters, LLM cost |
| `demo_optimization.py` | ML optimizer, brute force, QuantLib, Zipline, Grafana/K8s |

All demos run locally without external services or optional dependencies.

## Documentation

- [API Reference](docs/API_REFERENCE_v3.0.md)
- [Release Notes](docs/releases/)
- [Examples](examples/v4/)

## License

**Open Source**: AGPL-3.0-or-later
**Commercial**: Contact [nfo@forgottenforge.xyz](mailto:nfo@forgottenforge.xyz)

Copyright (c) 2025 ForgottenForge.xyz
