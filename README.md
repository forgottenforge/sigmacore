# Sigma-C Framework v1.2.3 "Universal Optimization"

The Universal Optimization Framework for Quantum, GPU, Financial, and ML Systems.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-1.2.3-green.svg)](https://pypi.org/project/sigma-c-framework/)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

## 🚀 Overview

Sigma-C is a unified framework for optimizing complex systems by balancing **Performance** (Efficiency/Returns/Accuracy) against **Stability** (Resilience/Sigma_c).

It provides a consistent API to optimize:
- **Quantum Circuits**: Maximize fidelity while minimizing noise susceptibility
- **GPU Kernels**: Maximize throughput while maintaining thermal/memory stability
- **Financial Strategies**: Maximize returns while minimizing crash risk (sigma_c)
- **ML Models**: Maximize accuracy while ensuring adversarial robustness

## ✨ New in v1.2.3

- **Machine Learning Optimizer**: Optimize neural networks for robustness (`BalancedMLOptimizer`)
- **Hardware-Aware Quantum**: Native gate optimization for Rigetti, IQM, and IBM
- **Enhanced Physics**: Holevo bound, Roofline model, and No-Cloning theorem validation
- **Extended Documentation**: Comprehensive guides for hardware and domain extensions

## 📦 Installation

```bash
pip install sigma-c-framework
```

Or from source:
```bash
git clone https://github.com/forgottenforge/sigma-c-framework.git
cd sigma-c-framework
pip install -e .
```

## 🔧 Quick Start

### 1. Quantum Optimization
```python
from sigma_c.adapters.quantum import QuantumAdapter
from sigma_c.optimization.quantum import BalancedQuantumOptimizer

# Initialize with hardware-aware compilation
adapter = QuantumAdapter(config={'device': 'rigetti', 'auto_compile': True})
optimizer = BalancedQuantumOptimizer(adapter)

# Optimize Grover's Algorithm
result = optimizer.optimize_circuit(
    circuit_factory=my_grover_circuit,
    param_space={'epsilon': [0.0, 0.01], 'idle_frac': [0.0, 0.1]}
)
print(f"Optimal Params: {result.optimal_params}")
```

### 2. ML Optimization (New!)
```python
from sigma_c.optimization.ml import BalancedMLOptimizer

optimizer = BalancedMLOptimizer(performance_weight=0.7, stability_weight=0.3)

# Optimize Neural Network Hyperparameters
result = optimizer.optimize_model(
    model_factory=create_model,
    param_space={
        'learning_rate': [0.001, 0.01],
        'dropout': [0.1, 0.2, 0.3]
    }
)
print(f"Robust Accuracy: {result.score}")
```

### 3. Financial Optimization
```python
from sigma_c.adapters.financial import FinancialAdapter
from sigma_c.optimization.financial import BalancedFinancialOptimizer

adapter = FinancialAdapter()
optimizer = BalancedFinancialOptimizer(adapter)

# Optimize Trading Strategy
result = optimizer.optimize_strategy(
    param_space={'lookback': [60, 126, 252], 'threshold': [0.01, 0.02]}
)
print(f"Stable Returns: {result.performance_after}")
```

## 📚 Documentation

- [Full Documentation](DOCUMENTATION.md)
- [Hardware Compatibility](HARDWARE_COMPATIBILITY.md)
- [Extending Domains](EXTENDING_DOMAINS.md)
- [Release Notes](RELEASE_NOTES_v1.2.3.md)

## 🛡️ License

**Open Source**: AGPL-3.0-or-later  
**Commercial**: Contact [info@forgottenforge.xyz](mailto:info@forgottenforge.xyz) for commercial licensing options.

Copyright © 2025 ForgottenForge.xyz
