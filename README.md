# Sigma-C Framework v2.0.0 "Rigorous Control"

Universal Criticality Analysis & Active Control System

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://pypi.org/project/sigma-c-framework/)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

## 🚀 Overview

Sigma-C v2.0 is a rigorous active control system that detects, analyzes, and maintains critical points across quantum, GPU, financial, climate, and ML systems.

**New in v2.0**: **22+ Framework Integrations** - Connect to Qiskit, PyTorch, Kubernetes, Grafana, and more!

## ✨ What's New in v2.0

### Core Features
- **Observable Discovery**: Automatic identification of optimal order parameters
- **Multi-Scale Analysis**: Wavelet-based criticality detection across scales
- **Active Control**: PID controller for critical point maintenance
- **Streaming Calculation**: O(1) real-time susceptibility updates

### New Domains
- **Climate**: Mesoscale boundary detection
- **Seismic**: Gutenberg-Richter analysis
- **Magnetic**: Critical exponents validation
- **Edge Computing**: Power efficiency optimization
- **LLM Cost**: Model selection via Pareto frontier

### 🔌 Universal Connectivity
- **Quantum**: Qiskit, PennyLane, Cirq, AWS Braket
- **ML**: PyTorch, JAX, TensorFlow
- **Finance**: QuantLib, Zipline
- **DevOps**: Kubernetes, GitHub Actions, Grafana
- **Web**: REST API, GraphQL, WebAssembly

## 📦 Installation

```bash
# Core framework
pip install sigma-c-framework

# With all integrations
pip install sigma-c-framework[all]

# Specific integrations
pip install sigma-c-framework[quantum]   # Qiskit, PennyLane
pip install sigma-c-framework[ml]        # PyTorch, JAX
pip install sigma-c-framework[devops]    # K8s, Grafana
```

## 🔧 Quick Start

### Quantum (Qiskit)
```python
from qiskit import QuantumCircuit
from sigma_c.connectors.qiskit import QiskitSigmaC

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)

# Automatic criticality analysis
result = QiskitSigmaC.analyze(circuit)
print(f"σ_c = {result['sigma_c']:.4f}")
```

### Machine Learning (PyTorch)
```python
from sigma_c.ml.pytorch import CriticalModule, SigmaCLoss

class MyNet(CriticalModule):
    def forward(self, x):
        return self.critical_forward(x)  # Auto σ_c tracking

criterion = SigmaCLoss(lambda_critical=0.1)
```

### Universal Bridge (Any Framework!)
```python
from sigma_c.connectors.bridge import SigmaCBridge

@SigmaCBridge.wrap_any_function
def my_function(x):
    return x ** 2

result = my_function(5)
print(result.__sigma_c__)  # Criticality metadata
```

### DevOps (Kubernetes)
```yaml
apiVersion: sigma-c.io/v1
kind: CriticalityMonitor
metadata:
  name: app-monitor
spec:
  target:
    app: my-app
  thresholds:
    cpu: 0.8
  actions:
    scale: true
```

## 📚 Documentation

- **[Integrations Guide](INTEGRATIONS.md)** - All 22+ integrations
- **[API Reference](API_REFERENCE_v2.0.md)** - Complete API docs
- **[Release Notes](RELEASE_NOTES_v2.0.0.md)** - What's new in v2.0
- **[Examples](examples_v2.0/)** - Code examples

## 🎯 Use Cases

- **Quantum Computing**: Optimize circuits for NISQ devices
- **GPU/HPC**: Detect cache transitions, thermal throttling
- **Finance**: Predict market crashes, optimize portfolios
- **ML**: Train robust models, detect overfitting
- **Climate**: Identify mesoscale boundaries
- **Edge/IoT**: Optimize power efficiency

## 🛡️ License

**Open Source**: AGPL-3.0-or-later  
**Commercial**: Contact [info@forgottenforge.xyz](mailto:info@forgottenforge.xyz)

Copyright © 2025 ForgottenForge.xyz
