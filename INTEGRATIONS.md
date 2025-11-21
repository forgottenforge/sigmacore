# Sigma-C Framework - Integrations Guide

## 🔌 Available Integrations

Sigma-C v2.0 provides **22+ integrations** across quantum computing, ML, finance, DevOps, and web platforms.

## Quick Start

```bash
# Install core framework
pip install sigma-c-framework

# Install with all integrations
pip install sigma-c-framework[all]

# Install specific integrations
pip install sigma-c-framework[quantum]  # Qiskit, PennyLane, Cirq
pip install sigma-c-framework[ml]       # PyTorch, JAX
pip install sigma-c-framework[finance]  # QuantLib
pip install sigma-c-framework[devops]   # Kubernetes, Grafana
```

## 🎯 Quantum Computing

### Qiskit
```python
from qiskit import QuantumCircuit
from sigma_c.connectors.qiskit import QiskitSigmaC

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)

# Automatic σ_c analysis
sigma_c = QiskitSigmaC.analyze(circuit)
optimized = QiskitSigmaC.optimize_for_backend(circuit, 'ibmq_manila')
```

### PennyLane
```python
import pennylane as qml
from sigma_c.plugins.pennylane import SigmaCDevice

dev = qml.device('sigma_c.simulator', wires=4, critical_point=0.1)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))
```

## 🧠 Machine Learning

### PyTorch
```python
import torch
from sigma_c.ml.pytorch import CriticalModule, SigmaCLoss

class MyNet(CriticalModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        return self.critical_forward(x)

# Loss with criticality regularization
criterion = SigmaCLoss(lambda_critical=0.1)
```

### JAX
```python
import jax
from sigma_c.ml.jax import critical_jit

@critical_jit
def train_step(params, batch):
    # σ_c is automatically tracked
    loss, sigma_c = ...
    return loss, sigma_c
```

## 💰 Finance

### QuantLib
```python
from sigma_c.finance.quantlib import CriticalPricing

option_price = CriticalPricing.black_scholes(
    S=100, K=110, r=0.05, sigma=0.2,
    criticality_adjustment=True
)
```

## 🔧 DevOps

### Kubernetes
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
    memory: 0.7
  actions:
    scale: true
    alert: true
```

### GitHub Actions
```yaml
name: Criticality Check
on: [push]
jobs:
  sigma-c:
    runs-on: ubuntu-latest
    steps:
      - uses: sigma-c/action@v1
        with:
          threshold: 0.7
          fail-on-critical: true
```

### Grafana
```python
from sigma_c.monitoring.grafana import GrafanaExporter

exporter = GrafanaExporter(prometheus_gateway='localhost:9091')
exporter.push_metrics('my-system', sigma_c=0.5, kappa=10.0, chi_max=100.0)
```

## 🌐 Web APIs

### REST API
```python
from fastapi import FastAPI
from sigma_c.api import SigmaCAPI

app = FastAPI()
sigma_api = SigmaCAPI()

@app.post("/analyze")
async def analyze(data: List[float]):
    return {"sigma_c": sigma_api.compute(data)}
```

### GraphQL
```graphql
query {
  analyzeSystem(input: {
    epsilon: [0.0, 0.1, 0.2]
    observable: [1.0, 0.8, 0.5]
  }) {
    sigmaC
    kappa
    confidence
  }
}
```

## 🔥 Universal Bridge

Make **ANY** function criticality-aware:

```python
from sigma_c.connectors.bridge import SigmaCBridge

# Wrap any function
@SigmaCBridge.wrap_any_function
def my_function(x):
    return x ** 2

result = my_function(5)
print(result.__sigma_c__)  # Criticality metadata

# Wrap entire classes
from sklearn.ensemble import RandomForestClassifier
CriticalRF = SigmaCBridge.wrap_class(RandomForestClassifier)
```

## 📦 Installation Matrix

| Integration | Package | Install Command |
|------------|---------|-----------------|
| Qiskit | `qiskit` | `pip install sigma-c-framework[quantum]` |
| PyTorch | `torch` | `pip install sigma-c-framework[ml]` |
| FastAPI | `fastapi` | `pip install sigma-c-framework[api]` |
| Kubernetes | `kubernetes` | `pip install sigma-c-framework[k8s]` |
| Grafana | `prometheus-client` | `pip install sigma-c-framework[monitoring]` |

## 🚀 One-Liner Installation

```bash
curl -sSL https://sigma-c.io/install.sh | sh
```

This script:
- Detects installed frameworks
- Installs appropriate connectors
- Configures auto-discovery

## 📚 Documentation

- [API Reference](API_REFERENCE_v2.0.md)
- [Release Notes](RELEASE_NOTES_v2.0.0.md)
- [Examples](examples_v2.0/)

## 🤝 Contributing

Want to add a new integration? See [EXTENDING_DOMAINS.md](EXTENDING_DOMAINS.md)

## 📄 License

Copyright (c) 2025 ForgottenForge.xyz  
Licensed under AGPL-3.0-or-later OR Commercial License
