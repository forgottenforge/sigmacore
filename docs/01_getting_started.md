# Getting Started with Sigma-C

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA Drivers for GPU support
- (Optional) AWS Credentials for Quantum support

### Install via pip
The easiest way to install Sigma-C is via PyPI:

```bash
pip install sigma-c-framework
```

### Install from Source
For the latest features and development:

```bash
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework
pip install -e .
```

## Quickstart: Your First Optimization

Let's optimize a simple mathematical function to understand the workflow. We'll balance finding the minimum of a parabola (Performance) with staying close to zero (Stability).

Create a file named `hello_sigma.py`:

```python
from sigma_c.optimization.universal import UniversalOptimizer

# 1. Define the System (The problem to solve)
class SimpleSystem:
    pass

# 2. Define the Optimizer
class MyOptimizer(UniversalOptimizer):
    def _evaluate_performance(self, system, params):
        # Goal: Minimize (x-2)^2. We return negative because higher is better.
        x = params['x']
        return -((x - 2) ** 2)

    def _evaluate_stability(self, system, params):
        # Goal: Keep x close to 0. Stability = 1 / (1 + |x|)
        x = params['x']
        return 1.0 / (1.0 + abs(x))

    def _apply_params(self, system, params):
        return system

# 3. Run Optimization
optimizer = MyOptimizer(performance_weight=0.7, stability_weight=0.3)
param_space = {'x': [i * 0.1 for i in range(-50, 50)]} # -5.0 to 5.0

result = optimizer.optimize(SimpleSystem(), param_space)

print(f"Best X: {result.optimal_params['x']:.2f}")
print(f"Score: {result.score:.4f}")
```

Run it:
```bash
python hello_sigma.py
```

## CLI Basics

Sigma-C includes a powerful Command Line Interface (CLI).

### Check Version
```bash
sigma-c --version
```

### Run Optimization from Config
You can define optimizations in YAML and run them directly:

```yaml
# config.yaml
optimizer:
  type: "sigma_c.optimization.gpu.BalancedGPUOptimizer"
  performance_weight: 0.8
  stability_weight: 0.2
params:
  block_size: [128, 256, 512]
```

Run it:
```bash
sigma-c run config.yaml
```

### Visualize Results
If you saved a result to JSON:
```bash
sigma-c visualize result.json
```
