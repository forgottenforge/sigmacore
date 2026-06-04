5. [Financial Optimization](#financial-optimization)
6. [ML Optimization (New!)](#ml-optimization)
7. [Physics Validation](#physics-validation)
8. [Extending the Framework](#extending-the-framework)


The framework now supports hardware-specific optimization for Rigetti, IQM, and IBM devices.

```python
from sigma_c.adapters.quantum import QuantumAdapter

# Auto-detect and optimize for Rigetti
adapter = QuantumAdapter(config={'device': 'rigetti', 'auto_compile': True})
```

### Grover's Algorithm

Optimized implementation using native CZ gates for maximum fidelity.

```python
circuit = adapter.create_grover_with_noise(n_qubits=2, epsilon=0.0)
```

## GPU Optimization

Optimizes CUDA kernels for throughput and thermal stability.

- **Roofline Model**: Validates performance against hardware limits
- **Regime Detection**: Automatically detects memory-bound vs. compute-bound workloads

```python
from sigma_c.optimization.gpu import BalancedGPUOptimizer
optimizer = BalancedGPUOptimizer(gpu_adapter)
```

## Financial Optimization

Optimizes trading strategies for risk-adjusted returns.

```python
from sigma_c.optimization.financial import BalancedFinancialOptimizer
optimizer = BalancedFinancialOptimizer(financial_adapter)
```

## ML Optimization (New in v1.2.3)

Optimizes neural networks for robustness and accuracy.

### BalancedMLOptimizer

Balances validation accuracy with adversarial robustness (sigma_c).

```python
from sigma_c.optimization.ml import BalancedMLOptimizer

optimizer = BalancedMLOptimizer(performance_weight=0.7, stability_weight=0.3)
result = optimizer.optimize_model(
    model_factory=create_model,
    param_space={'learning_rate': [0.001, 0.01], 'dropout': [0.1, 0.2]}
)
```

## Physics Validation

Rigorous theoretical checks ensure optimization results are physically valid.

- **Quantum**: Holevo bound, No-Cloning theorem
- **GPU**: Roofline model, Little's Law
- **Financial**: Arbitrage-free conditions
- **ML**: PAC learning bounds

## Extending the Framework

See [EXTENDING_DOMAINS.md](sigma_c_framework/EXTENDING_DOMAINS.md) for a complete guide on adding new domains.

## Hardware Compatibility

See [HARDWARE_COMPATIBILITY.md](sigma_c_framework/HARDWARE_COMPATIBILITY.md) for supported quantum and GPU hardware.
