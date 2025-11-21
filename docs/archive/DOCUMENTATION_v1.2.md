# Sigma-C Framework v1.2.0 "Full Power" Documentation

**Copyright (c) 2025 ForgottenForge.xyz**  
**License:** AGPL-3.0-or-later OR Commercial License

---

## 1. Introduction & Philosophy

**Sigma-C** is a universal framework for optimizing complex systems by balancing **Performance** (Fidelity, Throughput, Profit) against **Resilience** (Stability, Robustness). 

The core philosophy is based on the **Criticality Hypothesis**:
> Systems perform best near a critical phase transition ($\sigma_c$), where they are flexible enough to adapt but stable enough to function.

v1.2.0 "Full Power" extends this philosophy from Quantum Processing Units (QPUs) to **GPUs** (High-Performance Computing) and **Financial Markets**.

---

## 2. Installation

```bash
pip install .
```

**Requirements:**
- Python 3.8+
- C++ Compiler (for core engine)
- CUDA Toolkit (optional, for GPU features)

---

## 3. Core Concepts

### $\sigma_c$ (Sigma-C)
The critical susceptibility parameter. It measures how sensitive a system is to perturbations.
- **Low $\sigma_c$**: System is rigid, stable, but potentially stuck in local optima.
- **High $\sigma_c$**: System is chaotic, unstable, high variance.
- **Optimal $\sigma_c$**: The "Edge of Chaos" where maximum computational/financial power exists.

### $\kappa$ (Kappa)
The criticality score. Indicates how close the system is to a phase transition.

---

## 4. Modules Guide

### 4.1 Optimization (`sigma_c.optimization`)

The engine room of the framework. These classes find the optimal parameters for your system.

#### `UniversalOptimizer` (Base Class)
- **Rationale**: Provides a standard interface (`optimize()`) for all domains.
- **Usage**: Subclass this to create custom optimizers.
- **Modification**: Override `_evaluate_performance` and `_evaluate_stability`.

#### `BalancedQuantumOptimizer`
- **Target**: Quantum Circuits (Braket, Qiskit).
- **Rationale**: Quantum circuits degrade with noise. We balance Fidelity (success rate) with Resilience (stability against noise).
- **Strategies**: 
    - `add_echo_pulses`: Reduces dephasing.
    - `virtualize_z_gates`: Zero-error gates.
    - `decompose_complex_gates`: Reduces error rates.

#### `BalancedGPUOptimizer`
- **Target**: CUDA Kernels, HPC.
- **Rationale**: GPU kernels face thermal throttling and memory bottlenecks. We optimize for throughput while maintaining thermal/memory stability.
- **Strategies**:
    - `tensor_core_enable`: Uses mixed precision (float16) for 2x-4x speedup.
    - `memory_coalescing`: Reorders memory access for bandwidth efficiency.
    - `async_streams`: Overlaps compute and data transfer.
    - `block_size_tuning`: Finds optimal thread occupancy.
- **Usage**:
  ```python
  from sigma_c.optimization.gpu import BalancedGPUOptimizer
  opt = BalancedGPUOptimizer(target_sigma_c=0.1)
  result = opt.optimize(my_kernel_evaluator, initial_params)
  ```

#### `BalancedFinancialOptimizer`
- **Target**: Trading Strategies, Portfolios.
- **Rationale**: High returns often come with crash risk. We optimize Sharpe Ratio while constraining $\sigma_c$ (crash probability).
- **Strategies**:
    - `dynamic_hedging`: Adjusts hedge ratio based on volatility.
    - `portfolio_diversification`: Spreads risk.

---

### 4.2 Physics (`sigma_c.physics`)

The "Lawyer" of the framework. Validates that results are physically possible and meaningful.

#### `RigorousTheoreticalCheck` (Base Class)
- **Rationale**: Prevents "hallucinated" good results by checking against physical laws.

#### `RigorousGPUSigmaC`
- **Features**:
    - **Roofline Model**: Checks if performance is Compute-bound or Memory-bound.
    - **Thermal Monitoring**: Uses `pynvml` to check GPU temperature and power.
    - **Spectral Analysis**: Uses FFT to measure $\sigma_c$ of periodic signals (e.g., frame times).
- **Usage**: Automatically called by `BalancedGPUOptimizer`.

#### `RigorousQuantumSigmaC`
- **Features**: Checks against Quantum Fisher Information (QFI) limits.

---

### 4.3 Prediction (`sigma_c.prediction`)

**New in v1.2.0**. Uses ML to predict system behavior.

#### `MLDiscovery`
- **Rationale**: Finds hidden correlations in high-dimensional data.
- **Usage**:
  ```python
  from sigma_c.prediction.ml import MLDiscovery
  ml = MLDiscovery()
  features = ml.find_critical_features(data_X, data_y, feature_names)
  ```

#### `BlindPredictor`
- **Rationale**: Predicts $\sigma_c$ when direct measurement is impossible (e.g., live trading).

---

### 4.4 Reporting (`sigma_c.reporting`)

Generates professional, publication-ready outputs.

#### `LatexGenerator`
- **Rationale**: Automates the creation of scientific papers/reports.
- **Usage**:
  ```python
  from sigma_c.reporting.latex import LatexGenerator
  gen = LatexGenerator()
  gen.generate_report(results, "my_report.tex")
  ```

#### `PublicationVisualizer`
- **Rationale**: Creates high-DPI, aesthetically pleasing plots (matplotlib/seaborn).

---

## 5. Advanced Usage & Modification

### Adding a New Domain
1. Create `sigma_c/adapters/new_domain.py` inheriting from `SigmaCAdapter`.
2. Create `sigma_c/optimization/new_domain.py` inheriting from `UniversalOptimizer`.
3. Create `sigma_c/physics/new_domain.py` inheriting from `RigorousTheoreticalCheck`.

### Customizing Optimization Strategies
Modify the `strategies` list in your optimizer class and implement the logic in `_apply_strategy`.

```python
class MyOptimizer(UniversalOptimizer):
    def __init__(self):
        super().__init__()
        self.strategies = ['my_new_strategy']

    def _apply_strategy(self, params, strategy):
        if strategy == 'my_new_strategy':
            params['x'] *= 2  # Example logic
        return params
```

---

## 6. Release Notes v1.2.0

- **Full Power Release**: All domains (Quantum, GPU, Finance) fully implemented.
- **Universal Rigor**: Physics checks generalized to all domains.
- **New Features**:
    - GPU: Tensor Cores, Async Streams, Thermal Monitoring.
    - Finance: Portfolio Optimization, Crash Prediction.
    - ML: Feature Discovery.
    - Reporting: LaTeX generation.
