# GPU Optimization Guide

## Overview
The GPU domain optimizes CUDA kernels and compute workloads. It balances **Throughput** (GFLOPS) against **Thermal Stability** and **Memory Bandwidth Efficiency**.

## The Roofline Model
Sigma-C uses the Roofline Model to determine if a kernel is **Compute Bound** or **Memory Bound**.

- **Arithmetic Intensity (AI)**: FLOPS / Bytes Transferred.
- **Ridge Point**: The AI value where the GPU switches from memory-bound to compute-bound.

### Automatic Regime Detection
The `BalancedGPUOptimizer` automatically detects the regime:

```python
from sigma_c.optimization.gpu import BalancedGPUOptimizer

optimizer = BalancedGPUOptimizer()
# Returns 'memory_bound' or 'compute_bound'
regime = optimizer.detect_regime(kernel_params)
```

## Thermal Management
High performance often leads to thermal throttling, which destroys stability. Sigma-C models this relationship.

### Stability Metric
$$ \sigma_c = \frac{T_{max} - T_{current}}{T_{max} - T_{ambient}} $$
A higher $\sigma_c$ means more thermal headroom.

## Example: Matrix Multiplication Tuning
```python
from sigma_c.adapters.gpu import GPUAdapter
from sigma_c.optimization.gpu import BalancedGPUOptimizer

# Define parameter space for block sizes
param_space = {
    'block_x': [16, 32],
    'block_y': [16, 32],
    'thread_per_block': [128, 256, 512]
}

# Run optimization
optimizer = BalancedGPUOptimizer()
result = optimizer.optimize_kernel(my_kernel_factory, param_space)
```
