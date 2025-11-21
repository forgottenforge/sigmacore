# API Reference

## Core

### `sigma_c.optimization.universal.UniversalOptimizer`
The abstract base class for all optimizers.

**Methods:**
- `optimize(system, param_space, strategy='brute_force', callbacks=None)`: Main entry point.
- `save(filepath)`: Save state to JSON.
- `load(filepath)`: Load state from JSON.
- `calculate_score(performance, stability)`: Returns weighted composite score.

### `sigma_c.core.callbacks.OptimizationCallback`
Base class for callbacks.

**Methods:**
- `on_optimization_start(optimizer, param_space)`
- `on_step_end(optimizer, step, logs)`
- `on_optimization_end(optimizer, result)`

## Adapters

### `sigma_c.adapters.quantum.QuantumAdapter`
Interface for Quantum Processors.

**Config:**
- `device`: 'rigetti', 'iqm', 'ibm', or AWS ARN.
- `auto_compile`: Boolean.

**Methods:**
- `create_grover_with_noise(n_qubits, epsilon)`
- `compile_for_hardware(circuit)`

### `sigma_c.adapters.gpu.GPUAdapter`
Interface for NVIDIA GPUs.

**Methods:**
- `get_device_info()`: Returns memory, cores, and thermal state.
- `measure_kernel(kernel_func, *args)`: Returns execution time and temperature.

## Visualizations

### `sigma_c.visualization`
- `plot_convergence(result, ...)`
- `plot_pareto_frontier(result, ...)`
- `plot_landscape(optimizer, param_space, ...)`
