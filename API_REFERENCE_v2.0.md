# Sigma-C Framework v2.0.0 - Complete API Reference

## Table of Contents
1. [Core Modules](#core-modules)
2. [Domain Adapters](#domain-adapters)
3. [Optimization](#optimization)
4. [Physics Validation](#physics-validation)
5. [Beyond Paper Features](#beyond-paper)
6. [Examples](#examples)

---

## Core Modules

### `sigma_c.core.engine`
**Purpose**: Core susceptibility calculation engine

#### `Engine.compute_susceptibility(epsilon, observable)`
Computes critical susceptibility $\chi_c$ from parameter sweep data.

**Parameters**:
- `epsilon` (np.ndarray): Control parameter values
- `observable` (np.ndarray): Measured observable values

**Returns**: `dict` with keys:
- `sigma_c` (float): Critical point location
- `kappa` (float): Peak sharpness
- `chi_max` (float): Maximum susceptibility

**Mathematical Foundation**:
$$\chi = \frac{\partial \langle O \rangle}{\partial \epsilon}$$

---

### `sigma_c.core.discovery`
**Purpose**: Automatic observable discovery and multi-scale analysis

#### `ObservableDiscovery.find_optimal_observable(data, candidate_names, method='gradient')`
Identifies the optimal order parameter from candidate observables.

**Parameters**:
- `data` (np.ndarray): Shape (n_samples, n_features)
- `candidate_names` (List[str]): Names of candidate observables
- `method` (str): 'gradient', 'entropy', or 'pca'

**Returns**: `ObservableCandidate` with:
- `name` (str): Best observable name
- `score` (float): Quality metric
- `index` (int): Column index in data

**Methods**:
- **Gradient**: Maximizes $\|\nabla O\|$
- **Entropy**: Minimizes $H(O) = -\sum p_i \log p_i$
- **PCA**: First principal component

#### `MultiScaleAnalysis.compute_susceptibility_spectrum(parameter, observable, scales=None)`
Performs continuous wavelet transform to detect criticality across scales.

**Parameters**:
- `parameter` (np.ndarray): Control parameter
- `observable` (np.ndarray): Observable values
- `scales` (np.ndarray, optional): Wavelet scales

**Returns**: `dict` with:
- `scales` (np.ndarray): Scale values
- `susceptibility` (np.ndarray): $\chi$ at each scale
- `wavelet_coefficients` (np.ndarray): Full CWT output

**Wavelet**: Ricker (Mexican hat) $\psi(t) = (1-t^2)e^{-t^2/2}$

---

### `sigma_c.core.control`
**Purpose**: Active control and real-time streaming

#### `AdaptiveController(target_sigma, kp=1.0, ki=0.1, kd=0.05)`
PID controller to maintain system at critical point.

**Parameters**:
- `target_sigma` (float): Desired $\sigma_c$ setpoint
- `kp, ki, kd` (float): PID gains

**Methods**:
- `compute_correction(current_sigma)`: Returns control signal
- `reset()`: Clears integral/derivative history

**Control Law**:
$$u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de}{dt}$$

#### `StreamingSigmaC(window_size=100)`
O(1) incremental susceptibility calculation.

**Methods**:
- `update(epsilon, observable)`: Add new data point
- `get_current_sigma_c()`: Returns current estimate

**Algorithm**: Sliding window with Welford's online variance

---

## Domain Adapters

### `sigma_c.adapters.quantum.QuantumAdapter`
**Purpose**: Quantum circuit optimization

#### Key Methods

**`create_grover_with_noise(n_qubits=2, epsilon=0.0, idle_frac=0.0)`**
Creates Grover search circuit with configurable noise.

**Hardware Compatibility**: CZ-gate native (Rigetti Ankaa-3, IQM)

**`analyze_depth_scaling(circuit_factory, max_depth=20)`**
Validates $\sigma_c \sim D^{1-\alpha}$ scaling.

**Returns**:
- `alpha` (float): Scaling exponent (expected ~0.7)
- `raw_depths`, `raw_sigma_c`: Measurement data

**`analyze_idle_sensitivity(circuit_factory)`**
Measures $\frac{d\sigma_c}{df_{idle}}$ (expected: -0.133 ± 0.077)

**`compute_fisher_information(epsilon, observables)`**
Calculates Fisher Information for peak clarity $\kappa$.

**`compute_correlation_length(n_qubits, coupling_map)`**
Estimates quantum correlation length $\xi_c$ using graph diameter.

---

### `sigma_c.adapters.gpu.GPUAdapter`
**Purpose**: GPU kernel optimization

#### Key Methods

**`detect_cache_transitions(working_set_sizes)`**
Detects L1/L2/L3 cache boundaries via multi-scale analysis.

**Expected Values**:
- L1: $\sigma_c = 0.023$
- L2: $\sigma_c = 0.072$
- L3: $\sigma_c = 0.241$

**`analyze_roofline()`**
Computes Roofline Ridge Point:
$$AI_{ridge} = \frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}}$$

**`predict_thermal_throttling(current_temp, base_temp=25.0)`**
Power-law scaling: $\sigma_c(T) = \sigma_0 \cdot (T/T_0)^{-\beta}$

---

### `sigma_c.adapters.financial.FinancialAdapter`
**Purpose**: Financial market analysis

#### Key Methods

**`compute_hurst_exponent(prices)`**
R/S analysis for market regime classification.

**Interpretation**:
- $H < 0.5$: Mean-reverting
- $H = 0.5$: Random walk
- $H > 0.5$: Trending

**`analyze_volatility_clustering(returns)`**
GARCH(1,1) model: $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$

**Returns**:
- `persistence` (float): $\alpha + \beta$
- `sigma_c` (float): Volatility criticality

**`analyze_order_flow(order_imbalance)`**
Diffusion exponent from MSD: $\langle (\Delta OFI)^2 \rangle \sim t^{2H}$

---

### `sigma_c.adapters.edge.EdgeAdapter`
**Purpose**: Edge computing / IoT optimization

#### Key Methods

**`analyze_power_efficiency(frequency, power, performance)`**
Finds DVFS knee point where $\frac{d\text{Perf}}{dP} < \theta$.

**Returns**:
- `critical_frequency` (float): Optimal operating point
- `max_efficiency` (float): Peak Perf/Watt
- `sigma_c_power` (float): Normalized power at critical point

---

### `sigma_c.adapters.llm_cost.LLMCostAdapter`
**Purpose**: LLM model selection

#### Key Methods

**`analyze_cost_safety(models)`**
Pareto frontier analysis for cost vs. hallucination rate.

**Input**: List of `{'name', 'cost', 'hallucination_rate'}`

**Returns**:
- `best_model` (str): Optimal model name
- `optimal_cost` (float): Cost per query
- `safety_score` (float): $1 - \text{hallucination\_rate}$
- `sigma_c` (float): Stability metric

---

## Optimization

### `sigma_c.optimization.universal.UniversalOptimizer`
**Purpose**: Abstract base for domain-specific optimizers

#### Key Methods

**`optimize(system, param_space, strategy='brute_force')`**
Main optimization loop.

**Parameters**:
- `system`: Domain-specific system (circuit, kernel, model)
- `param_space` (dict): `{param_name: [values]}`
- `strategy` (str): 'brute_force' or 'gradient_descent'

**Returns**: `OptimizationResult` with:
- `optimal_params` (dict)
- `score` (float): Composite metric
- `sigma_c_before`, `sigma_c_after` (float)
- `performance_before`, `performance_after` (float)
- `history` (list): Full optimization trajectory

**Scoring Function**:
$$S = w_p \cdot P + w_s \cdot \sigma_c$$

---

## Beyond Paper Features

### `sigma_c.beyond.coupling.CouplingMatrix`
**Purpose**: Cross-domain cascade risk detection

#### Key Methods

**`set_coupling(source, target, strength)`**
Defines interaction $J_{ij}$ between domains.

**`analyze_stability()`**
Eigenvalue analysis of coupling matrix.

**Returns**:
- `max_eigenvalue` (float): Largest $|\lambda|$
- `meta_sigma_c` (float): $1/(|\lambda_{max}| + \epsilon)$
- `stability` (str): 'stable', 'critical', or 'unstable'
- `cascade_risk` (float): 0-1 scale

**Stability Criterion**: $|\lambda_{max}| < 1$

**`simulate_cascade(initial_perturbation, steps=10)`**
Propagates perturbation: $\mathbf{x}(t+1) = J \cdot \mathbf{x}(t)$

---

### `sigma_c.beyond.self_opt.GeneticOptimizer`
**Purpose**: Evolutionary parameter optimization

#### Key Methods

**`evolve_parameters(fitness_func, bounds, generations=20)`**
Genetic algorithm for parameter search.

**Parameters**:
- `fitness_func` (Callable): $f: \mathbb{R}^n \to \mathbb{R}$
- `bounds` (List[Tuple]): Parameter bounds
- `generations` (int): Evolution iterations

**Returns**:
- `best_parameters` (list)
- `best_fitness` (float)
- `history` (list): Fitness evolution

**Operators**: Tournament selection, single-point crossover, Gaussian mutation

---

## Mathematical Foundations

### Susceptibility Definition
$$\chi(\epsilon) = \frac{\partial \langle O \rangle}{\partial \epsilon} \bigg|_{\epsilon = \epsilon_c}$$

### Critical Point Detection
Peak finding in $\chi(\epsilon)$ with Savitzky-Golay smoothing.

### Finite-Size Scaling
$$\sigma_c(L) = \sigma_c(\infty) + \frac{A}{L^{1/\nu}}$$

### Fisher Information
$$I(\epsilon) = \mathbb{E}\left[\left(\frac{\partial \log p(x|\epsilon)}{\partial \epsilon}\right)^2\right]$$

---

## Usage Examples

### Quantum Circuit Optimization
```python
from sigma_c.adapters.quantum import QuantumAdapter
from sigma_c.optimization.quantum import BalancedQuantumOptimizer

adapter = QuantumAdapter()
optimizer = BalancedQuantumOptimizer(adapter)

def circuit_factory(**params):
    return adapter.create_grover_with_noise(**params)

result = optimizer.optimize_circuit(
    circuit_factory,
    param_space={
        'epsilon': [0.0, 0.05, 0.1],
        'idle_frac': [0.0, 0.1, 0.2]
    }
)

print(f"Optimal: {result.optimal_params}")
print(f"Sigma_c: {result.sigma_c_after:.4f}")
```

### GPU Cache Detection
```python
from sigma_c.adapters.gpu import GPUAdapter

gpu = GPUAdapter()
sizes = [64*1024, 1024*1024, 10*1024*1024]  # 64KB, 1MB, 10MB
transitions = gpu.detect_cache_transitions(sizes)

print(f"L1: {transitions['L1_transition']}")
print(f"L2: {transitions['L2_transition']}")
print(f"L3: {transitions['L3_transition']}")
```

### Cross-Domain Coupling
```python
from sigma_c.beyond.coupling import CouplingMatrix

cm = CouplingMatrix(['Quantum', 'GPU', 'Financial'])
cm.set_coupling('Quantum', 'GPU', 0.5)
cm.set_coupling('GPU', 'Financial', 0.3)

stability = cm.analyze_stability()
print(f"Cascade Risk: {stability['cascade_risk']:.2%}")
```

---

## Performance Considerations

### Computational Complexity
- `compute_susceptibility`: O(n log n) via FFT
- `MultiScaleAnalysis`: O(n × m) for m scales
- `UniversalOptimizer.brute_force`: O(k^d) for d parameters, k values each
- `StreamingSigmaC`: O(1) per update

### Memory Requirements
- Wavelet transform: O(n × m) for n samples, m scales
- Optimization history: O(k^d) for full trajectory

---

## References

1. **Quantum Paper**: "Criticality-Based Quantum Advantage" (arXiv:XXXX.XXXXX)
2. **GPU Paper**: "Cache Hierarchy Criticality" (IEEE TPDS 2024)
3. **Financial Paper**: "Market Microstructure Criticality" (J. Finance 2023)

---

## License

Copyright (c) 2025 ForgottenForge.xyz

Licensed under AGPL-3.0-or-later OR Commercial License.
For commercial licensing, contact: info@forgottenforge.xyz
