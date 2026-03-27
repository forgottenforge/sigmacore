# Sigma-C Framework v3.0.0 -- Complete API Reference

## Table of Contents

1. [Core Modules](#core-modules)
2. [Domain Adapters (New in v3.0)](#domain-adapters-new-in-v30)
3. [Domain Adapters (Existing)](#domain-adapters-existing)
4. [Physics Validation](#physics-validation)
5. [Optimization](#optimization)
6. [Extended Features](#beyond-paper-features)
7. [Utilities](#utilities)

---

## Core Modules

### `sigma_c.core.engine`

**Purpose**: Core susceptibility calculation engine (C++ accelerated with Python fallback).

#### `Engine.compute_susceptibility(epsilon, observable, kernel_sigma, derivative_method, validate)`

Compute susceptibility, $\sigma_c$, and peak clarity $\kappa$.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | `np.ndarray` | -- | Scale parameter values |
| `observable` | `np.ndarray` | -- | Measured observable values |
| `kernel_sigma` | `float` | `0.6` | Gaussian smoothing width |
| `derivative_method` | `str` | `'gaussian'` | `'gaussian'`, `'savitzky_golay'`, `'spline'`, `'gp'`, or `'auto'` |
| `validate` | `bool` | `False` | Include peak significance and quality metrics |

**Returns**: `dict` with `sigma_c`, `kappa`, `chi_max`, `chi` array, and optionally `validation`.

---

### `sigma_c.core.base`

**Purpose**: Abstract base class for all domain adapters.

#### `SigmaCAdapter(config: Optional[Dict] = None)`

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | *Abstract*. Domain-specific observable |
| `compute_susceptibility` | `(epsilon, observable, kernel_sigma) -> Dict` | Compute chi and sigma_c |
| `diagnose` | `(data=None, **kwargs) -> Dict` | Universal diagnostics entry point |
| `validate` | `(data=None, **kwargs) -> Dict[str, bool]` | Validate technique requirements |
| `explain` | `(result: Dict, **kwargs) -> str` | Human-readable explanation |

---

### `sigma_c.core.contraction`

**Purpose**: Contraction geometry: defect, drift, classification, and countdown decomposition.

#### Primitive Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `v2` | `(n: int) -> int` | 2-adic valuation of n |
| `odd_part` | `(n: int) -> int` | Odd part: n / 2^v2(n) |
| `single_step_map` | `(n: int, q: int = 3, c: int = 1) -> int` | Map n -> odd(qn + c) |
| `cycle_map` | `(n: int) -> int` | Collatz cycle map F(n) |
| `embedding_depth` | `(n: int) -> int` | Embedding depth v2(n + 1) |

#### Analysis Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_contraction_defect` | `(f: Callable, M: int) -> float` | D_M = \|S_M\| / \|f(S_M)\| |
| `compute_drift` | `(f: Callable, M: int) -> float` | Geometric mean growth gamma_M |
| `compute_drift_v2_sum` | `(q: int, M: int) -> float` | Drift via v2-sum (approaches q/4) |
| `classify_map` | `(D: float, gamma: float, has_cycles: bool = False, D_threshold: float = 1.0, gamma_margin: float = 0.05) -> Dict` | Predict behavior from (D, gamma) |
| `sweep_modular_resolution` | `(f: Callable, M_range: range, compute_gamma: bool = True) -> List[Dict]` | Sweep D_M, gamma_M across resolutions |
| `contraction_principle_check` | `(D_values: List[float], gamma_values: List[float], D_min: float = 1.0, gamma_threshold: float = 1.0) -> Dict` | Check stabilization and convergence |
| `countdown_decomposition` | `(n: int, max_steps: int = 1000) -> List[Dict]` | Decompose orbit into countdown/reset phases |
| `deterministic_trajectory` | `(n: int) -> List[int]` | Closed-form countdown trajectory |

#### Constants

| Name | Type | Description |
|------|------|-------------|
| `TWELVE_MAP_PREDICTIONS` | `List[Dict]` | Reference table of 12 qn+c map predictions |

---

### `sigma_c.core.classification`

**Purpose**: Four-type classification of operations by injectivity structure.

#### `MapType` Enum

| Value | String | Description |
|-------|--------|-------------|
| `DISSIPATIVE` | `"D"` | D > 1, non-injective contraction |
| `OVERSATURATED` | `"O"` | Growing pre-image count |
| `SYMMETRIC` | `"S"` | D = 1, bijective with symmetry |
| `REVERSIBLE` | `"R"` | D = 1, bijective preservation |

#### Classification Function

| Function | Signature | Returns |
|----------|-----------|---------|
| `classify_operation` | `(D: float, gamma: Optional[float] = None, has_cycles: bool = False, is_bijective: Optional[bool] = None, has_symmetry: bool = False, has_growing_preimage: bool = False) -> MapType` | MapType enum |

#### Type Analysis Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `analyze_type_d` | `(D: float, gamma: float, has_cycles: bool = False) -> TypeDResult` | Prediction with confidence |
| `analyze_type_o` | `(counts: np.ndarray, predictions: np.ndarray, n_range: Optional[np.ndarray] = None) -> TypeOResult` | Oversaturation ratio |
| `analyze_type_s` | `(deviations: np.ndarray, reference_variance: float = 0.178) -> TypeSResult` | Symmetry deviation |
| `analyze_type_r` | `(f_values: np.ndarray, g_orbits: np.ndarray) -> TypeRResult` | Orbit preservation check |

#### Result Classes

**`TypeDResult`**: `D`, `gamma`, `has_cycles`, `prediction`, `details`, `to_dict()`

**`TypeOResult`**: `oversaturation_ratio`, `min_count`, `reference_prediction`, `details`, `to_dict()`

**`TypeSResult`**: `symmetry_deviation`, `constraint_tightness`, `details`, `to_dict()`

**`TypeRResult`**: `is_bijective`, `orbits_preserved`, `conserved_quantity`, `details`, `to_dict()`

---

### `sigma_c.core.derivatives`

**Purpose**: Extended derivative estimation for noisy observables.

| Function | Signature | Description |
|----------|-----------|-------------|
| `savitzky_golay_derivative` | `(x: np.ndarray, y: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray` | Savitzky-Golay filter derivative |
| `spline_derivative` | `(x: np.ndarray, y: np.ndarray, smoothing_factor: Optional[float] = None) -> np.ndarray` | Regularized cubic spline derivative |
| `gp_regression_derivative` | `(x: np.ndarray, y: np.ndarray, length_scale: Optional[float] = None, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]` | GP regression derivative with uncertainty |
| `select_best_method` | `(x: np.ndarray, y: np.ndarray) -> str` | Auto-select method based on data properties |
| `compute_derivative` | `(x: np.ndarray, y: np.ndarray, method: str = 'auto', **kwargs) -> Dict` | Unified interface for all methods |

**`compute_derivative` returns**: `{'derivative': np.ndarray, 'method_used': str, 'uncertainty': np.ndarray (GP only)}`

---

### `sigma_c.core.validation`

**Purpose**: Formal statistical validation of sigma_c measurements.

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_boundary_conditions` | `(observable_values: np.ndarray, epsilon_values: np.ndarray) -> Dict` | Check existence proof for sigma_c |
| `permutation_test` | `(epsilon: np.ndarray, observable: np.ndarray, n_permutations: int = 10000, kernel_sigma: float = 0.6) -> Dict` | Permutation test for peak significance |
| `peak_clarity_test` | `(kappa: float, kappa_min: float = 3.0) -> Dict` | Test peak clarity against threshold |
| `fisher_information_bound` | `(epsilon: np.ndarray, observable: np.ndarray, observable_variance: Optional[np.ndarray] = None) -> Dict` | Fisher information and Cramer-Rao bound |
| `observable_quality_score` | `(data: np.ndarray, epsilon: np.ndarray, kernel_sigma: float = 0.6) -> Dict` | Decision-tree quality assessment (0--1 score) |

**`permutation_test` returns**: `p_value`, `observed_kappa`, `null_mean`, `null_std`, `null_95th`, `significant`.

**`observable_quality_score` returns**: `score`, `passes`, `criteria` (dict of individual checks), `recommendation`.

---

## Domain Adapters (New in v3.0)

### `sigma_c.adapters.number_theory.NumberTheoryAdapter`

**Purpose**: Contraction geometry analysis of Collatz-type maps.

```python
NumberTheoryAdapter(map_type: str = 'collatz', q: int = 3, c: int = 1, config: Optional[Dict] = None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | Normalized embedding depth |
| `compute_D_M` | `(M: int) -> float` | Contraction defect at resolution M |
| `compute_gamma_M` | `(M: int) -> float` | Drift at resolution M |
| `sweep_resolution` | `(M_range: range = range(4, 17)) -> List[Dict]` | Sweep (D, gamma) across resolutions |
| `predict_behavior` | `() -> Dict` | Predict convergent/divergent from (D, gamma) at M=14 |
| `verify_prediction` | `(n_samples: int = 1000, max_steps: int = 10000) -> Dict` | Empirical trajectory verification |
| `analyze_countdown` | `(n: int) -> Dict` | Countdown/reset phase decomposition |
| `verify_reset_distribution` | `(M: int = 12, n_samples: int = 10000) -> Dict` | Chi-squared test against Geo(1/2) |
| `get_twelve_map_table` | `() -> List[Dict]` | Pre-computed twelve-map reference table |
| `verify_twelve_predictions` | `(M: int = 12) -> Dict` | Verify all 12 predictions computationally |

**Class Attributes**: `D_M_REFERENCE`, `GAMMA_REFERENCE` (reference values for validation).

---

### `sigma_c.adapters.protein.ProteinAdapter`

**Purpose**: Protein stability analysis via contraction index sigma = D * gamma.

```python
ProteinAdapter(protein_name: Optional[str] = None, N: Optional[int] = None, T: float = 310.0, delta_G_wt: Optional[float] = None, config: Optional[Dict] = None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | Compute sigma from mutation dict or delta_G float |
| `sigma_product` | `(D: float, gamma: float) -> float` | sigma = D * gamma |
| `sigma_thermodynamic` | `(delta_G: float, N: int, T: Optional[float] = None) -> float` | exp(-delta_G / (N * R * T)) |
| `sigma_mutation` | `(delta_delta_G: float, N: int, T: Optional[float] = None) -> float` | exp(ddG / (N * R * T)) |
| `sigma_drift` | `(sigma_baseline: float, age: float, rate: float = 0.003) -> float` | Age-dependent sigma |
| `predict_onset` | `(sigma_baseline: float, rate: float = 0.003) -> float` | Predicted onset age (years) |
| `onset_envelope` | `(sigma_baseline: float, rate_range: Tuple[float, float] = (0.02, 0.05)) -> Tuple[float, float]` | (earliest, latest) onset |
| `classify_mechanism` | `(protein_data: dict) -> dict` | Classify: IDP, stability_driven, GOF, templated |
| `validate_scope` | `(protein_data: dict) -> dict` | Check applicability of sigma analysis |
| `analyze_protein` | `(mutations: List[Dict]) -> dict` | Full analysis with Spearman correlations |

**Class Attributes**: `R` (gas constant), `TTR_MUTATIONS`, `TTR_PARAMS`, `LYZ_MUTATIONS`, `LYZ_PARAMS`, `GSN_MUTATIONS`, `GSN_PARAMS`, `SOD1_MUTATIONS`, `SOD1_PARAMS`, `PRNP_MUTATIONS`, `PRNP_PARAMS`, `REFERENCE_PROTEINS`.

### DualBasinModel

Simplified Monte Carlo dual-basin Go model for protein folding.

```python
from sigma_c.adapters.protein import DualBasinModel
model = DualBasinModel(N=30, S=8, contacts=12)
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `N=30, S=8, contacts=12, epsilon_nat=1.0, epsilon_amy=0.5` | — | Initialize dual-basin model |
| `energy` | `config: ndarray, alpha: float` | `float` | Compute E(alpha) = (1-alpha)*E_nat + alpha*E_amy |
| `native_contacts_fraction` | `config: ndarray` | `float` | Q_nat: fraction of native contacts formed |
| `simulate` | `alpha, n_steps=3000, n_trials=10, temperature=1.0` | `Dict` | Run MC simulation, returns D, gamma, sigma, Q_nat |
| `sweep_alpha` | `alpha_range=None, **kwargs` | `List[Dict]` | Sweep alpha, compute sigma at each point |
| `find_critical_alpha` | `**kwargs` | `float` | Find alpha where sigma crosses 1.0 |

---

### `sigma_c.adapters.linguistics.LinguisticsAdapter`

**Purpose**: Etymological depth and semantic change analysis.

```python
LinguisticsAdapter(language: str = 'english', config: Optional[Dict] = None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | Semantic change for a word string or data dict |
| `etymological_depth` | `(word: str) -> Optional[int]` | ED value (1--5) or None |
| `semantic_change` | `(word: str) -> Optional[float]` | Pre-computed semantic change magnitude |
| `correlation_analysis` | `(ed_values: Optional[np.ndarray] = None, change_values: Optional[np.ndarray] = None) -> Dict` | Pearson and Spearman correlations |
| `fixed_point_test` | `(ed_values: Optional[np.ndarray] = None, change_values: Optional[np.ndarray] = None) -> Dict` | Welch t-test: ED=1 vs ED>1 |
| `mediation_analysis` | `(ed_values, freq_values, change_values) -> Dict` | Baron-Kenny mediation (ED -> Freq -> Change) |
| `transparency_effect` | `(change_values: Optional[np.ndarray] = None, transparency_labels: Optional[List[bool]] = None) -> Dict` | Transparent vs opaque comparison (ED >= 2) |
| `german_anchor_test` | `() -> Dict` | ANOVA across German P/T/O categories |
| `run_full_analysis` | `() -> Dict` | Run all analyses, return combined results |
| `orthogonal_procrustes` | `W_t1: ndarray, W_t2: ndarray, n_anchors=5000` | `ndarray` | Compute Procrustes rotation matrix R |
| `aligned_cosine_distance` | `v1: ndarray, v2: ndarray, R: ndarray` | `float` | Cosine distance after alignment |
| `three_regime_model` | — | `Dict` | Primes/Opaque/Transparent regime ANOVA |
| `french_replication` | — | `Dict` | French cross-linguistic replication |
| `cross_linguistic_comparison` | — | `Dict` | EN/DE/FR comparison summary |

**Class Attributes**: `ED1_WORDS` through `ED5_WORDS` (English word lists), `TRANSPARENT_WORDS`, `GERMAN_PRIMES`, `GERMAN_TRANSPARENT`, `GERMAN_OPAQUE`.

---

## Domain Adapters (Existing)

### `sigma_c.adapters.quantum.QuantumAdapter`

Quantum circuit optimization balancing fidelity against noise resilience.

| Method | Description |
|--------|-------------|
| `get_observable(data, **kwargs)` | Circuit fidelity observable |
| `create_grover_with_noise(n_qubits, epsilon)` | Create Grover circuit with noise |
| `set_noise_model(depolarizing_prob, dephasing_prob, readout_error)` | Configure noise parameters |

### `sigma_c.adapters.gpu.GPUAdapter`

GPU performance optimization balancing throughput against thermal stability.

### `sigma_c.adapters.financial.FinancialAdapter`

Financial strategy optimization balancing returns against market risk.

### `sigma_c.adapters.climate.ClimateAdapter`

Climate boundary detection (mesoscale/synoptic transitions).

### `sigma_c.adapters.seismic.SeismicAdapter`

Seismic analysis (Gutenberg-Richter b-value).

### `sigma_c.adapters.magnetic.MagneticAdapter`

Magnetic phase transition analysis (critical exponents).

### `sigma_c.adapters.edge.EdgeAdapter`

Edge computing power efficiency optimization.

### `sigma_c.adapters.llm_cost.LLMCostAdapter`

LLM model selection via cost-safety Pareto frontier.

### `sigma_c.adapters.ml.MLAdapter`

Machine learning robustness optimization.

---

## Physics Validation

### `sigma_c.physics.number_theory.RigorousNumberTheorySigmaC`

Validates sigma_c values against number-theoretic bounds.

| Method | Signature | Description |
|--------|-----------|-------------|
| `check_theoretical_bounds` | `(data: Dict) -> Dict` | Verify D_M >= 4/3 and compute gamma = q/4 |
| `check_scaling_laws` | `(data, param_range: List[float]) -> Dict` | Verify D_M stabilization with resolution |
| `quantify_resource` | `(data) -> float` | Return modular resolution M |
| `validate_sigma_c` | `(sigma_c_value: float, context: Dict) -> Dict` | Check consistency of D_M, gamma, sigma_c |

### `sigma_c.physics.protein.RigorousProteinSigmaC`

Validates sigma_c values against protein thermodynamic constraints.

| Method | Signature | Description |
|--------|-----------|-------------|
| `check_theoretical_bounds` | `(data: Dict) -> Dict` | Verify sigma(T_m) = 1 and sigma(310K) < 1 |
| `check_scaling_laws` | `(data, param_range: List[float]) -> Dict` | Verify monotonic sigma vs delta_delta_G |
| `quantify_resource` | `(data) -> float` | Return chain length N |
| `validate_sigma_c` | `(sigma_c_value: float, context: Dict) -> Dict` | Check consistency with biophysical mechanism |

### `sigma_c.physics.rigorous.RigorousTheoreticalCheck` (Base Class)

Abstract base for all physics validation modules.

| Method | Description |
|--------|-------------|
| `check_theoretical_bounds(data, **kwargs)` | *Abstract*. Domain-specific bound checking |
| `check_scaling_laws(data, param_range, **kwargs)` | *Abstract*. Scaling law verification |
| `quantify_resource(data)` | *Abstract*. Identify fundamental resource |
| `validate_sigma_c(sigma_c_value, context)` | *Abstract*. Full consistency check |

---

## Optimization

### `sigma_c.optimization.universal.UniversalOptimizer` (Base Class)

Abstract base for all domain-specific optimizers.

```python
UniversalOptimizer(performance_weight: float = 0.7, stability_weight: float = 0.3)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `optimize` | `(system, param_space: Dict[str, List], strategy: str = 'brute_force', callbacks: Optional[List] = None) -> OptimizationResult` | Main optimization entry point |
| `calculate_score` | `(performance: float, stability: float) -> float` | Composite score: w_p * P + w_s * S |
| `_evaluate_performance` | `(system, params) -> float` | *Abstract*. Domain performance metric |
| `_evaluate_stability` | `(system, params) -> float` | *Abstract*. Domain stability metric |
| `_apply_params` | `(system, params) -> Any` | *Abstract*. Apply parameters to system |

### `sigma_c.optimization.universal.OptimizationResult`

Dataclass returned by all optimizers.

| Field | Type | Description |
|-------|------|-------------|
| `optimal_params` | `Dict[str, Any]` | Best parameter combination |
| `score` | `float` | Composite score at optimum |
| `history` | `List[Dict]` | Full evaluation history |
| `sigma_c_before` | `float` | Stability before optimization |
| `sigma_c_after` | `float` | Stability after optimization |
| `performance_metric_name` | `str` | Name of performance metric |
| `performance_before` | `float` | Performance before optimization |
| `performance_after` | `float` | Performance after optimization |
| `strategy_used` | `str` | Optimization strategy used |

### `sigma_c.optimization.protein.ProteinInterventionOptimizer`

Finds minimal D * gamma < 1 interventions for protein stabilization.

```python
ProteinInterventionOptimizer(performance_weight: float = 0.7, stability_weight: float = 0.3, sigma_target: float = 1.0)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `optimize_intervention` | `(system, D_values: List[int], gamma_values: List[float], strategy: str = 'brute_force', callbacks: Optional[List] = None) -> OptimizationResult` | Find best D * gamma < 1 combination |
| `_evaluate_performance` | `(system, params) -> float` | Native contact fraction Q_nat |
| `_evaluate_stability` | `(system, params) -> float` | Distance from sigma = 1 threshold |

---

## Extended Features

### `sigma_c.beyond.information`

**Purpose**: Information-theoretic connections to contraction geometry.

| Function | Signature | Description |
|----------|-----------|-------------|
| `bits_lost` | `(D: float) -> float` | Information loss: log2(D) bits per step |
| `landauer_cost` | `(D: float, T: float = 300.0) -> float` | Minimum erasure energy (Joules) |
| `entropy_production_rate` | `(D: float, steps_per_second: float, T: float = 300.0) -> float` | Entropy production (J/(K*s)) |
| `information_summary` | `(D: float, gamma: float, T: float = 300.0, steps_per_second: float = 1.0) -> Dict` | Complete thermodynamic summary |

**Constants**: `K_B = 1.380649e-23` (Boltzmann constant, J/K), `LN2 = ln(2)`.

**`information_summary` returns**: `bits_lost_per_step`, `landauer_cost_J`, `landauer_cost_eV`, `entropy_production_rate`, `D`, `gamma`, `sigma_product`, `net_contraction`, `interpretation`.

### `sigma_c.beyond.coupling`

**Purpose**: Cross-domain coupling analysis.

| Method | Description |
|--------|-------------|
| `CouplingAnalyzer.compute_coupling_matrix(domains)` | Compute inter-domain coupling strengths |
| `CouplingAnalyzer.eigenvalue_stability(coupling_matrix)` | Assess stability from eigenvalues |
| `CouplingAnalyzer.cascade_risk(coupling_matrix)` | Quantify cascade propagation risk |

### `sigma_c.beyond.self_opt`

**Purpose**: Self-optimization via genetic algorithms.

| Method | Description |
|--------|-------------|
| `SelfOptimizer.evolve(population, generations)` | Genetic algorithm optimization |
| `SelfOptimizer.tournament_select(population, k)` | Tournament selection |

---

## Utilities

### `sigma_c.core.orchestrator.Universe`

**Purpose**: Unified factory for all domain adapters.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Universe.quantum` | `(device: str = 'simulator', **kwargs) -> SigmaCAdapter` | Create QuantumAdapter |
| `Universe.gpu` | `(**kwargs) -> SigmaCAdapter` | Create GPUAdapter |
| `Universe.finance` | `(**kwargs) -> SigmaCAdapter` | Create FinancialAdapter |
| `Universe.climate` | `(**kwargs) -> SigmaCAdapter` | Create ClimateAdapter |
| `Universe.seismic` | `(**kwargs) -> SigmaCAdapter` | Create SeismicAdapter |
| `Universe.magnetic` | `(**kwargs) -> SigmaCAdapter` | Create MagneticAdapter |
| `Universe.number_theory` | `(map_type='collatz', **kwargs) -> NumberTheoryAdapter` | Create NumberTheoryAdapter |
| `Universe.protein` | `(protein_name=None, **kwargs) -> ProteinAdapter` | Create ProteinAdapter |
| `Universe.linguistics` | `(language='english', **kwargs) -> LinguisticsAdapter` | Create LinguisticsAdapter |

### `sigma_c.adapters.factory.AdapterFactory`

| Method | Signature | Description |
|--------|-----------|-------------|
| `AdapterFactory.create` | `(domain: str, **kwargs) -> SigmaCAdapter` | Create adapter by domain name string |

---

## Version Information

```python
import sigma_c
print(sigma_c.__version__)  # "3.0.0"
```

**License**: AGPL-3.0-or-later OR Commercial
**Copyright**: 2025-2026 ForgottenForge.xyz
