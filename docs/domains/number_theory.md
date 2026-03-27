# Number Theory Domain

## Overview

The Number Theory domain applies contraction geometry to Collatz-type maps of the form $n \to \text{odd}(qn + c)$. By computing the **contraction defect** $D_M$ and **drift** $\gamma_M$ at modular resolution $M$, the framework predicts whether a given map is convergent, divergent, or converges to cycles -- without requiring trajectory enumeration.

The central insight is that non-injectivity of the map on residue classes (captured by $D > 1$) combined with average contraction (captured by $\gamma < 1$) creates a structural compression that forces orbits downward.

## Key Concepts

### Contraction Defect ($D_M$)

The contraction defect measures how much a map compresses the space of odd residues modulo $2^M$:

$$D_M = \frac{|S_M|}{|f(S_M)|}$$

where $S_M$ is the set of odd residues in $\{1, 3, \ldots, 2^M - 1\}$ and $f(S_M)$ is the image set. A value $D_M > 1$ indicates structural non-injectivity: multiple inputs map to the same output residue class.

For the standard Collatz cycle map, $D_M \approx 2.06$ at moderate resolutions (M = 10--16), meaning roughly half the residue classes are "lost" at each step.

### Drift ($\gamma_M$)

The drift quantifies the average multiplicative growth per map application:

$$\gamma_M = 2^{\frac{1}{|S_M|} \sum_{n \in S_M} \log_2 \frac{f(n)}{n}}$$

For the single-step map $n \to \text{odd}(qn + c)$, the drift converges to $\gamma = q/4$ as $M \to \infty$. This is independent of the additive constant $c$.

| Map | $q$ | $\gamma$ | Behavior |
|-----|-----|----------|----------|
| $3n+1$ (single-step) | 3 | 3/4 = 0.75 | Convergent |
| $5n+1$ | 5 | 5/4 = 1.25 | Divergent |
| $7n+1$ | 7 | 7/4 = 1.75 | Divergent |

### Embedding Depth

The embedding depth $\text{ed}(n) = v_2(n+1)$ counts the trailing ones in the binary representation of $n$. It measures how deeply $n$ is "embedded" in the 2-adic structure and determines the length of the deterministic countdown phase.

### Countdown Theorem

Each Collatz orbit decomposes into alternating phases:

1. **Countdown phase** (deterministic): When $\text{ed}(n) \geq 2$, the orbit follows a closed-form trajectory $f^j(n) = 3^j \cdot 2^{k-j} \cdot m - 1$ for $j = 0, \ldots, k-1$, where $k = \text{ed}(n)$ and $m = (n+1)/2^k$.

2. **Reset event** (stochastic): When $\text{ed}(n) = 1$, a single application of the map produces a new embedding depth drawn from a $\text{Geo}(1/2)$ distribution.

This decomposition reveals that the apparent complexity of Collatz dynamics arises entirely from the reset events.

## API Reference

### `NumberTheoryAdapter`

```python
from sigma_c.adapters.number_theory import NumberTheoryAdapter
```

**Constructor**:
```python
NumberTheoryAdapter(
    map_type: str = 'collatz',   # 'collatz', 'collatz_single', 'collatz_cycle', 'custom'
    q: int = 3,                  # Multiplier for qn+c maps
    c: int = 1,                  # Additive constant
    config: Optional[Dict] = None
)
```

**Core Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | Compute normalized embedding depth from integer(s) |
| `compute_D_M` | `(M: int) -> float` | Compute contraction defect at resolution M |
| `compute_gamma_M` | `(M: int) -> float` | Compute drift at resolution M |
| `sweep_resolution` | `(M_range: range) -> List[Dict]` | Compute (D, gamma) across resolutions |
| `predict_behavior` | `() -> Dict` | Predict convergent/divergent behavior from (D, gamma) |
| `verify_prediction` | `(n_samples: int, max_steps: int) -> Dict` | Empirically verify predicted behavior |
| `analyze_countdown` | `(n: int) -> Dict` | Decompose orbit into countdown/reset phases |
| `verify_reset_distribution` | `(M: int, n_samples: int) -> Dict` | Test reset depths against Geo(1/2) |
| `get_twelve_map_table` | `() -> List[Dict]` | Return the twelve-map predictions reference table |
| `verify_twelve_predictions` | `(M: int) -> Dict` | Verify all twelve map predictions computationally |

**Diagnostics Methods** (inherited from `SigmaCAdapter`):

| Method | Signature | Description |
|--------|-----------|-------------|
| `diagnose` | `(data=None, **kwargs) -> Dict` | Check map validity, D and gamma ranges |
| `validate` | `(data=None, **kwargs) -> Dict[str, bool]` | Validate technique requirements |
| `explain` | `(result: Dict, **kwargs) -> str` | Generate human-readable explanation |

### Contraction Module Functions

```python
from sigma_c.core.contraction import (
    v2, odd_part, single_step_map, cycle_map,
    embedding_depth, compute_contraction_defect,
    compute_drift, classify_map, sweep_modular_resolution,
    contraction_principle_check, countdown_decomposition,
    deterministic_trajectory
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `v2` | `(n: int) -> int` | 2-adic valuation of n |
| `odd_part` | `(n: int) -> int` | Extract odd part n / 2^v2(n) |
| `single_step_map` | `(n: int, q: int, c: int) -> int` | Single-step map odd(qn + c) |
| `cycle_map` | `(n: int) -> int` | Collatz cycle map (full countdown in one step) |
| `embedding_depth` | `(n: int) -> int` | Embedding depth v2(n + 1) |
| `compute_contraction_defect` | `(f, M: int) -> float` | Compute D_M for map f at resolution M |
| `compute_drift` | `(f, M: int) -> float` | Compute gamma_M for map f at resolution M |
| `classify_map` | `(D, gamma, has_cycles, ...) -> Dict` | Classify behavior from (D, gamma) |
| `contraction_principle_check` | `(D_values, gamma_values, ...) -> Dict` | Check stabilization of (D, gamma) |
| `countdown_decomposition` | `(n: int, max_steps: int) -> List[Dict]` | Decompose orbit into phases |
| `deterministic_trajectory` | `(n: int) -> List[int]` | Closed-form countdown trajectory |

## Quick Example

```python
from sigma_c.adapters.number_theory import NumberTheoryAdapter

adapter = NumberTheoryAdapter(map_type='collatz')
result = adapter.predict_behavior()
print(f"D={result['D']:.3f}, gamma={result['gamma']:.3f} -> {result['prediction']}")
```

## Usage Patterns

### Analyzing a Custom Map

To analyze any $qn+c$ map, use the `custom` map type:

```python
# Analyze the 5n+1 map (expected: divergent)
adapter = NumberTheoryAdapter(map_type='custom', q=5, c=1)
result = adapter.predict_behavior()
print(result['prediction'])  # 'divergent'

# Sweep across resolutions to see stabilization
sweep = adapter.sweep_resolution(range(4, 15))
for row in sweep:
    print(f"M={row['M']:2d}  D={row['D_M']:.4f}  gamma={row['gamma_M']:.4f}")
```

### Countdown Analysis

The countdown decomposition reveals the internal structure of orbits:

```python
adapter = NumberTheoryAdapter(map_type='collatz')
result = adapter.analyze_countdown(n=27)
print(f"Total steps: {result['total_steps']}")
print(f"Countdown fraction: {result['countdown_fraction']:.1%}")
print(f"Mean reset ED: {result['mean_reset_ed']:.1f}")
```

### Checking the Contraction Principle

```python
from sigma_c.core.contraction import contraction_principle_check

sweep = adapter.sweep_resolution(range(4, 17))
D_vals = [r['D_M'] for r in sweep]
gamma_vals = [r['gamma_M'] for r in sweep if r['gamma_M'] is not None]

check = contraction_principle_check(D_vals[-len(gamma_vals):], gamma_vals)
print(f"Satisfies principle: {check['satisfies_principle']}")
print(f"D stable: {check['D_stable']}, gamma stable: {check['gamma_stable']}")
```

### Integration with the Universe Factory

```python
from sigma_c import Universe

adapter = Universe.number_theory(map_type='collatz')
result = adapter.predict_behavior()
```

## Validation

### Twelve-Map Predictions

The framework correctly predicts the behavior of 12 distinct $qn+c$ maps (12/12 accuracy):

| Map | $D$ | $\gamma$ | Prediction | Status |
|-----|-----|----------|------------|--------|
| $3n+1$ (cycle) | 2.06 | 9/16 | Convergent | Verified |
| $3n+1$ (single) | 1.71 | 3/4 | Convergent | Verified |
| $5n+1$ | 1.43 | 5/4 | Divergent | Verified |
| $7n+1$ | 1.60 | 7/4 | Divergent | Verified |
| $3n-1$ | 1.33 | 3/4 | Convergent to cycles | Verified |
| $3n+3$ | 2.00 | 3/4 | Convergent to cycles | Verified |
| $3n+5$ | 1.48 | 3/4 | Convergent to cycles | Verified |
| $3n+7$ | 1.56 | 3/4 | Convergent to cycles | Verified |
| $9n+1$ | 1.34 | 9/4 | Divergent | Verified |
| $11n+1$ | 1.37 | 11/4 | Divergent | Verified |
| $5n+3$ | 1.36 | 5/4 | Divergent | Verified |
| $5n-1$ | 1.43 | 5/4 | Divergent | Verified |

### Theoretical Bounds

- **Contraction defect**: $D_M \geq 4/3$ for all well-behaved maps at sufficient resolution.
- **Drift theorem**: $\gamma = q/4$ for the single-step map $\text{odd}(qn+c)$, independent of $c$.
- **Countdown determinism**: Verified via `deterministic_trajectory` against iterated computation.
- **Reset distribution**: Chi-squared goodness-of-fit confirms $\text{Geo}(1/2)$ distribution ($p > 0.01$).

### Verification Code

```python
adapter = NumberTheoryAdapter(map_type='collatz')
result = adapter.verify_twelve_predictions(M=12)
print(f"Success rate: {result['success_rate']:.0%}")  # 100%
```
