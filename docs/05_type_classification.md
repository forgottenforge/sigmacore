# Four-Type Classification

## Overview

The Sigma-C framework classifies all operations on finite state spaces into four types based on their injectivity structure and contraction properties. This taxonomy provides a unified language for describing systems across number theory, physics, biology, and computation.

The core module is `sigma_c.core.classification`.

## The Taxonomy

### Type D -- Dissipative

**Definition**: $D > 1$ (non-injective). The map loses information at each step.

**Properties**:
- Multiple inputs map to the same output.
- Information is irreversibly destroyed ($\log_2(D)$ bits per step).
- Behavior is determined by the drift $\gamma$:
  - $\gamma < 1$: convergent (orbits collapse).
  - $\gamma > 1$: divergent (orbits escape despite collisions).
  - $\gamma \approx 1$: critical (marginal behavior).

**Examples**:
- Collatz map $n \to \text{odd}(3n+1)$: $D \approx 2.06$, $\gamma \approx 0.56$ (convergent).
- Map $n \to \text{odd}(5n+1)$: $D \approx 1.43$, $\gamma \approx 1.25$ (divergent).
- Protein folding: destabilizing mutations increase $\sigma = D \cdot \gamma$ toward 1.

### Type O -- Oversaturated

**Definition**: $D > 1$ with growing pre-image counts. The map has redundancy that increases with scale.

**Properties**:
- Representation counts exceed theoretical predictions.
- The oversaturation ratio $O_M = \min(\text{count}/\text{prediction})$ quantifies the excess.
- Relevant to additive number theory (e.g., Goldbach-type representation counts).

**Examples**:
- Goldbach representations: every even $n > 4$ has at least $O_M$ representations as a sum of two primes, with $O_M$ growing as $n$ increases.
- Waring's problem: representation counts for sums of $k$-th powers.

### Type S -- Symmetric

**Definition**: $D = 1$ (bijective) with a symmetry constraint that restricts the dynamics.

**Properties**:
- The map is invertible, so no information is lost.
- A symmetry group constrains the allowed orbits.
- The constraint tightness $C_M$ measures how closely the system conforms to a reference variance (e.g., GUE statistics for random matrix ensembles).

**Examples**:
- Zeta zero spacings: the Riemann zeta function zeros, when normalized, exhibit GUE-like statistics. The operation on spacings is bijective but constrained by the functional equation symmetry.
- Random matrix eigenvalue dynamics: unitary evolution preserves eigenvalue count (bijective) but constrains spacings via repulsion.

### Type R -- Reversible

**Definition**: $D = 1$ (bijective) with orbit preservation (conservation law).

**Properties**:
- The map is fully invertible.
- Orbits under a symmetry operation $g$ are preserved by $f$.
- A conserved quantity exists (Noether-type invariant).

**Examples**:
- Hamiltonian dynamics: phase-space volume is preserved (Liouville's theorem).
- Reversible cellular automata: the state evolution is bijective by construction.
- Galois group actions on field extensions: the action permutes roots, preserving orbit structure.

## Decision Tree

```
Is the map injective (D = 1)?
    |
    +-- YES: Does it have a symmetry constraint?
    |         |
    |         +-- YES --> Type S (Symmetric)
    |         +-- NO  --> Type R (Reversible)
    |
    +-- NO (D > 1): Do pre-image counts grow with scale?
              |
              +-- YES --> Type O (Oversaturated)
              +-- NO  --> Type D (Dissipative)
```

## Classification Summary

| Type | $D$ | Key Property | Behavior |
|------|-----|-------------|----------|
| **D** (Dissipative) | $> 1$ | Non-injective | Convergent or divergent (per $\gamma$) |
| **O** (Oversaturated) | $> 1$ | Growing pre-images | Redundancy increases with scale |
| **S** (Symmetric) | $= 1$ | Bijective + symmetry | Constrained by symmetry group |
| **R** (Reversible) | $= 1$ | Bijective + conservation | Orbits preserved, energy conserved |

## API Reference

### `MapType` Enum

```python
from sigma_c.core.classification import MapType

MapType.DISSIPATIVE      # "D"
MapType.OVERSATURATED    # "O"
MapType.SYMMETRIC        # "S"
MapType.REVERSIBLE       # "R"
```

### `classify_operation`

```python
from sigma_c.core.classification import classify_operation

map_type = classify_operation(
    D=2.06,                       # Contraction defect
    gamma=0.5625,                 # Drift (optional)
    has_cycles=False,             # Whether cycles are known
    is_bijective=None,            # Override: force D=1 interpretation
    has_symmetry=False,           # Whether symmetry constraints exist
    has_growing_preimage=False    # Whether pre-image counts grow
)
# Returns: MapType.DISSIPATIVE
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `D` | `float` | Contraction defect |
| `gamma` | `Optional[float]` | Drift (required for Type D analysis) |
| `has_cycles` | `bool` | Whether non-trivial cycles exist |
| `is_bijective` | `Optional[bool]` | Override bijective detection |
| `has_symmetry` | `bool` | Whether symmetry constraints are present |
| `has_growing_preimage` | `bool` | Whether pre-image counts grow with scale |

### Analysis Functions

```python
from sigma_c.core.classification import (
    analyze_type_d, analyze_type_o,
    analyze_type_s, analyze_type_r
)
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `analyze_type_d` | `(D: float, gamma: float, has_cycles: bool) -> TypeDResult` | Prediction, confidence, details |
| `analyze_type_o` | `(counts: ndarray, predictions: ndarray) -> TypeOResult` | Oversaturation ratio, min count |
| `analyze_type_s` | `(deviations: ndarray, reference_variance: float) -> TypeSResult` | Symmetry deviation, constraint tightness |
| `analyze_type_r` | `(f_values: ndarray, g_orbits: ndarray) -> TypeRResult` | Bijective check, orbit preservation |

### Result Classes

Each type has a corresponding result class with a `to_dict()` method:

- `TypeDResult`: `D`, `gamma`, `has_cycles`, `prediction`, `details`
- `TypeOResult`: `oversaturation_ratio`, `min_count`, `reference_prediction`, `details`
- `TypeSResult`: `symmetry_deviation`, `constraint_tightness`, `details`
- `TypeRResult`: `is_bijective`, `orbits_preserved`, `conserved_quantity`, `details`

### Example

```python
from sigma_c.core.classification import classify_operation, analyze_type_d

# Classify the Collatz map
map_type = classify_operation(D=2.06, gamma=0.5625)
assert map_type == MapType.DISSIPATIVE

# Full Type D analysis
result = analyze_type_d(D=2.06, gamma=0.5625, has_cycles=False)
print(result.prediction)  # 'convergent'
print(result.to_dict())
```
