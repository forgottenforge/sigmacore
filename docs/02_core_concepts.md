# Core Concepts

## The Philosophy: Performance vs. Stability

Traditional optimization often focuses on a single metric: **Performance** (Speed, Accuracy, Profit).
**Sigma-C** introduces a second, equally critical dimension: **Stability** (Resilience, Robustness, Safety).

In critical systems (Quantum Computing, High-Frequency Trading, AI Safety), the absolute peak performance is often fragile. A solution that is 1% slower but 10x more stable is often preferable.

## The $\sigma_c$ Metric

$\sigma_c$ (Critical Susceptibility) is a dimensionless metric representing the system's resilience to perturbations.

### Mathematical Definition
$$ \sigma_c = \frac{1}{1 + \chi} $$
Where $\chi$ (Susceptibility) is the response of the system to external noise or parameter drift.

- $\sigma_c \to 1.0$: Perfectly stable (System ignores noise).
- $\sigma_c \to 0.0$: Critical instability (System collapses under noise).

### Domain Interpretations

| Domain | Performance Metric | Stability Metric ($\sigma_c$) |
|--------|-------------------|-------------------------------|
| **Quantum** | Fidelity ($F$) | Noise Resilience (1 - Decoherence Rate) |
| **GPU** | Throughput (GFLOPS) | Thermal Headroom / Error Rate |
| **Finance** | Returns (ROI) | Risk-Adjusted Stability (1 / Volatility) |
| **ML** | Accuracy | Adversarial Robustness / Generalization |

## The Universal Optimizer

The `UniversalOptimizer` class is the heart of the framework. It implements a unified interface for all domains.

### The Objective Function
The optimizer maximizes a composite score:

$$ Score = w_p \cdot P + w_s \cdot S $$

- $P$: Normalized Performance
- $S$: Normalized Stability ($\sigma_c$)
- $w_p, w_s$: User-defined weights

### Workflow
1.  **Define Parameter Space**: Range of values to explore.
2.  **Apply Parameters**: Configure the system.
3.  **Measure**:
    - `_evaluate_performance()`
    - `_evaluate_stability()`
4.  **Score**: Calculate composite score.
5.  **Iterate**: Use Brute Force or Gradient Descent to find the maximum.

## Contraction Geometry (v3.0)

Version 3.0 introduces **Contraction Geometry**, a deeper analytical layer that explains *why* systems converge or diverge, going beyond the sigma_c metric which tells you *where* the critical point lies.

### Contraction Defect (D)

The contraction defect measures how much information a map loses:

$$ D_M = \frac{|S_M|}{|f(S_M)|} $$

- $D > 1$: The map is **non-injective** (multiple inputs collapse to the same output).
- $D = 1$: The map is injective (no information loss).
- Higher $D$ means more "folding" in the dynamics.

### Drift (gamma)

Drift $\gamma$ measures the average multiplicative growth per iteration. For a family of maps $qn + c$:

$$ \gamma = \frac{q}{4} $$

- $\gamma < 1$: Orbits shrink on average.
- $\gamma > 1$: Orbits grow on average.

### Product sigma = D * gamma

The product $\sigma = D \cdot \gamma$ is a **universal convergence threshold**:

- $\sigma < 1$: **Convergent** -- the system contracts overall.
- $\sigma = 1$: **Critical** -- marginal behavior, sensitive to initial conditions.
- $\sigma > 1$: **Divergent** -- the system expands.

### Relationship to sigma_c

While $\sigma_c$ identifies the *location* of the critical susceptibility peak, contraction geometry explains the *mechanism*:

- $\sigma_c$ answers: "Where is the tipping point?"
- $D$ and $\gamma$ answer: "Why does the system tip there?"

Together they provide both diagnostic (sigma_c) and explanatory (D, gamma) power.

### Quick Example

```python
from sigma_c_framework.sigma_c.physics.contraction import (
    compute_contraction_defect,
    classify_map
)

# Analyze the 3n+1 map
D, gamma = compute_contraction_defect(q=3, modulus=2)
sigma = D * gamma
print(f"D={D:.3f}, gamma={gamma:.3f}, sigma={sigma:.3f}")

# Classify any qn+c map automatically
result = classify_map(q=5, c=1, modulus=2)
print(f"Type: {result.map_type}, Convergent: {result.sigma < 1}")
```

## Four-Type Classification (v3.0)

Contraction geometry enables a rigorous classification of iterative maps into four types based on their $(D, \gamma)$ values:

| Type | Name | Condition | Behavior | Example |
|------|------|-----------|----------|---------|
| **D** | Deterministic | $D = 1, \gamma < 1$ | Always converges, injective map | $n/2$ |
| **O** | Oscillatory | $D > 1, \sigma < 1$ | Converges despite information loss | $3n+1$ (Collatz) |
| **S** | Stagnant | $D > 1, \sigma = 1$ | Marginal, may cycle indefinitely | Critical-point maps |
| **R** | Runaway | $D \geq 1, \sigma > 1$ | Diverges, orbits escape to infinity | $5n+1$ |

The classification is determined by the contraction product $\sigma = D \cdot \gamma$:

- **D-type** maps are the simplest: no information loss and natural shrinkage.
- **O-type** maps lose information ($D > 1$) but the drift is small enough that the product stays below 1, so orbits still converge.
- **S-type** maps sit exactly at the critical boundary -- they neither grow nor shrink on average.
- **R-type** maps have drift that overwhelms any contraction, leading to unbounded growth.
