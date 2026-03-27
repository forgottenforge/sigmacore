# Contraction Geometry

## Overview

Contraction geometry is the mathematical framework underlying Sigma-C v3.0. It provides two fundamental quantities -- the **contraction defect** $D$ and the **drift** $\gamma$ -- that together characterize how a non-injective operation compresses and rescales a discrete state space. These quantities are sufficient to predict whether an iterated system converges, diverges, or cycles.

The core module is `sigma_c.core.contraction`.

## Fundamental Concepts

### Contraction Defect ($D$)

The contraction defect quantifies the degree of non-injectivity of a map $f$ acting on a finite domain $S$:

$$D_M = \frac{|S_M|}{|f(S_M)|}$$

where $S_M$ is the domain at modular resolution $M$ and $f(S_M)$ is its image under $f$.

**Interpretation**:
- $D = 1$: The map is injective (bijective on its image). No information is lost.
- $D > 1$: The map is non-injective. On average, $D$ elements map to each image element.
- $D = 2$: Each image element has, on average, two pre-images. Half the state space is "lost" per step.

For Collatz-type maps on odd residues modulo $2^M$, the contraction defect stabilizes as $M$ increases, converging to a well-defined limit $D_\infty$.

### Drift ($\gamma$)

The drift measures the average multiplicative growth of values under one application of the map:

$$\gamma_M = 2^{\frac{1}{|S_M|} \sum_{n \in S_M} \log_2 \frac{f(n)}{n}}$$

This is the geometric mean of the ratio $f(n)/n$ across all elements of the domain.

**Interpretation**:
- $\gamma < 1$: Values decrease on average (contraction).
- $\gamma = 1$: Values are preserved on average (neutral).
- $\gamma > 1$: Values increase on average (expansion).

For the single-step map $n \to \text{odd}(qn + c)$, the drift converges to $\gamma = q/4$ as $M \to \infty$, independent of the additive constant $c$. This universality is a consequence of the 2-adic structure.

### Observation Scale ($\sigma_c$)

The critical susceptibility $\sigma_c$ identifies the scale at which a system's response to perturbation is maximized. In the contraction geometry framework, the product $\sigma = D \cdot \gamma$ serves as the contraction index:

- $\sigma < 1$: Net contraction per step (convergent system).
- $\sigma = 1$: Critical threshold (marginal behavior).
- $\sigma > 1$: Net expansion per step (divergent system).

## Mathematical Foundations

### Domain and Image

For a map $f: S \to S$ on a finite set $S$ of size $N$:

$$D = \frac{N}{|f(S)|}$$

The contraction defect is always $\geq 1$ by the pigeonhole principle. Equality holds if and only if $f$ is injective.

### Modular Resolution

In the number-theoretic setting, $S_M$ consists of odd residues modulo $2^M$:

$$S_M = \{1, 3, 5, \ldots, 2^M - 1\}$$

with $|S_M| = 2^{M-1}$. The map $f$ acts on residue classes, and we track which classes collide:

$$D_M = \frac{2^{M-1}}{|\{f(n) \bmod 2^M : n \in S_M\}|}$$

As $M$ increases, $D_M$ converges to a limit that captures the global non-injectivity of $f$.

### Drift via 2-adic Valuation

For the map $n \to \text{odd}(qn + c)$, the output is $(qn + c) / 2^{v_2(qn+c)}$. The drift can be expressed as:

$$\gamma = q \cdot 2^{-\langle v_2(qn+c) \rangle}$$

where $\langle v_2(qn+c) \rangle$ is the average 2-adic valuation over the domain. For equidistributed odd $n$, this average converges to 2, yielding $\gamma = q/4$.

## The Contraction Principle

**Axiom (Contraction Principle)**: If a map $f$ on a finite state space satisfies:
1. $D > 1$ (structural non-injectivity), and
2. $\gamma < 1$ (average contraction),

then the iterated system $f^n(x)$ cannot escape to infinity. The orbit must eventually enter a cycle or converge to a fixed point.

**Checking the principle**:

```python
from sigma_c.core.contraction import contraction_principle_check

result = contraction_principle_check(
    D_values=[2.05, 2.06, 2.06, 2.06],
    gamma_values=[0.56, 0.56, 0.56, 0.56]
)
print(result['satisfies_principle'])  # True
print(result['prediction'])          # 'convergent'
```

The function verifies that both $D$ and $\gamma$ stabilize (coefficient of variation $< 5\%$ in the tail half of the sequence) and that $D > 1$ throughout.

## Local-Global Interpretation

### 2-adic (Local) Perspective

The contraction defect $D_M$ captures **local** structure at resolution $M$: how many residue classes collide under the map. This is inherently a $p$-adic quantity -- it measures the structure of pre-images in the 2-adic completion $\mathbb{Z}_2$.

- At each resolution $M$, the map $f$ is examined modulo $2^M$.
- Collisions reveal the non-injectivity structure.
- The limit $D_\infty = \lim_{M \to \infty} D_M$ captures the full 2-adic behavior.

### Archimedean (Global) Perspective

The drift $\gamma$ captures **global** behavior: the average multiplicative change in the real-valued magnitude of the iterates. This is an archimedean quantity -- it measures growth in $\mathbb{R}$.

- $\gamma < 1$ means iterates shrink in absolute value.
- The rate of shrinkage is $\log_2(\gamma)$ bits per step.

### Combined Prediction

The contraction principle combines both perspectives:

| 2-adic ($D$) | Archimedean ($\gamma$) | Prediction |
|--------------|----------------------|------------|
| $D > 1$ | $\gamma < 1$ | **Convergent** |
| $D > 1$ | $\gamma > 1$ | **Divergent** |
| $D > 1$ | $\gamma \approx 1$ | **Indeterminate** |
| $D = 1$ | any | **Injective** (reversible) |

## Information-Theoretic Interpretation

### Bits Lost per Step

Each application of a non-injective map with contraction defect $D$ irreversibly erases information:

$$\text{bits\_lost} = \log_2(D)$$

For the Collatz cycle map with $D \approx 2.06$:

$$\text{bits\_lost} \approx \log_2(2.06) \approx 1.04 \text{ bits per step}$$

This means each step of the Collatz map destroys slightly more than one bit of information about the input.

### Landauer Principle

The Landauer principle establishes a lower bound on the thermodynamic cost of information erasure:

$$E_{\min} = k_B T \ln(2) \cdot \log_2(D)$$

At room temperature (300 K), erasing one bit costs at least $2.87 \times 10^{-21}$ J. The contraction defect directly determines the minimum energy dissipation per map application.

```python
from sigma_c.beyond.information import bits_lost, landauer_cost, information_summary

# Collatz cycle map: D ~ 2.06
print(f"Bits lost: {bits_lost(2.06):.3f}")           # ~1.04 bits
print(f"Landauer cost: {landauer_cost(2.06):.2e} J")  # ~2.99e-21 J

summary = information_summary(D=2.06, gamma=0.5625)
print(summary['interpretation'])
```

### Entropy Production Rate

For a map applied at rate $r$ steps per second:

$$\frac{dS}{dt} = k_B \ln(2) \cdot \log_2(D) \cdot r$$

This connects the abstract contraction defect to measurable thermodynamic quantities.

### Net Contraction Index

The product $\sigma = D \cdot \gamma$ serves as a dimensionless contraction index that unifies the 2-adic and archimedean perspectives:

- When $D > 1$ and $\gamma < 1$, the net effect depends on their product.
- $\sigma < 1$: The compression (from non-injectivity) dominates the expansion (from growth). Net information loss per step exceeds the bits gained from growth.
- $\sigma > 1$: Growth dominates compression. The system expands despite losing distinct states.

This quantity appears across all domains:
- Number theory: $\sigma = D \cdot \gamma$ predicts convergence vs divergence.
- Protein stability: $\sigma = \exp(\Delta\Delta G / (NRT))$ crosses 1.0 at the folding threshold.
- Linguistics: higher morphological complexity (analogous to $D$) correlates with greater semantic drift ($\gamma$).

## Usage Example

```python
from sigma_c.core.contraction import (
    compute_contraction_defect, compute_drift,
    classify_map, cycle_map, sweep_modular_resolution
)

# Compute D and gamma for the Collatz cycle map at M=14
D = compute_contraction_defect(cycle_map, M=14)
gamma = compute_drift(cycle_map, M=14)
print(f"D = {D:.4f}, gamma = {gamma:.4f}")  # D ~ 2.06, gamma ~ 0.56

# Classify the behavior
result = classify_map(D, gamma)
print(result['prediction'])  # 'convergent'

# Full resolution sweep
sweep = sweep_modular_resolution(cycle_map, range(4, 15))
for row in sweep:
    print(f"M={row['M']:2d}  D_M={row['D_M']:.4f}  gamma_M={row['gamma_M']:.4f}")
```

## API Summary

### Core Functions (`sigma_c.core.contraction`)

| Function | Description |
|----------|-------------|
| `compute_contraction_defect(f, M)` | Compute $D_M$ at resolution $M$ |
| `compute_drift(f, M)` | Compute $\gamma_M$ at resolution $M$ |
| `classify_map(D, gamma, ...)` | Predict behavior from $(D, \gamma)$ |
| `sweep_modular_resolution(f, M_range)` | Compute $(D_M, \gamma_M)$ across resolutions |
| `contraction_principle_check(D_values, gamma_values)` | Check stabilization and convergence |

### Information Theory (`sigma_c.beyond.information`)

| Function | Description |
|----------|-------------|
| `bits_lost(D)` | Information loss in bits per step |
| `landauer_cost(D, T)` | Minimum energy cost of erasure (Joules) |
| `entropy_production_rate(D, rate, T)` | Entropy production rate (J/(K*s)) |
| `information_summary(D, gamma, T, rate)` | Complete information-theoretic summary |
