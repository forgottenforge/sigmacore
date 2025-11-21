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
