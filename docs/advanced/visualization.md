# Visualization Guide

## Overview
Sigma-C provides a suite of visualization tools to help you understand the optimization landscape and the trade-off between performance and stability.

## Convergence Plot
Shows how the composite score (or individual metrics) improves over time.

```python
from sigma_c.visualization import plot_convergence
plot_convergence(result, metric='score', save_path='convergence.png')
```

## Pareto Frontier
Visualizes the trade-off between Performance and Stability. This is crucial for finding the "sweet spot".

- **X-Axis**: Performance
- **Y-Axis**: Stability ($\sigma_c$)
- **Color**: Composite Score

```python
from sigma_c.visualization import plot_pareto_frontier
plot_pareto_frontier(result, save_path='pareto.png')
```

## Landscape Plot
For 2-parameter problems, you can visualize the entire objective function landscape. This helps identify local minima and stability plateaus.

```python
from sigma_c.visualization import plot_landscape
plot_landscape(optimizer, param_space, system, resolution=50)
```

## Custom Plots
The `OptimizationResult` object contains a full history of the optimization run, which you can use to create custom plots with Matplotlib or Seaborn.

```python
history = result.history
scores = [h['score'] for h in history]
# ... your custom plotting code ...
```
