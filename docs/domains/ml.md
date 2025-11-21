# Machine Learning Optimization Guide

## Overview
The ML domain optimizes neural networks and models. It balances **Validation Accuracy** (Performance) against **Robustness** (Stability).

## Robustness as Stability
In ML, a model with 99% accuracy that fails on slightly noisy data is "unstable". Sigma-C quantifies this using:
- **Adversarial Robustness**: Performance under attack.
- **Generalization Gap**: Difference between Train and Test accuracy.

## Hyperparameter Tuning
Use `BalancedMLOptimizer` to find the sweet spot for learning rates, batch sizes, and regularization.

```python
from sigma_c.optimization.ml import BalancedMLOptimizer

optimizer = BalancedMLOptimizer(performance_weight=0.7, stability_weight=0.3)

result = optimizer.optimize_model(
    model_factory=create_model,
    param_space={
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'dropout': [0.2, 0.5],
        'batch_size': [32, 64]
    }
)
```

## Architecture Search
You can also optimize architectural parameters:
- Number of layers
- Hidden unit size
- Activation functions

## Integration
Sigma-C is framework-agnostic. You can use it with:
- **PyTorch**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **JAX**

The `model_factory` simply needs to return a trained model or a performance metric dictionary.
