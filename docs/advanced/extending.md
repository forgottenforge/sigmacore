# Extending Sigma-C

## Overview
Sigma-C is designed to be extensible. You can add support for new domains (e.g., Robotics, Climate Science) by implementing a few abstract base classes.

## The Architecture
1.  **Adapter**: Interfaces with the domain-specific hardware or simulation.
2.  **Optimizer**: Inherits from `UniversalOptimizer` and defines the objective function.
3.  **Physics**: (Optional) Defines theoretical bounds and validation checks.

## Step-by-Step Guide

### 1. Create the Adapter
```python
from sigma_c.core.base import SigmaCAdapter

class RoboticsAdapter(SigmaCAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.robot = connect_to_robot(config['ip'])

    def run_simulation(self, params):
        # ... run physics engine ...
        return {'speed': 1.5, 'wobble': 0.1}
```

### 2. Create the Optimizer
```python
from sigma_c.optimization.universal import UniversalOptimizer

class BalancedRoboticsOptimizer(UniversalOptimizer):
    def _evaluate_performance(self, system, params):
        # Metric: Speed (m/s)
        return system.run_simulation(params)['speed']

    def _evaluate_stability(self, system, params):
        # Metric: Inverse of wobble
        wobble = system.run_simulation(params)['wobble']
        return 1.0 / (1.0 + wobble)

    def _apply_params(self, system, params):
        # No-op if system is the adapter itself
        return system
```

### 3. Use It
```python
adapter = RoboticsAdapter({'ip': '192.168.1.100'})
optimizer = BalancedRoboticsOptimizer()
result = optimizer.optimize(adapter, {'motor_gain': [10, 20, 30]})
```
