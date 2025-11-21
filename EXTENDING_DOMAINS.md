# Extending Sigma-C Framework to New Domains

## Overview
The Sigma-C Framework is designed to be universal and extensible. This guide shows you how to add support for new domains beyond Quantum, GPU, Financial, and ML.

## Architecture Overview

The framework consists of three main components:

1. **Adapter** (`sigma_c/adapters/`): Interfaces with the domain-specific system
2. **Optimizer** (`sigma_c/optimization/`): Balances performance vs. stability
3. **Physics** (`sigma_c/physics/`): Validates theoretical bounds

## Step 1: Create Domain Adapter

Create `sigma_c/adapters/your_domain.py`:

```python
from ..core.base import SigmaCAdapter
from typing import Any, Dict
import numpy as np

class YourDomainAdapter(SigmaCAdapter):
    """Adapter for Your Domain."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Initialize domain-specific connections
        
    def get_observable(self, data: Any, **kwargs) -> float:
        """Extract observable metric from domain data."""
        # Return normalized metric [0, 1]
        return 0.5
    
    def compute_susceptibility(self, epsilon: np.ndarray, 
                              observable: np.ndarray) -> Dict[str, Any]:
        """Compute sigma_c using core engine."""
        return super().compute_susceptibility(epsilon, observable)
```

## Step 2: Create Domain Optimizer

Create `sigma_c/optimization/your_domain.py`:

```python
from .universal import UniversalOptimizer, OptimizationResult
from ..adapters.your_domain import YourDomainAdapter
from typing import Any, Dict, List

class BalancedYourDomainOptimizer(UniversalOptimizer):
    """Optimizes systems in your domain."""
    
    def __init__(self, adapter: YourDomainAdapter, 
                 performance_weight: float = 0.6, 
                 stability_weight: float = 0.4):
        super().__init__(performance_weight, stability_weight)
        self.adapter = adapter
        
    def _evaluate_performance(self, system: Any, params: Dict[str, Any]) -> float:
        """Measure domain-specific performance metric."""
        # Example: throughput, accuracy, returns, etc.
        result = self.adapter.measure_performance(system, params)
        return result['performance']
    
    def _evaluate_stability(self, system: Any, params: Dict[str, Any]) -> float:
        """Measure sigma_c (stability/resilience)."""
        result = self.adapter.analyze_stability(system, params)
        return result['sigma_c']
    
    def _apply_params(self, system: Any, params: Dict[str, Any]) -> Any:
        """Apply parameters to system."""
        return self.adapter.configure_system(system, params)
```

## Step 3: Create Physics Validation

Create `sigma_c/physics/your_domain.py`:

```python
from .rigorous import RigorousTheoreticalCheck
from typing import Any, Dict, List
import numpy as np

class RigorousYourDomainSigmaC(RigorousTheoreticalCheck):
    """Validates sigma_c against domain-specific theory."""
    
    def check_theoretical_bounds(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Check if sigma_c respects theoretical bounds."""
        # Define bounds based on domain theory
        lower_bound = 0.01
        upper_bound = 0.5
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'metric': 'sigma_c',
            'theory': 'Your Domain Theory'
        }
    
    def check_scaling_laws(self, data: Any, param_range: List[float], **kwargs) -> Dict[str, Any]:
        """Verify scaling laws (e.g., power-law, exponential)."""
        # Fit scaling model
        return {'exponent': -0.5, 'fit_success': True}
    
    def quantify_resource(self, data: Any) -> float:
        """Quantify resource consumption."""
        # Return normalized resource metric
        return 0.5
```

## Step 4: Update Module Exports

Add to `sigma_c/adapters/__init__.py`:
```python
from .your_domain import YourDomainAdapter
```

Add to `sigma_c/optimization/__init__.py`:
```python
from .your_domain import BalancedYourDomainOptimizer
```

## Step 5: Create Example Usage

Create `examples_v1.2/demo_your_domain.py`:

```python
from sigma_c_framework.sigma_c.adapters.your_domain import YourDomainAdapter
from sigma_c_framework.sigma_c.optimization.your_domain import BalancedYourDomainOptimizer

# Initialize
adapter = YourDomainAdapter()
optimizer = BalancedYourDomainOptimizer(adapter, performance_weight=0.7, stability_weight=0.3)

# Optimize
result = optimizer.optimize(
    system=my_system,
    param_space={'param1': [1, 2, 3], 'param2': [0.1, 0.2]},
    strategy='brute_force'
)

print(f"Optimal params: {result.optimal_params}")
print(f"Performance: {result.performance_after}")
print(f"Sigma_c: {result.sigma_c_after}")
```

## Best Practices

### 1. Performance Metrics
- Should be normalized [0, 1] or have clear interpretation
- Higher is better
- Examples: accuracy, throughput, returns, efficiency

### 2. Stability Metrics (Sigma_c)
- Represents critical threshold or resilience
- Should be measurable via susceptibility analysis
- Higher sigma_c = more stable/resilient

### 3. Theoretical Validation
- Define bounds based on domain theory
- Check scaling laws (power-law, exponential, etc.)
- Quantify resource consumption

### 4. Testing
- Create unit tests in `tests/`
- Verify optimizer converges
- Validate physics bounds

## Example Domains

### Robotics
- **Performance**: Task completion rate
- **Stability**: Robustness to perturbations (sigma_c)
- **Theory**: Control theory, Lyapunov stability

### Network Systems
- **Performance**: Throughput
- **Stability**: Congestion threshold (sigma_c)
- **Theory**: Queueing theory, network calculus

### Chemical Reactions
- **Performance**: Yield
- **Stability**: Critical temperature (sigma_c)
- **Theory**: Thermodynamics, reaction kinetics

### Climate Models
- **Performance**: Prediction accuracy
- **Stability**: Tipping point threshold (sigma_c)
- **Theory**: Dynamical systems, bifurcation theory

## Support

For questions or assistance extending the framework:
- Email: info@forgottenforge.xyz
- Documentation: See `DOCUMENTATION.md`
- Examples: See `examples_v1.2/`

## License

Extensions must comply with AGPL-3.0-or-later OR Commercial License.
Contact info@forgottenforge.xyz for commercial licensing.
