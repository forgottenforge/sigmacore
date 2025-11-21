"""
Balanced Quantum Optimizer
==========================
Optimizes quantum circuits by balancing Fidelity (Performance) vs. Resilience (Stability/Sigma_c).

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .universal import UniversalOptimizer, OptimizationResult
from ..adapters.quantum import QuantumAdapter

class BalancedQuantumOptimizer(UniversalOptimizer):
    """
    Optimizes quantum circuits for both performance (fidelity) and stability (sigma_c).
    """
    
    def __init__(self, adapter: QuantumAdapter, performance_weight: float = 0.6, stability_weight: float = 0.4):
        super().__init__(performance_weight, stability_weight)
        self.adapter = adapter
        
    def _evaluate_performance(self, system: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate circuit fidelity or success probability.
        'system' is expected to be a circuit object or a circuit generation function.
        """
        # If system is a function, call it with params to get circuit
        if callable(system):
            circuit = system(**params)
        else:
            # If it's a static circuit, we can't easily parameterize it here without more context
            # For now, assume system is a callable factory
            raise ValueError("System must be a callable that returns a circuit given params.")
            
        # Measure pure fidelity at eps=0.0 (no additional noise beyond what's in params)
        # We run the circuit once with minimal shots to get fidelity estimate
        result = self.adapter.run_optimization(
            circuit_type='custom',
            epsilon_values=[0.0],  # No additional noise
            shots=100,
            custom_circuit=circuit
        )
        
        # Observable is the success probability (fidelity)
        fidelity = result['observable'][0]
        return fidelity

    def _evaluate_stability(self, system: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate circuit resilience (sigma_c).
        """
        if callable(system):
            circuit = system(**params)
        else:
            raise ValueError("System must be a callable that returns a circuit given params.")
            
        # Run full sigma_c analysis
        # We use a coarser grid for speed during optimization
        result = self.adapter.run_optimization(
            circuit_type='custom',
            epsilon_values=np.linspace(0.02, 0.2, 5),
            shots=100,
            custom_circuit=circuit
        )
        
        return result['sigma_c']

    def _apply_params(self, system: Any, params: Dict[str, Any]) -> Any:
        """
        For quantum, applying params usually means regenerating the circuit.
        So we just return the callable system, as evaluation handles generation.
        """
        return system

    def optimize_circuit(self, 
                        circuit_factory: callable, 
                        param_space: Dict[str, List[Any]],
                        strategy: str = 'brute_force') -> OptimizationResult:
        """
        Specialized optimize method for circuits.
        """
        return self.optimize(circuit_factory, param_space, strategy)
