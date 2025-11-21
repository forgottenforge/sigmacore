"""
Universal Optimizer Module
==========================
Core logic of the Sigma-C Framework v1.2.3.
Generalizes QPU optimization concepts (fidelity vs. resilience) to all domains.

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Standardized result from any optimization run."""
    optimal_params: Dict[str, Any]
    score: float
    history: List[Dict[str, Any]]
    sigma_c_before: float
    sigma_c_after: float
    performance_metric_name: str
    performance_before: float
    performance_after: float
    strategy_used: str

class UniversalOptimizer(ABC):
    """
    Abstract base class for domain-specific optimizers.
    Balances a primary performance metric against a stability metric (sigma_c).
    """

    def __init__(self, 
                 performance_weight: float = 0.7, 
                 stability_weight: float = 0.3):
        self.performance_weight = performance_weight
        self.stability_weight = stability_weight
        self.history = []

    @abstractmethod
    def _evaluate_performance(self, system: Any, params: Dict[str, Any]) -> float:
        """Calculate the domain-specific performance metric (e.g., Fidelity, FLOPS, Sharpe)."""
        pass

    @abstractmethod
    def _evaluate_stability(self, system: Any, params: Dict[str, Any]) -> float:
        """Calculate the domain-specific stability metric (usually related to sigma_c)."""
        pass

    @abstractmethod
    def _apply_params(self, system: Any, params: Dict[str, Any]) -> Any:
        """Apply parameters to the system (returns a new system or modifies in-place)."""
        pass

    def calculate_score(self, performance: float, stability: float) -> float:
        """
        Composite score: w_p * performance + w_s * stability
        Higher is better.
        """
        # Normalize inputs if possible, but for now assume they are scaled reasonably
        return (self.performance_weight * performance) + (self.stability_weight * stability)

    def optimize(self, 
                 system: Any, 
                 param_space: Dict[str, List[Any]], 
                 strategy: str = 'brute_force',
                 callbacks: Optional[List[Any]] = None,
                 **kwargs) -> OptimizationResult:
        """
        Main entry point for optimization.
        """
        # 0. Setup callbacks
        self.stop_optimization = False
        callbacks = callbacks or []
        for cb in callbacks:
            if hasattr(cb, 'on_optimization_start'):
                cb.on_optimization_start(self, param_space)

        # 1. Baseline measurement
        base_perf = self._evaluate_performance(system, {})
        base_stab = self._evaluate_stability(system, {})
        
        best_score = self.calculate_score(base_perf, base_stab)
        best_params = {}
        best_perf = base_perf
        best_stab = base_stab
        
        self.history = [{
            'params': {},
            'performance': base_perf,
            'stability': base_stab,
            'score': best_score
        }]

        # 2. Strategy selection
        if strategy == 'brute_force':
            result = self._optimize_brute_force(system, param_space, callbacks)
        elif strategy == 'gradient_descent':
            result = self._optimize_gradient(system, param_space) # Placeholder
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        # 3. Final callback
        for cb in callbacks:
            if hasattr(cb, 'on_optimization_end'):
                cb.on_optimization_end(self, result)
            
        return result

    def _optimize_brute_force(self, system: Any, param_space: Dict[str, List[Any]], callbacks: List[Any]) -> OptimizationResult:
        """
        Exhaustive search over the parameter space.
        """
        import itertools
        
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))
        
        best_params = {}
        best_score = -float('inf')
        best_perf = 0.0
        best_stab = 0.0
        
        # Baseline
        base_perf = self._evaluate_performance(system, {})
        base_stab = self._evaluate_stability(system, {})
        
        step = 0
        for combo in combinations:
            if self.stop_optimization:
                break
                
            params = dict(zip(keys, combo))
            
            # Apply and Measure
            try:
                modified_system = self._apply_params(system, params)
                perf = self._evaluate_performance(modified_system, params)
                stab = self._evaluate_stability(modified_system, params)
                score = self.calculate_score(perf, stab)
                
                log_entry = {
                    'params': params,
                    'performance': perf,
                    'stability': stab,
                    'score': score
                }
                self.history.append(log_entry)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_perf = perf
                    best_stab = stab
                
                # Callbacks
                step += 1
                for cb in callbacks:
                    if hasattr(cb, 'on_step_end'):
                        cb.on_step_end(self, step, log_entry)
                    
            except Exception as e:
                # Log failure but continue
                continue
                
        return OptimizationResult(
            optimal_params=best_params,
            score=best_score,
            history=self.history,
            sigma_c_before=base_stab,
            sigma_c_after=best_stab,
            performance_metric_name="Composite",
            performance_before=base_perf,
            performance_after=best_perf,
            strategy_used="brute_force"
        )

    def _optimize_gradient(self, system: Any, param_space: Dict[str, List[Any]]) -> OptimizationResult:
        """Gradient-descent optimisation using ``scipy.optimize.minimize``."""
        # Try to import scipy; if unavailable we fall back to a simple random search.
        try:
            from scipy.optimize import minimize
            _has_scipy = True
        except Exception:
            _has_scipy = False

        # Separate numeric and non-numeric parameters.
        numeric_params = {k: v for k, v in param_space.items() if all(isinstance(x, (int, float)) for x in v)}
        non_numeric_params = {k: v for k, v in param_space.items() if k not in numeric_params}

        # Helper to convert vector -> dict and evaluate score.
        def vector_to_dict(vec):
            return {k: float(val) for k, val in zip(numeric_params.keys(), vec)}

        def objective(vec):
            # Convert vector to param dict and merge with any non-numeric defaults (first entry).
            param_dict = vector_to_dict(vec)
            for k, v in non_numeric_params.items():
                param_dict[k] = v[0]  # pick first value as a placeholder
            # Apply parameters and evaluate.
            try:
                modified = self._apply_params(system, param_dict)
                perf = self._evaluate_performance(modified, param_dict)
                stab = self._evaluate_stability(modified, param_dict)
                # We *minimise* the negative score.
                return -self.calculate_score(perf, stab)
            except Exception:
                # If evaluation fails, return a large penalty.
                return 1e6

        if _has_scipy and numeric_params:
            # Initialise at the centre of each bound.
            x0 = []
            bounds = []
            for vals in numeric_params.values():
                lo, hi = min(vals), max(vals)
                x0.append((lo + hi) / 2.0)
                bounds.append((lo, hi))
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            best_vec = res.x
            best_params = vector_to_dict(best_vec)
            # Merge non-numeric defaults.
            for k, v in non_numeric_params.items():
                best_params[k] = v[0]
            # Final evaluation for result construction.
            final_system = self._apply_params(system, best_params)
            final_perf = self._evaluate_performance(final_system, best_params)
            final_stab = self._evaluate_stability(final_system, best_params)
            best_score = self.calculate_score(final_perf, final_stab)
            return OptimizationResult(
                optimal_params=best_params,
                score=best_score,
                history=self.history,
                sigma_c_before=self.history[0]['stability'],
                sigma_c_after=final_stab,
                performance_metric_name="Composite",
                performance_before=self.history[0]['performance'],
                performance_after=final_perf,
                strategy_used="gradient_descent"
            )
        else:
            # Simple random search fallback - sample 20 random combos.
            import random, itertools
            keys = list(param_space.keys())
            values = list(param_space.values())
            combos = []
            for _ in range(20):
                combo = [random.choice(v) for v in values]
                combos.append(dict(zip(keys, combo)))
            best_score = -float('inf')
            best_params = {}
            best_perf = 0.0
            best_stab = 0.0
            for params in combos:
                try:
                    mod = self._apply_params(system, params)
                    perf = self._evaluate_performance(mod, params)
                    stab = self._evaluate_stability(mod, params)
                    score = self.calculate_score(perf, stab)
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_perf = perf
                        best_stab = stab
                except Exception:
                    continue
            return OptimizationResult(
                optimal_params=best_params,
                score=best_score,
                history=self.history,
                sigma_c_before=self.history[0]['stability'],
                sigma_c_after=best_stab,
                performance_metric_name="Composite",
                performance_before=self.history[0]['performance'],
                performance_after=best_perf,
                strategy_used="random_fallback"
            )

    def save(self, filepath: str):
        """Save optimizer state and history to JSON."""
        import json
        data = {
            'history': self.history,
            'performance_weight': self.performance_weight,
            'stability_weight': self.stability_weight
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, filepath: str):
        """Load optimizer state from JSON."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.history = data.get('history', [])
        self.performance_weight = data.get('performance_weight', 0.7)
        self.stability_weight = data.get('stability_weight', 0.3)
