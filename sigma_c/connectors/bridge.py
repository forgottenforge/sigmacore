"""
Sigma-C Universal Bridge
=========================
Copyright (c) 2025 ForgottenForge.xyz

Universal wrapper that makes ANY function criticality-aware.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional
import numpy as np


class SigmaCBridge:
    """
    Universal adapter for ANY framework.
    
    Usage:
        from sigma_c.connectors.bridge import SigmaCBridge
        
        # Wrap any function
        @SigmaCBridge.wrap_any_function
        def my_function(x):
            return x ** 2
        
        result = my_function(5)
        print(result.__sigma_c__)  # Criticality metadata
    """
    
    @staticmethod
    def wrap_any_function(func: Callable) -> Callable:
        """
        Makes ANY function sigma_c-aware.
        
        Args:
            func: Any callable
            
        Returns:
            Wrapped function with criticality tracking
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Capture pre-state
            pre_state = SigmaCBridge._capture_state(args, kwargs)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Capture post-state
            post_state = SigmaCBridge._capture_state([result], {})
            
            # Compute criticality
            criticality = SigmaCBridge._compute_transition(pre_state, post_state)
            
            # Attach metadata
            if hasattr(result, '__dict__'):
                result.__sigma_c__ = criticality
            elif isinstance(result, (list, tuple)):
                # For collections, attach to first element if possible
                if result and hasattr(result[0], '__dict__'):
                    result[0].__sigma_c__ = criticality
            
            return result
        
        # Mark as sigma_c-aware
        wrapper.__sigma_c_enabled__ = True
        return wrapper
    
    @staticmethod
    def _capture_state(args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Capture system state from arguments.
        
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            State dictionary
        """
        state = {
            'n_args': len(args),
            'n_kwargs': len(kwargs),
            'numeric_values': []
        }
        
        # Extract numeric values
        for arg in args:
            if isinstance(arg, (int, float)):
                state['numeric_values'].append(float(arg))
            elif isinstance(arg, np.ndarray):
                state['numeric_values'].extend(arg.flatten().tolist()[:100])  # Limit size
        
        for val in kwargs.values():
            if isinstance(val, (int, float)):
                state['numeric_values'].append(float(val))
            elif isinstance(val, np.ndarray):
                state['numeric_values'].extend(val.flatten().tolist()[:100])
        
        return state
    
    @staticmethod
    def _compute_transition(pre_state: Dict, post_state: Dict) -> float:
        """
        Compute criticality from state transition.
        
        Args:
            pre_state: State before function execution
            post_state: State after function execution
            
        Returns:
            Criticality score (0-1)
        """
        # Simple heuristic: measure change in distribution
        pre_vals = np.array(pre_state.get('numeric_values', [0.0]))
        post_vals = np.array(post_state.get('numeric_values', [0.0]))
        
        if len(pre_vals) == 0 or len(post_vals) == 0:
            return 0.5  # Unknown
        
        # Compute variance ratio as proxy for criticality
        pre_var = np.var(pre_vals) if len(pre_vals) > 1 else 0.0
        post_var = np.var(post_vals) if len(post_vals) > 1 else 0.0
        
        if pre_var > 0:
            ratio = post_var / pre_var
            # Criticality is high when ratio is near 1 (scale-invariant)
            sigma_c = 1.0 / (1.0 + abs(np.log(ratio + 1e-9)))
        else:
            sigma_c = 0.5
        
        return float(np.clip(sigma_c, 0, 1))
    
    @staticmethod
    def wrap_class(cls: type) -> type:
        """
        Wrap all methods of a class.
        
        Args:
            cls: Class to wrap
            
        Returns:
            Wrapped class
        """
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_'):  # Skip private methods
                setattr(cls, name, SigmaCBridge.wrap_any_function(method))
        
        return cls
    
    @staticmethod
    def auto_detect_framework() -> Dict[str, bool]:
        """
        Detect which frameworks are installed.
        
        Returns:
            Dictionary of framework availability
        """
        frameworks = {}
        
        # Quantum
        try:
            import qiskit
            frameworks['qiskit'] = True
        except ImportError:
            frameworks['qiskit'] = False
        
        try:
            import pennylane
            frameworks['pennylane'] = True
        except ImportError:
            frameworks['pennylane'] = False
        
        # ML
        try:
            import torch
            frameworks['pytorch'] = True
        except ImportError:
            frameworks['pytorch'] = False
        
        try:
            import tensorflow
            frameworks['tensorflow'] = True
        except ImportError:
            frameworks['tensorflow'] = False
        
        # Scientific
        try:
            import julia
            frameworks['julia'] = True
        except ImportError:
            frameworks['julia'] = False
        
        return frameworks


# Example: Wrap sklearn
def wrap_sklearn():
    """
    Example: Make all sklearn estimators criticality-aware.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        CriticalRandomForest = SigmaCBridge.wrap_class(RandomForestClassifier)
        return CriticalRandomForest
    except ImportError:
        return None
