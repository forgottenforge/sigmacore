"""
Sigma-C JAX/Flax Integration
=============================
Copyright (c) 2025 ForgottenForge.xyz

JAX integration for criticality-aware differentiable programming.
"""

import functools
from typing import Callable, Any, Dict
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None
    jnp = None


def critical_jit(func: Callable) -> Callable:
    """
    Wrapper for jax.jit with criticality tracking.
    
    Usage:
        import jax
        from sigma_c.ml.jax import critical_jit
        
        @critical_jit
        def train_step(params, batch):
            loss, sigma_c = compute_loss(params, batch)
            return loss, sigma_c
    """
    if not _HAS_JAX:
        return func
    
    # Apply JAX JIT compilation
    jitted = jax.jit(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = jitted(*args, **kwargs)
        
        # Attach criticality metadata
        if isinstance(result, tuple) and len(result) >= 2:
            # Assume second return value is sigma_c
            loss, sigma_c = result[0], result[1]
            if hasattr(loss, '__dict__'):
                loss.__sigma_c__ = sigma_c
        
        return result
    
    wrapper.__sigma_c_enabled__ = True
    return wrapper


def critical_grad(func: Callable, argnums: int = 0) -> Callable:
    """
    Gradient computation with criticality awareness.
    
    Usage:
        grad_fn = critical_grad(loss_fn, argnums=0)
        grads, sigma_c = grad_fn(params, data)
    """
    if not _HAS_JAX:
        return func
    
    # Create gradient function
    grad_fn = jax.grad(func, argnums=argnums)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Compute gradients
        grads = grad_fn(*args, **kwargs)
        
        # Compute criticality from gradient statistics
        if isinstance(grads, dict):
            grad_values = jnp.concatenate([g.flatten() for g in grads.values()])
        else:
            grad_values = grads.flatten()
        
        # Criticality metric: gradient variance
        sigma_c = float(jnp.var(grad_values))
        
        return grads, sigma_c
    
    return wrapper


class CriticalOptimizer:
    """
    JAX optimizer with criticality regularization.
    
    Usage:
        from sigma_c.ml.jax import CriticalOptimizer
        
        optimizer = CriticalOptimizer(learning_rate=0.01, lambda_critical=0.1)
        state = optimizer.init(params)
        
        for batch in data:
            grads, sigma_c = compute_grads(params, batch)
            state = optimizer.update(grads, state, sigma_c)
    """
    
    def __init__(self, learning_rate: float = 0.01, lambda_critical: float = 0.1):
        if not _HAS_JAX:
            raise ImportError("JAX not installed. Run: pip install jax jaxlib")
        
        self.learning_rate = learning_rate
        self.lambda_critical = lambda_critical
    
    def init(self, params: Any) -> Dict[str, Any]:
        """
        Initialize optimizer state.
        
        Args:
            params: Model parameters
            
        Returns:
            Optimizer state
        """
        return {
            'params': params,
            'step': 0,
            'sigma_c_history': []
        }
    
    def update(self, grads: Any, state: Dict[str, Any], sigma_c: float) -> Dict[str, Any]:
        """
        Update parameters with criticality regularization.
        
        Args:
            grads: Gradients
            state: Optimizer state
            sigma_c: Current criticality
            
        Returns:
            Updated state
        """
        # Standard gradient descent
        if isinstance(state['params'], dict):
            new_params = {
                k: v - self.learning_rate * grads[k]
                for k, v in state['params'].items()
            }
        else:
            new_params = state['params'] - self.learning_rate * grads
        
        # Update state
        new_state = {
            'params': new_params,
            'step': state['step'] + 1,
            'sigma_c_history': state['sigma_c_history'] + [sigma_c]
        }
        
        return new_state
