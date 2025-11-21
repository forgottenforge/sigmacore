"""
Sigma-C PyTorch Integration
============================
Copyright (c) 2025 ForgottenForge.xyz

Criticality-aware neural network training.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = None

from ..core.engine import Engine


class CriticalModule(nn.Module if _HAS_TORCH else object):
    """
    Base class for criticality-aware PyTorch modules.
    
    Usage:
        class MyNet(CriticalModule):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 1)
            
            def forward(self, x):
                return self.critical_forward(x)
    """
    
    def __init__(self):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        super().__init__()
        self.sigma_c_history = []
        self.enable_criticality_tracking = True
    
    def critical_forward(self, x):
        """
        Forward pass with automatic criticality tracking.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with sigma_c metadata
        """
        if not self.enable_criticality_tracking:
            return self.forward(x)
        
        # Capture pre-state
        pre_activations = []
        
        def hook(module, input, output):
            pre_activations.append(output.detach().cpu().numpy())
        
        # Register hooks
        handles = []
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handles.append(module.register_forward_hook(hook))
        
        # Forward pass
        output = self.forward(x)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Compute criticality from activation distribution
        if pre_activations:
            activations = np.concatenate([a.flatten() for a in pre_activations])
            sigma_c = self._compute_activation_criticality(activations)
            self.sigma_c_history.append(sigma_c)
        
        return output
    
    def _compute_activation_criticality(self, activations: np.ndarray) -> float:
        """
        Compute criticality from activation statistics.
        
        Near criticality: activations have power-law distribution
        """
        # Compute variance as proxy for criticality
        # High variance = near phase transition
        var = np.var(activations)
        mean_abs = np.mean(np.abs(activations))
        
        # Normalized criticality score
        if mean_abs > 0:
            sigma_c = var / (mean_abs + 1e-9)
        else:
            sigma_c = 0.0
        
        return float(np.clip(sigma_c, 0, 1))
    
    def get_criticality_report(self) -> Dict[str, Any]:
        """
        Get criticality statistics over training.
        
        Returns:
            Dictionary with sigma_c statistics
        """
        if not self.sigma_c_history:
            return {'mean_sigma_c': 0.0, 'std_sigma_c': 0.0, 'samples': 0}
        
        history = np.array(self.sigma_c_history)
        return {
            'mean_sigma_c': float(np.mean(history)),
            'std_sigma_c': float(np.std(history)),
            'min_sigma_c': float(np.min(history)),
            'max_sigma_c': float(np.max(history)),
            'samples': len(history)
        }


class SigmaCLoss(nn.Module if _HAS_TORCH else object):
    """
    Loss function that penalizes/rewards criticality.
    
    Usage:
        criterion = SigmaCLoss(lambda_critical=0.1, target_sigma_c=0.5)
        loss = criterion(output, target, model)
    """
    
    def __init__(self, lambda_critical: float = 0.1, target_sigma_c: float = 0.5):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not installed")
        super().__init__()
        self.lambda_critical = lambda_critical
        self.target_sigma_c = target_sigma_c
        self.base_criterion = nn.MSELoss()
    
    def forward(self, output, target, model: Optional[CriticalModule] = None):
        """
        Compute loss with criticality regularization.
        
        Args:
            output: Model predictions
            target: Ground truth
            model: CriticalModule instance (optional)
            
        Returns:
            Combined loss
        """
        # Base loss
        base_loss = self.base_criterion(output, target)
        
        # Criticality penalty
        if model is not None and hasattr(model, 'sigma_c_history'):
            if model.sigma_c_history:
                current_sigma_c = model.sigma_c_history[-1]
                # Penalize deviation from target
                critical_loss = (current_sigma_c - self.target_sigma_c) ** 2
                total_loss = base_loss + self.lambda_critical * critical_loss
                return total_loss
        
        return base_loss


def critical_jit(func: Callable) -> Callable:
    """
    Decorator for criticality-aware JIT compilation (PyTorch 2.0+).
    
    Usage:
        @critical_jit
        def train_step(model, batch):
            ...
    """
    if not _HAS_TORCH:
        return func
    
    # Wrap with torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        compiled = torch.compile(func)
        
        def wrapper(*args, **kwargs):
            result = compiled(*args, **kwargs)
            # Attach criticality metadata
            if hasattr(result, '__dict__'):
                result.__sigma_c__ = 0.5  # Placeholder
            return result
        
        return wrapper
    
    return func
