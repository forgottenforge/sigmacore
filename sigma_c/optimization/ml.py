"""
Balanced ML Optimizer
=====================
Optimizes neural networks and ML models by balancing Accuracy (Performance) 
vs. Robustness/Generalization (Stability/Sigma_c).

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
from .universal import UniversalOptimizer, OptimizationResult

try:
    from ..adapters.ml import MLAdapter
    _HAS_ML_ADAPTER = True
except ImportError:
    _HAS_ML_ADAPTER = False
    # Placeholder for type hints
    class MLAdapter:
        pass

class BalancedMLOptimizer(UniversalOptimizer):
    """
    Optimizes ML models for both performance (accuracy/F1) and stability 
    (adversarial robustness, generalization via sigma_c).
    
    Use cases:
    - Hyperparameter optimization
    - Architecture search
    - Regularization tuning
    - Adversarial training
    """
    
    def __init__(self, adapter: Optional[MLAdapter] = None, 
                 performance_weight: float = 0.7, 
                 stability_weight: float = 0.3):
        super().__init__(performance_weight, stability_weight)
        self.adapter = adapter
        
    def _evaluate_performance(self, system: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate model performance (accuracy, F1, etc.).
        'system' can be a model factory, model instance, or config dict.
        """
        if self.adapter is not None:
            # Use adapter to train and evaluate
            result = self.adapter.train_and_evaluate(system, params)
            return result.get('accuracy', 0.0)
        else:
            # Simulation mode: assume params affect performance
            # Higher learning rate → potentially better performance (up to a point)
            lr = params.get('learning_rate', 0.001)
            batch_size = params.get('batch_size', 32)
            
            # Simple heuristic: optimal lr around 0.001, optimal batch_size around 64
            lr_score = 1.0 - abs(np.log10(lr) + 3.0) / 3.0  # Peak at lr=0.001
            batch_score = 1.0 - abs(batch_size - 64) / 128.0
            
            return max(0.0, min(1.0, (lr_score + batch_score) / 2.0))
    
    def _evaluate_stability(self, system: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate model stability/robustness (generalization, adversarial robustness).
        Higher sigma_c indicates more robust model.
        """
        if self.adapter is not None:
            # Use adapter to measure robustness
            result = self.adapter.measure_robustness(system, params)
            return result.get('sigma_c', 0.0)
        else:
            # Simulation mode: regularization improves stability
            dropout = params.get('dropout', 0.0)
            weight_decay = params.get('weight_decay', 0.0)
            
            # Higher regularization → higher stability (but diminishing returns)
            dropout_score = min(1.0, dropout * 5.0)  # Peak around 0.2
            decay_score = min(1.0, weight_decay * 1000.0)  # Peak around 0.001
            
            return (dropout_score + decay_score) / 2.0
    
    def _apply_params(self, system: Any, params: Dict[str, Any]) -> Any:
        """
        Apply hyperparameters to the model.
        For ML, this typically means creating a new model instance or updating config.
        """
        if callable(system):
            # System is a model factory
            return system(**params)
        elif isinstance(system, dict):
            # System is a config dict
            return {**system, **params}
        else:
            # System is a model instance - return as-is (params applied during training)
            return system
    
    def optimize_model(self, 
                      model_factory: Callable,
                      param_space: Dict[str, List[Any]],
                      strategy: str = 'brute_force') -> OptimizationResult:
        """
        Specialized optimize method for ML models.
        
        Args:
            model_factory: Callable that creates a model given hyperparameters
            param_space: Hyperparameters to optimize
                Example: {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'batch_size': [16, 32, 64],
                    'dropout': [0.0, 0.1, 0.2],
                    'weight_decay': [0.0, 0.0001, 0.001]
                }
            strategy: Optimization strategy ('brute_force' or 'gradient_descent')
            
        Returns:
            OptimizationResult with optimal hyperparameters
        """
        return self.optimize(model_factory, param_space, strategy)
