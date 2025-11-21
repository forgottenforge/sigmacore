"""
Sigma-C ML Adapter
==================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Machine Learning models.
Provides interfaces for training, evaluation, and robustness measurement.
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any, Callable, Union

class MLAdapter(SigmaCAdapter):
    """
    Adapter for ML frameworks (PyTorch, TensorFlow, Scikit-Learn).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
    def train_and_evaluate(self, system: Union[Callable, Any], params: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulates or performs training and evaluation.
        Returns dict with 'accuracy', 'loss', etc.
        """
        # Simulation logic for now
        lr = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        
        # Simple heuristic: optimal lr around 0.001, optimal batch_size around 64
        lr_score = 1.0 - abs(np.log10(lr) + 3.0) / 3.0  # Peak at lr=0.001
        batch_score = 1.0 - abs(batch_size - 64) / 128.0
        
        accuracy = max(0.0, min(1.0, (lr_score + batch_score) / 2.0))
        return {'accuracy': accuracy}

    def measure_robustness(self, system: Union[Callable, Any], params: Dict[str, Any]) -> Dict[str, float]:
        """
        Measures model robustness (sigma_c).
        """
        # Simulation logic
        dropout = params.get('dropout', 0.0)
        weight_decay = params.get('weight_decay', 0.0)
        
        # Higher regularization -> higher stability
        dropout_score = min(1.0, dropout * 5.0)  # Peak around 0.2
        decay_score = min(1.0, weight_decay * 1000.0)  # Peak around 0.001
        
        sigma_c = (dropout_score + decay_score) / 2.0
        return {'sigma_c': sigma_c}

    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns the primary observable (e.g., loss gradient norm).
        """
        if len(data) == 0: return 0.0
        return float(np.mean(data))
