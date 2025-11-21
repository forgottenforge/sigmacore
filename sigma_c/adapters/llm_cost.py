"""
Sigma-C LLM Cost Adapter
========================
Copyright (c) 2025 ForgottenForge.xyz

Optimizes Model Selection based on Cost vs. Safety (Hallucination Rate).
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any, List

class LLMCostAdapter(SigmaCAdapter):
    """
    Adapter for Large Language Models.
    Optimizes the trade-off between Inference Cost and Model Quality/Safety.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        if config is None:
            config = {}
        self.budget_per_1k = config.get('budget_per_1k', 0.01) # $0.01 per 1k tokens
        
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns the 'Value Ratio': Quality / Cost.
        """
        # Data: [quality_score, cost_per_query]
        if data.shape[1] < 2: return 0.0
        quality = data[:, 0]
        cost = data[:, 1]
        return float(np.mean(quality / (cost + 1e-9)))
    
    def analyze_cost_safety(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a list of models to find the Pareto frontier of Cost vs Safety.
        
        models: List of dicts with {'name', 'cost', 'hallucination_rate'}
        """
        costs = np.array([m['cost'] for m in models])
        rates = np.array([m['hallucination_rate'] for m in models])
        names = [m['name'] for m in models]
        
        # Calculate Criticality Score: 1 / (Cost * Rate)
        # We want low cost and low rate.
        scores = 1.0 / (costs * rates + 1e-9)
        
        best_idx = np.argmax(scores)
        
        # Calculate sigma_c for the selected model
        # sigma_c = 1 - hallucination_rate (Stability)
        sigma_c = 1.0 - rates[best_idx]
        
        return {
            'best_model': names[best_idx],
            'optimal_cost': float(costs[best_idx]),
            'safety_score': float(1.0 - rates[best_idx]),
            'sigma_c': float(sigma_c)
        }

    def _domain_specific_validate(self, result: Dict[str, Any]) -> bool:
        return result.get('sigma_c', 0.0) > 0.5
