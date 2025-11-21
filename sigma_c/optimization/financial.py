"""
Balanced Financial Optimizer
============================
Optimizes financial strategies by balancing Returns (Performance) vs. Crash Risk (Stability/Sigma_c).

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .universal import UniversalOptimizer, OptimizationResult
from ..adapters.financial import FinancialAdapter

class BalancedFinancialOptimizer(UniversalOptimizer):
    """
    Optimizes trading strategies for both Alpha (Sharpe) and Stability (sigma_c).
    """
    
    def __init__(self, adapter: FinancialAdapter, performance_weight: float = 0.5, stability_weight: float = 0.5):
        super().__init__(performance_weight, stability_weight)
        self.adapter = adapter
        
    def _evaluate_performance(self, system: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate Sharpe Ratio.
        """
        symbol = params.get('symbol', '^GSPC')
        lookback = int(params.get('lookback', 252))
        
        try:
            df = self.adapter.fetch_market_data(symbol)
            returns = df['Return'].values[-lookback:]
            
            if len(returns) < 10:
                return 0.0
                
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            
            if std_ret == 0:
                return 0.0
                
            # Annualized Sharpe (assuming daily data)
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return sharpe
            
        except Exception:
            return 0.0

    def _evaluate_stability(self, system: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate Crash Risk (sigma_c).
        Higher sigma_c means more stable (less prone to critical transitions).
        """
        symbol = params.get('symbol', '^GSPC')
        lookback = int(params.get('lookback', 252))
        
        try:
            result = self.adapter.detect_regime(symbol, window_days=lookback)
            return result['sigma_c']
        except Exception:
            return 0.0

    def _apply_params(self, system: Any, params: Dict[str, Any]) -> Any:
        # Financial adapter is stateless regarding strategy params
        return system

    def optimize_strategy(self, 
                         param_space: Dict[str, List[Any]],
                         strategy: str = 'brute_force') -> OptimizationResult:
        """
        Specialized optimize method for strategies.
        """
        return self.optimize(None, param_space, strategy)
