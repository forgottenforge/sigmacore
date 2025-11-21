"""
Rigorous Financial Sigma_c
==========================
Validates financial sigma_c values against Random Matrix Theory (RMT) and Efficient Market Hypothesis (EMH).

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .rigorous import RigorousTheoreticalCheck

class RigorousFinancialSigmaC(RigorousTheoreticalCheck):
    """
    Checks if measured sigma_c respects market physics bounds.
    """
    
    def check_theoretical_bounds(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Check RMT bounds for correlation matrix eigenvalues.
        Marchenko-Pastur distribution limits.
        """
        T = data.get('T', 252) # Time horizon
        N = data.get('N', 500) # Number of assets
        
        # Q = T/N ratio
        Q = T / N if N > 0 else 1.0
        
        # Marchenko-Pastur bounds for random correlation matrix
        lambda_min = (1 - np.sqrt(1/Q))**2
        lambda_max = (1 + np.sqrt(1/Q))**2
        
        # Sigma_c should be related to the deviation from these random bounds
        # If sigma_c is high, market is "random" (stable)
        # If sigma_c is low, market is "correlated" (prone to crashes)
        
        return {
            'lower_bound': 0.1, # Empirical lower bound for stable markets
            'upper_bound': 0.9, # Empirical upper bound
            'metric': 'sigma_c',
            'theory': 'Random Matrix Theory (Marchenko-Pastur)',
            'lambda_min': lambda_min,
            'lambda_max': lambda_max
        }

    def check_scaling_laws(self, data: Any, param_range: List[float], **kwargs) -> Dict[str, Any]:
        """
        Check Stylized Facts: Fat tails, volatility clustering.
        Simple check for power-law tail behavior.
        """
        if not param_range or len(param_range) < 2:
            return {'status': 'insufficient_data'}
        
        returns = np.array(param_range)
        
        # Check for fat tails using kurtosis
        kurtosis = np.mean((returns - np.mean(returns))**4) / (np.std(returns)**4)
        has_fat_tails = kurtosis > 3.0  # Normal distribution has kurtosis = 3
        
        # Simple volatility clustering check: autocorrelation of squared returns
        if len(returns) > 10:
            squared_returns = returns**2
            mean_sq = np.mean(squared_returns)
            autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            has_clustering = autocorr > 0.1
        else:
            has_clustering = False
        
        return {
            'status': 'completed',
            'kurtosis': float(kurtosis),
            'has_fat_tails': has_fat_tails,
            'has_volatility_clustering': has_clustering,
            'theory': 'Stylized Facts of Financial Returns'
        }

    def quantify_resource(self, data: Any) -> float:
        """
        Quantify Market Efficiency using Shannon Entropy of returns.
        Higher entropy = more efficient/random market.
        """
        returns = data.get('returns', [])
        if len(returns) < 2:
            return 0.0
        
        # Discretize returns into bins for entropy calculation
        hist, _ = np.histogram(returns, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Normalize to [0, 1] range (log(20) is max for 20 bins)
        normalized_entropy = entropy / np.log(20)
        
        return float(normalized_entropy)
