#!/usr/bin/env python3
"""
Sigma-C Core Engine Wrapper
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Python wrapper for the high-performance C++ core engine.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Dict, Any, Tuple
import warnings

try:
    import sigma_c_core
    _HAS_CPP_CORE = True
except ImportError:
    _HAS_CPP_CORE = False
    warnings.warn("Sigma-C C++ Core not found. Falling back to (slow) Python implementation.")

class Engine:
    """
    Wrapper for the high-performance C++ core.
    """
    
    def compute_susceptibility(self, 
                               epsilon: np.ndarray, 
                               observable: np.ndarray,
                               kernel_sigma: float = 0.6) -> Dict[str, Any]:
        """
        Compute susceptibility, sigma_c, and kappa.
        """
        epsilon = np.ascontiguousarray(epsilon, dtype=np.float64)
        observable = np.ascontiguousarray(observable, dtype=np.float64)
        
        if _HAS_CPP_CORE:
            result = sigma_c_core.SusceptibilityEngine.compute(
                epsilon, observable, kernel_sigma
            )
            chi_array = np.array(result.chi)
            return {
                'chi': chi_array,
                'sigma_c': result.sigma_c,
                'kappa': result.kappa,
                'chi_max': float(np.max(np.abs(chi_array))),
                'smoothed': np.array(result.smoothed),
                'baseline': result.baseline
            }
        else:
            return self._python_fallback(epsilon, observable, kernel_sigma)
            
    def bootstrap_ci(self, data: np.ndarray, n_reps: int = 1000) -> Tuple[float, float]:
        """
        Compute 95% confidence interval via bootstrap.
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        if _HAS_CPP_CORE:
            return sigma_c_core.StatsEngine.bootstrap_ci(data, n_reps)
        else:
            # Simple python fallback
            means = []
            for _ in range(n_reps):
                sample = np.random.choice(data, size=len(data), replace=True)
                means.append(np.mean(sample))
            return np.percentile(means, 2.5), np.percentile(means, 97.5)

    def _python_fallback(self, epsilon, observable, kernel_sigma):
        # Minimal fallback implementation
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(observable, kernel_sigma)
        chi = np.gradient(smoothed, epsilon)
        chi = np.abs(chi)
        idx = np.argmax(chi)
        return {
            'chi': chi,
            'sigma_c': epsilon[idx],
            'kappa': 0.0, # Simplified
            'chi_max': float(np.max(chi)),
            'smoothed': smoothed,
            'baseline': 1.0
        }
