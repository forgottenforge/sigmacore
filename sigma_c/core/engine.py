#!/usr/bin/env python3
"""
Sigma-C Core Engine Wrapper
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Python wrapper for the high-performance C++ core engine.
v3.0.0: Added derivative_method parameter and validation fields.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional

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

    v3.0.0: Supports multiple derivative estimation methods via
    the derivative_method parameter. Default behavior unchanged.
    """

    def compute_susceptibility(self,
                               epsilon: np.ndarray,
                               observable: np.ndarray,
                               kernel_sigma: float = 0.6,
                               derivative_method: str = 'gaussian',
                               validate: bool = False) -> Dict[str, Any]:
        """
        Compute susceptibility, sigma_c, and kappa.

        Args:
            epsilon: Scale parameter array
            observable: Observable values array
            kernel_sigma: Gaussian smoothing width (used when method='gaussian')
            derivative_method: 'gaussian' (default), 'savitzky_golay', 'spline',
                               'gp', or 'auto'
            validate: If True, include peak significance and quality metrics
        """
        epsilon = np.ascontiguousarray(epsilon, dtype=np.float64)
        observable = np.ascontiguousarray(observable, dtype=np.float64)

        if derivative_method == 'gaussian' and _HAS_CPP_CORE:
            result = sigma_c_core.SusceptibilityEngine.compute(
                epsilon, observable, kernel_sigma
            )
            chi_array = np.array(result.chi)
            out = {
                'chi': chi_array,
                'sigma_c': result.sigma_c,
                'kappa': result.kappa,
                'chi_max': float(np.max(np.abs(chi_array))),
                'smoothed': np.array(result.smoothed),
                'baseline': result.baseline
            }
        elif derivative_method == 'gaussian':
            out = self._python_fallback(epsilon, observable, kernel_sigma)
        else:
            out = self._alternative_derivative(epsilon, observable, derivative_method,
                                               kernel_sigma=kernel_sigma)

        if validate:
            out.update(self._validate_result(epsilon, observable, out, kernel_sigma))

        return out

    def bootstrap_ci(self, data: np.ndarray, n_reps: int = 1000) -> Tuple[float, float]:
        """
        Compute 95% confidence interval via bootstrap.
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        if _HAS_CPP_CORE:
            return sigma_c_core.StatsEngine.bootstrap_ci(data, n_reps)
        else:
            means = []
            for _ in range(n_reps):
                sample = np.random.choice(data, size=len(data), replace=True)
                means.append(np.mean(sample))
            return np.percentile(means, 2.5), np.percentile(means, 97.5)

    def _python_fallback(self, epsilon, observable, kernel_sigma):
        """Pure-Python fallback when C++ core is not available."""
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(observable, kernel_sigma)
        chi = np.gradient(smoothed, epsilon)
        chi = np.abs(chi)
        idx = int(np.argmax(chi))
        chi_max = float(np.max(chi))
        baseline = float(np.mean(chi)) if len(chi) > 0 else 1.0
        kappa = chi_max / baseline if baseline > 1e-12 else 0.0
        return {
            'chi': chi,
            'sigma_c': float(epsilon[idx]),
            'kappa': kappa,
            'chi_max': chi_max,
            'smoothed': smoothed,
            'baseline': baseline
        }

    def _alternative_derivative(self, epsilon, observable, method, **kwargs):
        """Compute susceptibility using alternative derivative methods."""
        from .derivatives import compute_derivative

        result = compute_derivative(epsilon, observable, method=method, **kwargs)
        chi = result['derivative']
        idx = int(np.argmax(chi))
        chi_max = float(np.max(chi))
        baseline = float(np.mean(chi)) if len(chi) > 0 else 1.0
        kappa = chi_max / baseline if baseline > 1e-12 else 0.0

        out = {
            'chi': chi,
            'sigma_c': float(epsilon[idx]),
            'kappa': kappa,
            'chi_max': chi_max,
            'smoothed': observable,
            'baseline': baseline,
            'derivative_method': result['method_used'],
        }
        if 'uncertainty' in result:
            out['chi_uncertainty'] = result['uncertainty']
        return out

    def _validate_result(self, epsilon, observable, result, kernel_sigma):
        """Add validation fields to result dict."""
        from .validation import permutation_test, peak_clarity_test, observable_quality_score

        extra = {}
        # Peak clarity
        pc = peak_clarity_test(result['kappa'])
        extra['peak_clarity_passes'] = pc['passes']

        # Observable quality (fast, no permutation)
        quality = observable_quality_score(observable, epsilon, kernel_sigma)
        extra['observable_quality'] = quality['score']
        extra['observable_passes'] = quality['passes']

        return extra
