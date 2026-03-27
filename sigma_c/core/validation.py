#!/usr/bin/env python3
"""
Formal Sigma-C Validation Module
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Implements formal validation tools for sigma_c measurements:
- Boundary condition check (existence proof)
- Permutation test (statistical significance)
- Peak clarity test (kappa threshold)
- Fisher information bound
- Observable quality scoring (decision tree)

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple


def check_boundary_conditions(observable_values: np.ndarray,
                              epsilon_values: np.ndarray) -> Dict[str, Any]:
    """
    Check the boundary conditions for sigma_c existence.

    From the existence proof: if f is non-injective on a finite set with
    O(sigma_min) = 1 and O(sigma_max) < 1, then chi has an interior maximum.

    Args:
        observable_values: O(sigma) values
        epsilon_values: sigma (scale parameter) values

    Returns:
        Dict with 'exists', 'O_min', 'O_max', 'is_nonconstant', 'details'
    """
    O_min = float(observable_values[0])
    O_max = float(observable_values[-1])
    O_range = float(np.max(observable_values) - np.min(observable_values))
    is_nonconstant = O_range > 1e-10

    # Boundary condition: O decreases from start to end (non-injective)
    decreasing = O_max < O_min

    # Chi (susceptibility) must have interior maximum
    chi = np.abs(np.gradient(observable_values, epsilon_values))
    chi_start = float(chi[0])
    chi_end = float(chi[-1])
    chi_max = float(np.max(chi))
    has_interior_max = chi_max > chi_start and chi_max > chi_end

    exists = is_nonconstant and (decreasing or has_interior_max)

    details = (f'O(sigma_min)={O_min:.4f}, O(sigma_max)={O_max:.4f}, '
               f'range={O_range:.4f}, chi_max={chi_max:.4f}')

    return {
        'exists': exists,
        'O_min': O_min,
        'O_max': O_max,
        'is_nonconstant': is_nonconstant,
        'has_interior_maximum': has_interior_max,
        'decreasing_boundary': decreasing,
        'details': details,
    }


def permutation_test(epsilon: np.ndarray, observable: np.ndarray,
                     n_permutations: int = 10000,
                     kernel_sigma: float = 0.6) -> Dict[str, Any]:
    """
    Permutation test for sigma_c significance.
    H0: No characteristic scale exists (peak is due to noise).

    Args:
        epsilon: Scale parameter values
        observable: Observed values
        n_permutations: Number of permutations
        kernel_sigma: Smoothing parameter

    Returns:
        Dict with 'p_value', 'observed_kappa', 'null_kappas', 'significant'
    """
    from scipy.ndimage import gaussian_filter1d

    # Compute observed kappa
    smoothed = gaussian_filter1d(observable, kernel_sigma)
    chi = np.abs(np.gradient(smoothed, epsilon))
    chi_max = float(np.max(chi))
    baseline = float(np.mean(chi))
    observed_kappa = chi_max / baseline if baseline > 1e-12 else 0.0

    # Generate null distribution
    null_kappas = []
    for _ in range(n_permutations):
        perm_obs = np.random.permutation(observable)
        perm_smooth = gaussian_filter1d(perm_obs, kernel_sigma)
        perm_chi = np.abs(np.gradient(perm_smooth, epsilon))
        perm_max = float(np.max(perm_chi))
        perm_base = float(np.mean(perm_chi))
        perm_kappa = perm_max / perm_base if perm_base > 1e-12 else 0.0
        null_kappas.append(perm_kappa)

    null_kappas = np.array(null_kappas)

    # p-value: fraction of null kappas >= observed
    p_value = float(np.sum(null_kappas >= observed_kappa) + 1) / (n_permutations + 1)

    return {
        'p_value': p_value,
        'observed_kappa': observed_kappa,
        'null_mean': float(np.mean(null_kappas)),
        'null_std': float(np.std(null_kappas)),
        'null_95th': float(np.percentile(null_kappas, 95)),
        'significant': p_value < 0.05,
    }


def peak_clarity_test(kappa: float, kappa_min: float = 3.0) -> Dict[str, Any]:
    """
    Test if peak clarity exceeds minimum threshold.

    Args:
        kappa: Observed peak clarity (chi_max / mean(chi))
        kappa_min: Minimum threshold (default 3.0)

    Returns:
        Dict with 'passes', 'kappa', 'threshold', 'margin'
    """
    passes = kappa >= kappa_min
    margin = kappa - kappa_min

    return {
        'passes': passes,
        'kappa': kappa,
        'threshold': kappa_min,
        'margin': margin,
        'interpretation': (
            f'kappa={kappa:.2f} {"exceeds" if passes else "below"} '
            f'threshold {kappa_min:.1f} (margin: {margin:+.2f})'
        ),
    }


def fisher_information_bound(epsilon: np.ndarray, observable: np.ndarray,
                             observable_variance: Optional[np.ndarray] = None
                             ) -> Dict[str, Any]:
    """
    Compute Fisher information bound for susceptibility.
    chi(sigma) >= |dg/dsigma| / sqrt(I_F)

    Args:
        epsilon: Scale parameter values
        observable: Mean observable values
        observable_variance: Variance of observable at each scale (optional)

    Returns:
        Dict with 'fisher_information', 'cramer_rao_bound', 'bound_saturated'
    """
    dg = np.gradient(observable, epsilon)

    if observable_variance is not None and np.any(observable_variance > 0):
        # Fisher information from variance
        I_F = 1.0 / (observable_variance + 1e-12)
    else:
        # Estimate from data: use gradient magnitude as proxy
        I_F = dg**2 / (np.var(observable) + 1e-12) * np.ones_like(dg)

    # Cramer-Rao lower bound on chi
    chi_lower = np.abs(dg) / np.sqrt(I_F + 1e-12)

    # Observed chi
    chi_observed = np.abs(dg)

    # Check if bound is approximately saturated
    ratio = chi_observed / (chi_lower + 1e-12)
    saturated = bool(np.mean(ratio[ratio > 0]) < 2.0) if np.any(ratio > 0) else False

    return {
        'fisher_information': I_F,
        'cramer_rao_bound': chi_lower,
        'chi_observed': chi_observed,
        'bound_saturated': saturated,
        'mean_fisher': float(np.mean(I_F)),
    }


def observable_quality_score(data: np.ndarray, epsilon: np.ndarray,
                             kernel_sigma: float = 0.6) -> Dict[str, Any]:
    """
    Score observable quality using the decision tree from ASBMFCSI.

    Criteria:
    1. Scale sensitivity: CV > 0.3
    2. Signal-to-noise ratio: SNR > 10
    3. Sufficient data points: n > 10
    4. Non-trivial variation: range > 0

    Args:
        data: Observable values
        epsilon: Scale parameter values
        kernel_sigma: Smoothing parameter for SNR estimation

    Returns:
        Dict with 'score', 'passes', 'criteria', 'recommendation'
    """
    from scipy.ndimage import gaussian_filter1d

    n = len(data)
    criteria = {}

    # 1. Scale sensitivity (coefficient of variation)
    cv = float(np.std(data) / (np.abs(np.mean(data)) + 1e-12))
    criteria['scale_sensitivity'] = {
        'value': cv,
        'threshold': 0.3,
        'passes': cv > 0.3,
    }

    # 2. Signal-to-noise ratio
    smoothed = gaussian_filter1d(data, kernel_sigma)
    signal_power = np.var(smoothed)
    noise_power = np.var(data - smoothed)
    snr = signal_power / (noise_power + 1e-12)
    criteria['snr'] = {
        'value': float(snr),
        'threshold': 10.0,
        'passes': snr > 10.0,
    }

    # 3. Sufficient data
    criteria['sufficient_data'] = {
        'value': n,
        'threshold': 10,
        'passes': n > 10,
    }

    # 4. Non-trivial variation
    data_range = float(np.max(data) - np.min(data))
    criteria['nontrivial_range'] = {
        'value': data_range,
        'threshold': 0.0,
        'passes': data_range > 1e-10,
    }

    n_passing = sum(1 for c in criteria.values() if c['passes'])
    total = len(criteria)
    score = n_passing / total

    if score == 1.0:
        recommendation = 'Observable is well-suited for sigma_c analysis.'
    elif score >= 0.75:
        failed = [k for k, v in criteria.items() if not v['passes']]
        recommendation = f'Observable is acceptable but consider improving: {", ".join(failed)}.'
    elif score >= 0.5:
        recommendation = 'Observable has significant quality issues. Consider alternative observables.'
    else:
        recommendation = 'Observable is not suitable for sigma_c analysis.'

    return {
        'score': score,
        'passes': score >= 0.75,
        'criteria': criteria,
        'recommendation': recommendation,
    }
