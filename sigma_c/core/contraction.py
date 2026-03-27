#!/usr/bin/env python3
"""
Contraction Geometry Module
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Implements the fundamental contraction geometry quantities:
- Contraction defect D_M = |domain| / |image|
- Drift gamma = geometric mean growth per step
- Map classification via (D, gamma) decision rule

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Callable, Dict, Any, List, Optional, Set, Tuple
from collections import Counter


def v2(n: int) -> int:
    """2-adic valuation: largest k such that 2^k divides n."""
    if n == 0:
        return float('inf')
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def odd_part(n: int) -> int:
    """Extract odd part: n / 2^v2(n)."""
    if n == 0:
        return 0
    while n % 2 == 0:
        n //= 2
    return n


def single_step_map(n: int, q: int = 3, c: int = 1) -> int:
    """Single-step map: n -> odd(q*n + c) for odd n."""
    return odd_part(q * n + c)


def cycle_map(n: int) -> int:
    """
    Collatz cycle map: F(n) = odd(3^L * (n+1)/2^L - 1)
    where L = v2(n+1) = embedding_depth(n).
    Processes entire countdown in one step.
    """
    L = v2(n + 1)
    s = (n + 1) >> L  # s = (n+1) / 2^L, guaranteed odd
    return odd_part(3**L * s - 1)


def embedding_depth(n: int) -> int:
    """
    Embedding depth: ed(n) = v2(n+1).
    Counts trailing ones in binary representation of n.
    """
    return v2(n + 1)


def compute_contraction_defect(f: Callable[[int], int], M: int) -> float:
    """
    Compute contraction defect D_M = |S_M| / |f(S_M)|.

    Args:
        f: Map on odd integers (e.g., single_step_map, cycle_map)
        M: Modular resolution (residue classes mod 2^M)

    Returns:
        D_M >= 1.0 (structurally non-injective if > 1)
    """
    S_M = [n for n in range(1, 2**M, 2)]  # odd residues
    domain_size = len(S_M)

    image = set()
    for n in S_M:
        fn = f(n) % (2**M)
        # Ensure image element is odd
        if fn % 2 == 0:
            fn = odd_part(fn) % (2**M)
        image.add(fn)

    image_size = len(image)
    if image_size == 0:
        return float('inf')
    return domain_size / image_size


def compute_drift(f: Callable[[int], int], M: int) -> float:
    """
    Compute drift gamma_M via v2-sum method.

    gamma_M = q * 2^(-V_M / |S_M|)
    where V_M = sum of v2(f(x)) over x in S_M for the raw map.

    For generic maps, uses geometric mean of |f(x)/x|.

    Args:
        f: Map on odd integers
        M: Modular resolution

    Returns:
        gamma_M: average multiplicative growth per step
    """
    S_M = [n for n in range(1, 2**M, 2)]
    n_points = len(S_M)

    log_sum = 0.0
    for n in S_M:
        fn = f(n)
        if fn > 0 and n > 0:
            log_sum += np.log2(fn / n)

    return 2.0 ** (log_sum / n_points)


def compute_drift_v2_sum(q: int, M: int) -> float:
    """
    Compute drift for single-step map odd(qn+c) using v2-sum.
    Result: gamma = q/4 in the limit (independent of c).

    Args:
        q: Multiplier (odd, >= 3)
        M: Modular resolution

    Returns:
        gamma_M approaching q/4
    """
    S_M = [n for n in range(1, 2**M, 2)]
    n_points = len(S_M)

    # For odd(qn+c), the v2-sum converges to 2 per element
    # gamma = q * 2^(-mean_v2) = q * 2^(-2) = q/4
    V_M = sum(v2(q * n + 1) for n in S_M)
    mean_v2 = V_M / n_points

    return q * (2.0 ** (-mean_v2))


def classify_map(D: float, gamma: float,
                 has_cycles: bool = False,
                 D_threshold: float = 1.0,
                 gamma_margin: float = 0.05) -> Dict[str, Any]:
    """
    Classify map behavior using (D, gamma) decision rule.

    Args:
        D: Contraction defect
        gamma: Drift
        has_cycles: Whether non-trivial cycles are known
        D_threshold: Threshold for structural non-injectivity
        gamma_margin: Margin around gamma=1 for indeterminate classification

    Returns:
        Dict with 'prediction', 'confidence', 'details'
    """
    is_non_injective = D > D_threshold
    contracting = gamma < (1.0 - gamma_margin)
    expanding = gamma > (1.0 + gamma_margin)
    borderline = not contracting and not expanding

    if not is_non_injective:
        return {
            'prediction': 'injective',
            'confidence': 'high',
            'details': f'D={D:.4f} <= {D_threshold}, map is injective'
        }

    if borderline:
        return {
            'prediction': 'indeterminate',
            'confidence': 'low',
            'details': f'D={D:.4f}>1, gamma={gamma:.4f} near 1 (within {gamma_margin})'
        }

    if contracting:
        if has_cycles:
            prediction = 'convergent_to_cycles'
            details = f'D={D:.4f}>1, gamma={gamma:.4f}<1, cycles present'
        else:
            prediction = 'convergent'
            details = f'D={D:.4f}>1, gamma={gamma:.4f}<1, no cycles known'
    else:
        prediction = 'divergent'
        details = f'D={D:.4f}>1, gamma={gamma:.4f}>1'

    return {
        'prediction': prediction,
        'confidence': 'high',
        'details': details
    }


def sweep_modular_resolution(f: Callable[[int], int],
                             M_range: range,
                             compute_gamma: bool = True) -> List[Dict[str, Any]]:
    """
    Compute (D_M, gamma_M) for a range of modular resolutions.

    Args:
        f: Map on odd integers
        M_range: Range of M values to compute
        compute_gamma: Whether to also compute drift

    Returns:
        List of dicts with 'M', 'D_M', 'gamma_M', 'domain_size', 'image_size'
    """
    results = []
    for M in M_range:
        S_M = [n for n in range(1, 2**M, 2)]
        domain_size = len(S_M)

        image = set()
        for n in S_M:
            fn = f(n) % (2**M)
            if fn % 2 == 0:
                fn = odd_part(fn) % (2**M)
            image.add(fn)
        image_size = len(image)

        D_M = domain_size / image_size if image_size > 0 else float('inf')

        row = {
            'M': M,
            'D_M': D_M,
            'domain_size': domain_size,
            'image_size': image_size,
        }

        if compute_gamma and M >= 4:
            log_sum = 0.0
            count = 0
            for n in S_M:
                fn = f(n)
                if fn > 0 and n > 0:
                    log_sum += np.log2(fn / n)
                    count += 1
            row['gamma_M'] = 2.0 ** (log_sum / count) if count > 0 else None
        else:
            row['gamma_M'] = None

        results.append(row)

    return results


def contraction_principle_check(D_values: List[float],
                                gamma_values: List[float],
                                D_min: float = 1.0,
                                gamma_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Check if (D, gamma) stabilize and satisfy the Contraction Principle.

    Args:
        D_values: Sequence of D_M values for increasing M
        gamma_values: Sequence of gamma_M values for increasing M
        D_min: Minimum D for non-injectivity
        gamma_threshold: Threshold for convergence/divergence

    Returns:
        Dict with 'satisfies_principle', 'D_stable', 'gamma_stable',
        'D_mean', 'gamma_mean', 'prediction'
    """
    D_arr = np.array(D_values)
    gamma_arr = np.array(gamma_values)

    # Check stabilization: CV of last half < 5%
    half = len(D_arr) // 2
    if half < 2:
        return {
            'satisfies_principle': False,
            'reason': 'insufficient_data',
            'D_stable': False,
            'gamma_stable': False,
        }

    D_tail = D_arr[half:]
    gamma_tail = gamma_arr[half:]

    D_cv = np.std(D_tail) / np.mean(D_tail) if np.mean(D_tail) > 0 else float('inf')
    gamma_cv = np.std(gamma_tail) / np.mean(gamma_tail) if np.mean(gamma_tail) > 0 else float('inf')

    D_stable = D_cv < 0.05
    gamma_stable = gamma_cv < 0.05
    D_mean = float(np.mean(D_tail))
    gamma_mean = float(np.mean(gamma_tail))

    all_D_above = bool(np.all(D_arr > D_min))
    gamma_below = gamma_mean < gamma_threshold
    gamma_above = gamma_mean > gamma_threshold

    satisfies = D_stable and gamma_stable and all_D_above

    if satisfies and gamma_below:
        prediction = 'convergent'
    elif satisfies and gamma_above:
        prediction = 'divergent'
    else:
        prediction = 'indeterminate'

    return {
        'satisfies_principle': satisfies,
        'D_stable': D_stable,
        'gamma_stable': gamma_stable,
        'D_mean': D_mean,
        'gamma_mean': gamma_mean,
        'D_cv': float(D_cv),
        'gamma_cv': float(gamma_cv),
        'prediction': prediction,
    }


def countdown_decomposition(n: int, max_steps: int = 1000) -> List[Dict[str, Any]]:
    """
    Decompose a Collatz orbit into countdown and reset phases.

    Each element:
        {'phase': 'countdown'|'reset', 'values': [...], 'ed_start': int, 'ed_end': int}

    Args:
        n: Starting odd integer (must be odd)
        max_steps: Maximum number of single-step iterations

    Returns:
        List of phase dicts
    """
    if n % 2 == 0:
        raise ValueError("n must be odd")

    phases = []
    current = n
    step = 0

    while step < max_steps and current != 1:
        ed = embedding_depth(current)

        if ed >= 2:
            # Countdown phase
            phase_values = [current]
            phase_ed_start = ed

            while ed >= 2 and step < max_steps and current != 1:
                current = single_step_map(current, 3, 1)
                step += 1
                phase_values.append(current)
                ed = embedding_depth(current)

            phases.append({
                'phase': 'countdown',
                'values': phase_values,
                'ed_start': phase_ed_start,
                'ed_end': 1,
                'length': len(phase_values) - 1,
            })
        else:
            # Reset event: ed = 1
            val_before = current
            current = single_step_map(current, 3, 1)
            step += 1
            new_ed = embedding_depth(current) if current != 1 else 0

            phases.append({
                'phase': 'reset',
                'values': [val_before, current],
                'ed_start': 1,
                'ed_end': new_ed,
                'v2_jump': v2(3 * val_before + 1),
            })

    return phases


def deterministic_trajectory(n: int) -> List[int]:
    """
    Compute the closed-form countdown trajectory.
    For n = 2^k * m - 1 with m odd and k >= 2:
        f^j(n) = 3^j * 2^(k-j) * m - 1 for j = 0, ..., k-1

    Args:
        n: Odd integer with ed(n) >= 2

    Returns:
        List of trajectory values during countdown phase
    """
    k = embedding_depth(n)
    if k < 2:
        return [n]

    m = (n + 1) >> k  # m = (n+1) / 2^k, must be odd
    trajectory = []
    for j in range(k):
        val = (3**j) * (2**(k - j)) * m - 1
        trajectory.append(val)
    return trajectory


# Pre-computed reference data for validation
TWELVE_MAP_PREDICTIONS = [
    {'map': '3n+1 (cycle)',   'q': 3,  'c': 1,  'D_approx': 2.06, 'gamma': 9/16,  'cycles': False, 'prediction': 'convergent'},
    {'map': '3n+1 (single)',  'q': 3,  'c': 1,  'D_approx': 1.71, 'gamma': 3/4,   'cycles': False, 'prediction': 'convergent'},
    {'map': '5n+1',           'q': 5,  'c': 1,  'D_approx': 1.43, 'gamma': 5/4,   'cycles': False, 'prediction': 'divergent'},
    {'map': '7n+1',           'q': 7,  'c': 1,  'D_approx': 1.60, 'gamma': 7/4,   'cycles': False, 'prediction': 'divergent'},
    {'map': '3n-1',           'q': 3,  'c': -1, 'D_approx': 1.33, 'gamma': 3/4,   'cycles': True,  'prediction': 'convergent_to_cycles'},
    {'map': '3n+3',           'q': 3,  'c': 3,  'D_approx': 2.00, 'gamma': 3/4,   'cycles': True,  'prediction': 'convergent_to_cycles'},
    {'map': '3n+5',           'q': 3,  'c': 5,  'D_approx': 1.48, 'gamma': 3/4,   'cycles': True,  'prediction': 'convergent_to_cycles'},
    {'map': '3n+7',           'q': 3,  'c': 7,  'D_approx': 1.56, 'gamma': 3/4,   'cycles': True,  'prediction': 'convergent_to_cycles'},
    {'map': '9n+1',           'q': 9,  'c': 1,  'D_approx': 1.34, 'gamma': 9/4,   'cycles': False, 'prediction': 'divergent'},
    {'map': '11n+1',          'q': 11, 'c': 1,  'D_approx': 1.37, 'gamma': 11/4,  'cycles': False, 'prediction': 'divergent'},
    {'map': '5n+3',           'q': 5,  'c': 3,  'D_approx': 1.36, 'gamma': 5/4,   'cycles': False, 'prediction': 'divergent'},
    {'map': '5n-1',           'q': 5,  'c': -1, 'D_approx': 1.43, 'gamma': 5/4,   'cycles': False, 'prediction': 'divergent'},
]
