#!/usr/bin/env python3
"""
Sigma-C Formal Validation Demo: Statistical Tests for sigma_c
=============================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the formal validation tools: engine susceptibility
computation with multiple derivative methods, permutation test,
peak clarity test, and observable quality scoring.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c.core.engine import Engine
from sigma_c.core.validation import (
    permutation_test,
    peak_clarity_test,
    observable_quality_score,
)


def main():
    print("=" * 65)
    print("  FORMAL VALIDATION: STATISTICAL TESTS FOR SIGMA_C")
    print("=" * 65)

    # ====================================================================
    # 1. Create test data with clear sigma_c peak
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. SYNTHETIC TEST DATA")
    print("-" * 65)

    np.random.seed(42)
    epsilon = np.linspace(0.0, 1.0, 50)
    # Sigmoid-like observable with transition at sigma_c ~ 0.4
    observable = 1.0 / (1.0 + np.exp(20 * (epsilon - 0.4)))
    observable += np.random.normal(0, 0.02, len(epsilon))

    print(f"  Data points   : {len(epsilon)}")
    print(f"  Epsilon range : [{epsilon[0]:.2f}, {epsilon[-1]:.2f}]")
    print(f"  Observable    : sigmoid transition + noise")
    print(f"  True sigma_c  : ~0.40 (designed transition point)")

    # ====================================================================
    # 2. Engine compute_susceptibility with different methods
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. SUSCEPTIBILITY COMPUTATION (multiple methods)")
    print("-" * 65)

    engine = Engine()
    methods = ['gaussian', 'savitzky_golay', 'spline']

    print(f"  {'Method':<18} {'sigma_c':>9} {'kappa':>8} {'chi_max':>10}")
    print("  " + "-" * 48)

    results = {}
    for method in methods:
        try:
            r = engine.compute_susceptibility(
                epsilon, observable,
                kernel_sigma=0.6,
                derivative_method=method,
            )
            results[method] = r
            print(f"  {method:<18} {r['sigma_c']:>9.4f} {r['kappa']:>8.2f} "
                  f"{r['chi_max']:>10.4f}")
        except Exception as e:
            print(f"  {method:<18} ERROR: {e}")

    if results:
        sigma_vals = [r['sigma_c'] for r in results.values()]
        print(f"\n  sigma_c range across methods: "
              f"[{min(sigma_vals):.4f}, {max(sigma_vals):.4f}]")
        print(f"  Spread: {max(sigma_vals) - min(sigma_vals):.4f}")

    # ====================================================================
    # 3. Permutation test
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. PERMUTATION TEST (n=1000 permutations)")
    print("-" * 65)

    perm = permutation_test(epsilon, observable, n_permutations=1000)
    print(f"  Observed kappa  : {perm['observed_kappa']:.4f}")
    print(f"  Null mean       : {perm['null_mean']:.4f}")
    print(f"  Null std        : {perm['null_std']:.4f}")
    print(f"  Null 95th pct   : {perm['null_95th']:.4f}")
    print(f"  p-value         : {perm['p_value']:.4f}")
    sig = "SIGNIFICANT" if perm['significant'] else "not significant"
    print(f"  Result          : {sig} (alpha=0.05)")

    # ====================================================================
    # 4. Peak clarity test
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. PEAK CLARITY TEST")
    print("-" * 65)

    # Use the kappa from the gaussian method
    if 'gaussian' in results:
        kappa_val = results['gaussian']['kappa']
    elif results:
        kappa_val = list(results.values())[0]['kappa']
    else:
        kappa_val = perm['observed_kappa']

    pc = peak_clarity_test(kappa_val)
    print(f"  kappa           : {pc['kappa']:.2f}")
    print(f"  Threshold       : {pc['threshold']:.1f}")
    print(f"  Margin          : {pc['margin']:+.2f}")
    print(f"  Passes          : {pc['passes']}")
    print(f"  Interpretation  : {pc['interpretation']}")

    # ====================================================================
    # 5. Observable quality score
    # ====================================================================
    print("\n" + "-" * 65)
    print("  5. OBSERVABLE QUALITY SCORE")
    print("-" * 65)

    quality = observable_quality_score(observable, epsilon)
    print(f"  Overall score    : {quality['score']:.2f}")
    print(f"  Passes           : {quality['passes']}")
    print(f"  Recommendation   : {quality['recommendation']}")
    print()
    print("  Criteria breakdown:")
    for name, crit in quality['criteria'].items():
        status = "PASS" if crit['passes'] else "FAIL"
        print(f"    {name:<22} {status:<6} "
              f"(value={crit['value']:.4f}, threshold={crit['threshold']})")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("  - Multiple derivative methods give consistent sigma_c estimates")
    print("  - Permutation test confirms statistical significance of peak")
    print("  - Peak clarity (kappa) measures signal strength vs noise")
    print("  - Observable quality score guides data suitability")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
