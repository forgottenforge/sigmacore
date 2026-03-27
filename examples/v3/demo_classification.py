#!/usr/bin/env python3
"""
Sigma-C Classification Demo: Four-Type Map Taxonomy
====================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the four-type classification of non-injective
operations: Type D (Dissipative), Type O (Oversaturated),
Type S (Symmetric), and Type R (Reversible).

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import MapType
from sigma_c.core.classification import (
    classify_operation,
    analyze_type_d,
    analyze_type_o,
    analyze_type_s,
    analyze_type_r,
)


def main():
    print("=" * 65)
    print("  MAP CLASSIFICATION: FOUR-TYPE TAXONOMY")
    print("=" * 65)

    # ====================================================================
    # 1. Classify four canonical examples
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. FOUR CANONICAL EXAMPLES")
    print("-" * 65)

    examples = [
        {
            'name': 'Collatz (3n+1)',
            'D': 2.06, 'gamma': 9/16,
            'is_bijective': False,
            'has_symmetry': False,
            'has_growing_preimage': False,
            'expected': 'Type D',
        },
        {
            'name': 'Goldbach sums',
            'D': 1.5, 'gamma': None,
            'is_bijective': False,
            'has_symmetry': False,
            'has_growing_preimage': True,
            'expected': 'Type O',
        },
        {
            'name': 'Riemann zeta zeros',
            'D': 1.0, 'gamma': None,
            'is_bijective': True,
            'has_symmetry': True,
            'has_growing_preimage': False,
            'expected': 'Type S',
        },
        {
            'name': "Noether's theorem",
            'D': 1.0, 'gamma': None,
            'is_bijective': True,
            'has_symmetry': False,
            'has_growing_preimage': False,
            'expected': 'Type R',
        },
    ]

    print(f"  {'System':<22} {'D':>5} {'gamma':>7}  {'Classification':<16} {'Expected':<10}")
    print("  " + "-" * 62)

    for ex in examples:
        map_type = classify_operation(
            D=ex['D'],
            gamma=ex.get('gamma'),
            is_bijective=ex.get('is_bijective'),
            has_symmetry=ex.get('has_symmetry', False),
            has_growing_preimage=ex.get('has_growing_preimage', False),
        )
        gamma_str = f"{ex['gamma']:.4f}" if ex['gamma'] is not None else "  N/A"
        type_str = f"Type {map_type.value} ({map_type.name})"
        match = "ok" if f"Type {map_type.value}" == ex['expected'] else "MISMATCH"
        print(f"  {ex['name']:<22} {ex['D']:>5.2f} {gamma_str:>7}  "
              f"{type_str:<16} {match:<10}")

    # ====================================================================
    # 2. Full Type D analysis for Collatz
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. FULL TYPE D ANALYSIS (Collatz)")
    print("-" * 65)

    result_d = analyze_type_d(D=2.06, gamma=9/16, has_cycles=False)
    d = result_d.to_dict()
    print(f"  Type         : {d['type']}")
    print(f"  D            : {d['D']:.4f}")
    print(f"  gamma        : {d['gamma']:.4f}")
    print(f"  Has cycles   : {d['has_cycles']}")
    print(f"  Prediction   : {d['prediction']}")
    print(f"  Details      : {d['details']}")

    # ====================================================================
    # 3. Type O analysis (Goldbach-like data)
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. TYPE O ANALYSIS (Goldbach-like)")
    print("-" * 65)

    # Simulate Goldbach representation counts vs predictions
    n_range = np.arange(6, 106, 2)
    counts = np.array([1 + int(x * 0.3) for x in range(len(n_range))])
    predictions = np.array([0.8 + x * 0.25 for x in range(len(n_range))])
    result_o = analyze_type_o(counts, predictions, n_range)
    o = result_o.to_dict()
    print(f"  Type                  : {o['type']}")
    print(f"  Oversaturation ratio  : {o['oversaturation_ratio']:.4f}")
    print(f"  Minimum count         : {o['min_count']}")
    print(f"  Reference prediction  : {o['reference_prediction']:.2f}")

    # ====================================================================
    # 4. Type S analysis (symmetry deviations)
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. TYPE S ANALYSIS (Riemann-like symmetry)")
    print("-" * 65)

    np.random.seed(42)
    deviations = np.random.normal(0, 0.42, 200)
    result_s = analyze_type_s(deviations, reference_variance=0.178)
    s = result_s.to_dict()
    print(f"  Type                  : {s['type']}")
    print(f"  Symmetry deviation    : {s['symmetry_deviation']:.6f}")
    print(f"  Constraint tightness  : {s['constraint_tightness']:.4f}")

    # ====================================================================
    # 5. Type R analysis (orbit preservation)
    # ====================================================================
    print("\n" + "-" * 65)
    print("  5. TYPE R ANALYSIS (Noether-like conservation)")
    print("-" * 65)

    # Identity map preserves all orbits
    f_vals = np.arange(10)
    g_orbits = np.array([i % 3 for i in range(10)])
    result_r = analyze_type_r(f_vals, g_orbits)
    r = result_r.to_dict()
    print(f"  Type               : {r['type']}")
    print(f"  Bijective          : {r['is_bijective']}")
    print(f"  Orbits preserved   : {r['orbits_preserved']}")
    print(f"  Conserved quantity : {r['conserved_quantity']}")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 65)
    print("  Type D: D>1, non-injective contraction (Collatz, protein folding)")
    print("  Type O: growing pre-image count (Goldbach, partition functions)")
    print("  Type S: D=1, bijective + symmetry constraint (Riemann zeros)")
    print("  Type R: D=1, bijective preservation (Noether conservation laws)")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
