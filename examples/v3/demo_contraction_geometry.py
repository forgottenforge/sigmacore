#!/usr/bin/env python3
"""
Sigma-C Contraction Geometry Demo: D, gamma, and Information Theory
===================================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the core contraction geometry module: computing
D and gamma for multiple maps, sweeping modular resolution,
contraction principle verification, and Landauer information cost.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c.core.contraction import (
    single_step_map,
    compute_contraction_defect,
    compute_drift,
    sweep_modular_resolution,
    contraction_principle_check,
)
from sigma_c.beyond.information import bits_lost, landauer_cost


def main():
    print("=" * 65)
    print("  CONTRACTION GEOMETRY: D, gamma, AND INFORMATION THEORY")
    print("=" * 65)

    # ====================================================================
    # 1. Compute D and gamma for three maps: 3n+1, 5n+1, 7n+1
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. D AND gamma FOR THREE MAPS (at M=12)")
    print("-" * 65)

    maps = [
        ('3n+1', 3, 1),
        ('5n+1', 5, 1),
        ('7n+1', 7, 1),
    ]

    M = 12
    print(f"  {'Map':<8} {'q':>3}  {'D_M':>8}  {'gamma_M':>9}  {'q/4':>6}  {'Prediction':<12}")
    print("  " + "-" * 52)
    for name, q, c in maps:
        f = lambda n, _q=q, _c=c: single_step_map(n, _q, _c)
        D = compute_contraction_defect(f, M)
        gamma = compute_drift(f, M)
        q_over_4 = q / 4.0
        pred = "convergent" if gamma < 1.0 else "divergent"
        print(f"  {name:<8} {q:>3}  {D:>8.4f}  {gamma:>9.4f}  {q_over_4:>6.2f}  {pred:<12}")

    print("\n  Note: gamma -> q/4 for single-step maps odd(qn+c)")
    print("        3/4 = 0.75 < 1 -> convergent")
    print("        5/4 = 1.25 > 1 -> divergent")
    print("        7/4 = 1.75 > 1 -> divergent")

    # ====================================================================
    # 2. Sweep modular resolution for Collatz (M=4..16)
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. MODULAR RESOLUTION SWEEP (Collatz 3n+1, M=4..14)")
    print("-" * 65)

    f_collatz = lambda n: single_step_map(n, 3, 1)
    sweep = sweep_modular_resolution(f_collatz, range(4, 15))

    print(f"  {'M':>4}  {'|S_M|':>8}  {'|f(S_M)|':>9}  {'D_M':>8}  {'gamma_M':>9}")
    print("  " + "-" * 42)
    D_values = []
    gamma_values = []
    for row in sweep:
        gamma_str = f"{row['gamma_M']:.4f}" if row['gamma_M'] is not None else "N/A"
        print(f"  {row['M']:>4}  {row['domain_size']:>8}  {row['image_size']:>9}  "
              f"{row['D_M']:>8.4f}  {gamma_str:>9}")
        D_values.append(row['D_M'])
        if row['gamma_M'] is not None:
            gamma_values.append(row['gamma_M'])

    # ====================================================================
    # 3. Contraction principle check
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. CONTRACTION PRINCIPLE CHECK")
    print("-" * 65)

    check = contraction_principle_check(D_values, gamma_values)
    print(f"  Satisfies principle : {check['satisfies_principle']}")
    print(f"  D stable           : {check['D_stable']}  (CV = {check.get('D_cv', 0):.4f})")
    print(f"  gamma stable       : {check['gamma_stable']}  (CV = {check.get('gamma_cv', 0):.4f})")
    print(f"  D mean (tail)      : {check.get('D_mean', 0):.4f}")
    print(f"  gamma mean (tail)  : {check.get('gamma_mean', 0):.4f}")
    print(f"  Prediction         : {check.get('prediction', 'N/A')}")

    # ====================================================================
    # 4. Information theory: bits lost & Landauer cost
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. INFORMATION THEORY")
    print("-" * 65)

    print(f"  {'Map':<8} {'D':>6}  {'bits_lost':>10}  {'Landauer (J)':>14}  {'Landauer (eV)':>14}")
    print("  " + "-" * 56)
    for name, q, c in maps:
        f = lambda n, _q=q, _c=c: single_step_map(n, _q, _c)
        D = compute_contraction_defect(f, M)
        bl = bits_lost(D)
        lc = landauer_cost(D, T=300.0)
        lc_ev = lc / 1.602176634e-19
        print(f"  {name:<8} {D:>6.3f}  {bl:>10.4f}  {lc:>14.4e}  {lc_ev:>14.4e}")

    print(f"\n  Physical meaning:")
    print(f"    Each application of 3n+1 erases ~{bits_lost(D_values[-1]):.2f} bits")
    print(f"    At T=300K, minimum energy cost: {landauer_cost(D_values[-1]):.2e} J per step")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("  - D > 1 for all three qn+1 maps: structurally non-injective")
    print("  - gamma ~ q/4: only 3n+1 has gamma < 1 (convergent)")
    print("  - D and gamma stabilize across resolutions (contraction principle)")
    print("  - Information erasure quantifies irreversibility via Landauer bound")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
