#!/usr/bin/env python3
"""
Sigma-C Information Theory Demo: Bits Lost and Landauer Principle
=================================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the information-theoretic module: bits lost from
non-injective maps, Landauer energy cost, complete information
summary, and cross-domain comparison.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sigma_c.beyond.information import (
    bits_lost,
    landauer_cost,
    information_summary,
)


def main():
    print("=" * 65)
    print("  INFORMATION THEORY: BITS LOST & LANDAUER PRINCIPLE")
    print("=" * 65)

    # ====================================================================
    # 1. Bits lost for different D values
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. INFORMATION LOSS vs CONTRACTION DEFECT")
    print("-" * 65)

    D_values = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    print(f"  {'D':>6}  {'bits_lost':>10}  {'interpretation'}")
    print("  " + "-" * 50)
    for D in D_values:
        bl = bits_lost(D)
        if D == 1.0:
            interp = "bijective (no erasure)"
        elif bl < 0.5:
            interp = "weak contraction"
        elif bl < 1.0:
            interp = "moderate contraction"
        else:
            interp = "strong contraction"
        print(f"  {D:>6.2f}  {bl:>10.4f}  {interp}")

    # ====================================================================
    # 2. Landauer cost at room temperature
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. LANDAUER ENERGY COST (T = 300 K)")
    print("-" * 65)

    print(f"  {'D':>6}  {'bits':>6}  {'Energy (J)':>14}  {'Energy (eV)':>14}")
    print("  " + "-" * 44)
    for D in [1.5, 2.0, 2.5, 3.0]:
        bl = bits_lost(D)
        lc = landauer_cost(D, T=300.0)
        lc_ev = lc / 1.602176634e-19
        print(f"  {D:>6.2f}  {bl:>6.3f}  {lc:>14.4e}  {lc_ev:>14.4e}")

    print("\n  k_B * T * ln(2) at 300K = 2.87e-21 J per bit erased")

    # ====================================================================
    # 3. Information summary for Collatz
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. COMPLETE INFORMATION SUMMARY (Collatz)")
    print("-" * 65)

    collatz_summary = information_summary(D=2.06, gamma=9/16, T=300.0)
    print(f"  D                      : {collatz_summary['D']:.4f}")
    print(f"  gamma                  : {collatz_summary['gamma']:.4f}")
    print(f"  sigma = D * gamma      : {collatz_summary['sigma_product']:.4f}")
    print(f"  Bits lost per step     : {collatz_summary['bits_lost_per_step']:.4f}")
    print(f"  Landauer cost (J)      : {collatz_summary['landauer_cost_J']:.4e}")
    print(f"  Landauer cost (eV)     : {collatz_summary['landauer_cost_eV']:.4e}")
    print(f"  Entropy production     : {collatz_summary['entropy_production_rate']:.4e} J/(K*s)")
    print(f"  Net contraction        : {collatz_summary['net_contraction']}")
    print(f"  Interpretation         : {collatz_summary['interpretation']}")

    # ====================================================================
    # 4. Cross-domain comparison: Collatz vs 5n+1
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. CROSS-DOMAIN COMPARISON")
    print("-" * 65)

    systems = [
        ('Collatz (3n+1)', 2.06, 9/16),
        ('5n+1',           1.43, 5/4),
    ]

    print(f"  {'System':<18} {'D':>6} {'gamma':>7} {'sigma':>7} {'bits':>6} {'contraction'}")
    print("  " + "-" * 60)
    for name, D, gamma in systems:
        s = information_summary(D, gamma)
        contract = "YES" if s['net_contraction'] else "NO"
        print(f"  {name:<18} {D:>6.2f} {gamma:>7.4f} {s['sigma_product']:>7.4f} "
              f"{s['bits_lost_per_step']:>6.3f} {contract}")

    print("\n  Key insight:")
    print("    Collatz: sigma < 1 -> net contraction despite information erasure")
    print("    5n+1:    sigma > 1 -> net expansion (gamma dominates)")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("  - Non-injective maps erase log2(D) bits per step")
    print("  - Landauer principle: minimum energy = k_B*T*ln(2) per bit")
    print("  - sigma = D*gamma: net contraction index")
    print("  - sigma < 1 predicts convergence, sigma > 1 predicts divergence")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
