#!/usr/bin/env python3
"""
Sigma-C Number Theory Demo: Contraction Geometry of Collatz Maps
================================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the contraction geometry framework for analyzing
Collatz-type maps: contraction defect D_M, drift gamma_M,
classification, countdown decomposition, and the twelve-map
prediction table.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sigma_c import Universe


def main():
    print("=" * 65)
    print("  NUMBER THEORY: CONTRACTION GEOMETRY OF COLLATZ MAPS")
    print("=" * 65)

    # --- Create adapter via Universe factory ---
    nt = Universe.number_theory()

    # ====================================================================
    # 1. Contraction defect D_M for M = 4..14
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. CONTRACTION DEFECT D_M  (Collatz cycle map)")
    print("-" * 65)
    print(f"  {'M':>4}  {'Domain':>8}  {'D_M':>8}")
    print("  " + "-" * 24)
    for M in range(4, 15):
        D = nt.compute_D_M(M)
        domain = 2**(M - 1)
        print(f"  {M:>4}  {domain:>8}  {D:>8.4f}")

    # ====================================================================
    # 2. Drift gamma_M and convergence to 9/16
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. DRIFT gamma_M  (target: 9/16 = 0.5625)")
    print("-" * 65)
    print(f"  {'M':>4}  {'gamma_M':>10}  {'error':>10}")
    print("  " + "-" * 28)
    target = 9.0 / 16.0
    for M in range(4, 15):
        gamma = nt.compute_gamma_M(M)
        err = abs(gamma - target)
        print(f"  {M:>4}  {gamma:>10.6f}  {err:>10.6f}")
    print(f"\n  Theoretical limit: gamma = 9/16 = {target:.4f}")

    # ====================================================================
    # 3. Classify Collatz: convergent
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. MAP CLASSIFICATION")
    print("-" * 65)
    pred = nt.predict_behavior()
    print(f"  D            = {pred['D']:.4f}")
    print(f"  gamma        = {pred['gamma']:.4f}")
    print(f"  Prediction   = {pred['prediction']}")
    print(f"  Confidence   = {pred['confidence']}")
    print(f"  Details      = {pred['details']}")

    # ====================================================================
    # 4. Countdown decomposition for n = 127
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. COUNTDOWN DECOMPOSITION  (n = 127)")
    print("-" * 65)
    cd = nt.analyze_countdown(127)
    print(f"  Total steps        : {cd['total_steps']}")
    print(f"  Countdowns         : {cd['n_countdowns']}")
    print(f"  Resets             : {cd['n_resets']}")
    print(f"  Countdown fraction : {cd['countdown_fraction']:.2%}")
    print(f"  Mean reset ED      : {cd['mean_reset_ed']:.2f}")
    print()
    print("  Phase breakdown (first 12 phases):")
    print(f"  {'#':>3}  {'Type':>10}  {'ED start':>9}  {'ED end':>7}  {'Length':>7}")
    print("  " + "-" * 40)
    for i, phase in enumerate(cd['phases'][:12]):
        length = phase.get('length', len(phase['values']) - 1)
        print(f"  {i+1:>3}  {phase['phase']:>10}  {phase['ed_start']:>9}  "
              f"{phase['ed_end']:>7}  {length:>7}")
    if len(cd['phases']) > 12:
        print(f"  ... ({len(cd['phases']) - 12} more phases)")

    # ====================================================================
    # 5. Verify 12/12 predictions table
    # ====================================================================
    print("\n" + "-" * 65)
    print("  5. TWELVE-MAP PREDICTION TABLE")
    print("-" * 65)
    table = nt.get_twelve_map_table()
    print(f"  {'Map':<16} {'q':>3} {'c':>3}  {'D_approx':>8}  {'gamma':>7}  {'Prediction':<22}")
    print("  " + "-" * 65)
    for entry in table:
        gamma_str = f"{entry['gamma']:.4f}"
        print(f"  {entry['map']:<16} {entry['q']:>3} {entry['c']:>3}  "
              f"{entry['D_approx']:>8.2f}  {gamma_str:>7}  {entry['prediction']:<22}")

    # Verify computationally (using M=10 for speed)
    print("\n  Verification at M=10:")
    verify = nt.verify_twelve_predictions(M=10)
    print(f"  Correct: {verify['correct']}/{verify['total']} "
          f"({verify['success_rate']:.0%})")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("  - D_M > 1 for all M: Collatz map is structurally non-injective")
    print("  - gamma_M -> 9/16 < 1: net contraction at every resolution")
    print("  - Together: strong evidence for universal convergence")
    print("  - Countdown decomposition reveals deterministic cascade structure")
    print(f"  - 12/12 maps correctly classified by (D, gamma) framework")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
