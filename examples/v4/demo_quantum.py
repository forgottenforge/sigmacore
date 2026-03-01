#!/usr/bin/env python3
"""
Sigma-C Quantum Demo: Finding the Noise Threshold of Grover's Algorithm
========================================================================
Copyright (c) 2025 ForgottenForge.xyz

Every quantum algorithm has a critical noise level beyond which it loses
its advantage over classical computation. This demo uses Sigma-C to find
that exact threshold -- the point where Grover's search breaks down.

The result: a single number (sigma_c) that tells you the maximum gate
error rate your hardware can tolerate before the algorithm stops working.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def main():
    print("=" * 60)
    print("  QUANTUM NOISE THRESHOLD DETECTION")
    print("  Grover's Algorithm on a Simulated 2-Qubit QPU")
    print("=" * 60)

    # Initialize the quantum adapter (uses local simulator)
    qpu = Universe.quantum(device='simulator')

    # Sweep noise levels from 0% to 25% gate error rate
    epsilon_range = np.linspace(0.0, 0.25, 20)

    print(f"\nSweeping {len(epsilon_range)} noise levels from "
          f"{epsilon_range[0]:.1%} to {epsilon_range[-1]:.1%} gate error...")

    result = qpu.run_optimization(
        circuit_type='grover',
        epsilon_values=epsilon_range,
        shots=1000
    )

    sigma_c = result['sigma_c']
    kappa = result['kappa']

    # Display the result
    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)
    print(f"  Critical noise level (sigma_c): {sigma_c:.4f}")
    print(f"  Peak sharpness (kappa):         {kappa:.1f}")
    print()
    print(f"  Interpretation:")
    print(f"    Gate error < {sigma_c:.1%}  ->  Grover's algorithm works")
    print(f"    Gate error > {sigma_c:.1%}  ->  quantum advantage is lost")
    print()

    # Show the observable decay
    print("  Success probability vs. noise:")
    print("  " + "-" * 44)
    for eps, obs in zip(result['epsilon'], result['observable']):
        bar_len = int(obs * 40)
        marker = " <-- sigma_c" if abs(eps - sigma_c) < 0.008 else ""
        print(f"  {eps:5.1%} | {'#' * bar_len}{' ' * (40 - bar_len)} | {obs:.2f}{marker}")
    print("  " + "-" * 44)

    print(f"\n  This tells hardware engineers: keep gate error below {sigma_c:.1%}")
    print(f"  to preserve quantum advantage on this circuit.\n")

    # Optionally save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(result['epsilon'], result['observable'], 'b-o', markersize=4)
        ax1.axvline(sigma_c, color='r', linestyle='--', alpha=0.7,
                    label=f'sigma_c = {sigma_c:.4f}')
        ax1.set_ylabel("Success Probability")
        ax1.set_title("Grover's Algorithm: Noise-Induced Phase Transition")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        chi = np.abs(np.gradient(result['observable'], result['epsilon']))
        ax2.plot(result['epsilon'], chi, 'r-s', markersize=4)
        ax2.axvline(sigma_c, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel("Gate Error Rate")
        ax2.set_ylabel("Susceptibility |chi|")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("quantum_results.png", dpi=150)
        print("  Plot saved to quantum_results.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
