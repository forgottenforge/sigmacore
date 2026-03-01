#!/usr/bin/env python3
"""
Sigma-C Magnetic Demo: Detecting the Curie Temperature
=======================================================
Copyright (c) 2025 ForgottenForge.xyz

The 2D Ising model has an exact critical temperature (Curie point):
  Tc = 2 / ln(1 + sqrt(2)) = 2.26918...

This demo generates synthetic magnetization data that follows the known
scaling law M ~ |Tc - T|^beta (beta = 1/8 for 2D Ising), then uses
Sigma-C's susceptibility analysis to recover Tc from the data alone.

The aha-moment: Sigma-C finds a value very close to the exact analytical
result, purely from the shape of the magnetization curve.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def generate_ising_data(n_temps=60, noise_level=0.02):
    """Generate synthetic 2D Ising magnetization data."""
    Tc_exact = 2.0 / np.log(1 + np.sqrt(2))  # 2.26918...
    beta_exact = 0.125  # 1/8 for 2D Ising universality class

    temperatures = np.linspace(1.0, 3.5, n_temps)
    magnetization = np.zeros(n_temps)

    for i, T in enumerate(temperatures):
        if T < Tc_exact:
            magnetization[i] = (Tc_exact - T) ** beta_exact
        else:
            magnetization[i] = 0.0
        magnetization[i] += np.random.normal(0, noise_level)
        magnetization[i] = max(0, magnetization[i])

    return temperatures, magnetization, Tc_exact, beta_exact


def main():
    print("=" * 60)
    print("  2D ISING MODEL: CURIE TEMPERATURE DETECTION")
    print("  Using susceptibility analysis to find Tc")
    print("=" * 60)

    np.random.seed(42)
    temperatures, magnetization, Tc_exact, beta_exact = generate_ising_data()

    mag = Universe.magnetic()
    result = mag.compute_susceptibility(temperatures, magnetization)

    sigma_c = result['sigma_c']
    kappa = result['kappa']
    error = abs(sigma_c - Tc_exact)
    error_pct = error / Tc_exact * 100

    print(f"\n  Exact Tc (Onsager 1944):   {Tc_exact:.5f}")
    print(f"  Detected Tc (Sigma-C):     {sigma_c:.5f}")
    print(f"  Absolute error:            {error:.5f}")
    print(f"  Relative error:            {error_pct:.2f}%")
    print(f"  Peak sharpness (kappa):    {kappa:.1f}")

    # Show magnetization profile
    print(f"\n  Magnetization profile:")
    print("  " + "-" * 50)
    step = max(1, len(temperatures) // 20)
    for i in range(0, len(temperatures), step):
        T = temperatures[i]
        M = magnetization[i]
        bar_len = int(M * 40)
        marker = " <-- Tc" if abs(T - sigma_c) < 0.06 else ""
        print(f"  T={T:5.2f} | {'#' * bar_len}{' ' * (40 - bar_len)} | M={M:.3f}{marker}")
    print("  " + "-" * 50)

    # Critical exponent extraction
    chi_data = np.abs(np.gradient(magnetization, temperatures))
    cv_data = np.abs(np.gradient(chi_data, temperatures))

    exponents = mag.analyze_critical_exponents(
        temperatures, magnetization, chi_data, cv_data
    )
    print(f"\n  Critical exponents:")
    print(f"    beta  = {exponents['beta']:.3f}  (exact: {beta_exact})")
    print(f"    gamma = {exponents['gamma']:.3f}  (exact: 1.75)")
    print(f"    T_c   = {exponents['T_c']:.3f}  (exact: {Tc_exact:.3f})")

    print(f"\n  Sigma-C recovers the Curie temperature to within {error_pct:.1f}%")
    print(f"  of the exact analytical result, purely from data.\n")

    # Optionally save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(temperatures, magnetization, 'b-o', markersize=3, label='M(T)')
        ax1.axvline(sigma_c, color='r', linestyle='--', alpha=0.7,
                    label=f'Detected Tc = {sigma_c:.3f}')
        ax1.axvline(Tc_exact, color='g', linestyle=':', alpha=0.7,
                    label=f'Exact Tc = {Tc_exact:.3f}')
        ax1.set_ylabel("Magnetization M")
        ax1.set_title("2D Ising Model Phase Transition")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(temperatures, result['chi'], 'r-s', markersize=3, label='chi(T)')
        ax2.axvline(sigma_c, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel("Temperature T")
        ax2.set_ylabel("Susceptibility chi")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("magnetic_results.png", dpi=150)
        print("  Plot saved to magnetic_results.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
