#!/usr/bin/env python3
"""
Sigma-C Financial Demo: Volatility Regime Detection
=====================================================
Copyright (c) 2025 ForgottenForge.xyz

Financial markets exhibit volatility clustering: large price moves tend
to be followed by more large moves. This is quantified by a GARCH(1,1)
model where the persistence parameter (alpha + beta) approaching 1.0
signals criticality -- the market is on the edge of a regime change.

This demo generates two synthetic return series (calm vs. turbulent),
fits GARCH models, and shows how Sigma-C detects the different regimes.

No external data or yfinance required.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def generate_garch_returns(n=2000, omega=1e-6, alpha=0.1, beta=0.85, seed=42):
    """Generate synthetic returns from a GARCH(1,1) process."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega * 100

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = rng.normal(0, np.sqrt(max(sigma2[t], 1e-12)))

    return returns


def main():
    print("=" * 60)
    print("  FINANCIAL VOLATILITY REGIME DETECTION")
    print("  GARCH(1,1) Analysis with Sigma-C")
    print("=" * 60)

    fin = Universe.finance()

    # Scenario 1: Calm market (low persistence)
    print("\n--- Scenario 1: Calm Market ---")
    calm_returns = generate_garch_returns(
        n=2000, omega=1e-6, alpha=0.05, beta=0.70, seed=42
    )
    calm = fin.analyze_volatility_clustering(calm_returns)
    print(f"  GARCH params:  omega={calm['omega']:.2e}, alpha={calm['alpha']:.3f}, beta={calm['beta']:.3f}")
    print(f"  Persistence:   {calm['persistence']:.3f}  (alpha + beta)")
    print(f"  Sigma-C:       {calm['sigma_c']:.4f}")
    print(f"  Regime:        {'Critical' if calm['persistence'] > 0.95 else 'Stable'}")

    # Scenario 2: Pre-crisis market (high persistence)
    print("\n--- Scenario 2: Pre-Crisis Market ---")
    crisis_returns = generate_garch_returns(
        n=2000, omega=1e-7, alpha=0.12, beta=0.87, seed=123
    )
    crisis = fin.analyze_volatility_clustering(crisis_returns)
    print(f"  GARCH params:  omega={crisis['omega']:.2e}, alpha={crisis['alpha']:.3f}, beta={crisis['beta']:.3f}")
    print(f"  Persistence:   {crisis['persistence']:.3f}  (alpha + beta)")
    print(f"  Sigma-C:       {crisis['sigma_c']:.4f}")
    print(f"  Regime:        {'Critical' if crisis['persistence'] > 0.95 else 'Stable'}")

    # Hurst exponent analysis
    print("\n--- Hurst Exponent Analysis ---")
    hurst_calm = fin.compute_hurst_exponent(calm_returns)
    hurst_crisis = fin.compute_hurst_exponent(crisis_returns)
    print(f"  Calm market:   H = {hurst_calm['hurst']:.3f}  ({hurst_calm['regime']})")
    print(f"  Crisis market: H = {hurst_crisis['hurst']:.3f}  ({hurst_crisis['regime']})")
    print(f"    H < 0.5: mean-reverting | H = 0.5: random walk | H > 0.5: trending")

    # Summary
    print("\n" + "-" * 60)
    print("  SUMMARY")
    print("-" * 60)
    print(f"  The pre-crisis market shows persistence {crisis['persistence']:.3f} vs {calm['persistence']:.3f}")
    print(f"  for the calm market. As persistence -> 1.0, volatility shocks")
    print(f"  persist indefinitely (criticality). Sigma-C quantifies this")
    print(f"  transition with a single number.\n")


if __name__ == "__main__":
    main()
