#!/usr/bin/env python3
"""
Sigma-C Diagnostics Demo: Universal Diagnostics System
========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates the diagnostics system: diagnose(), auto_search(),
validate_techniques(), and explain() across multiple domains.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def demo_gpu_diagnostics():
    print("\n" + "=" * 60)
    print("  GPU DIAGNOSTICS")
    print("=" * 60)

    gpu = Universe.gpu()

    # Run diagnostics
    diag = gpu.diagnose()
    print(f"\n  Status:          {diag['status']}")
    print(f"  Issues:          {diag['issues'] or 'None'}")
    print(f"  Recommendations: {diag['recommendations'] or 'None'}")

    # Validate techniques
    validation = gpu.validate_techniques()
    print(f"\n  Technique validation:")
    for check, passed in validation['checks'].items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {check}")


def demo_quantum_diagnostics():
    print("\n" + "=" * 60)
    print("  QUANTUM DIAGNOSTICS")
    print("=" * 60)

    qpu = Universe.quantum(device='simulator')

    # Run a quick optimization
    result = qpu.run_optimization(
        circuit_type='grover',
        epsilon_values=np.linspace(0.02, 0.20, 8),
        shots=100
    )

    # Explain results
    explanation = qpu.explain(result)
    print(explanation)


def demo_financial_diagnostics():
    print("\n" + "=" * 60)
    print("  FINANCIAL DIAGNOSTICS")
    print("=" * 60)

    fin = Universe.finance()

    # Generate synthetic price data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))

    try:
        import pandas as pd
        price_series = pd.Series(prices, index=pd.date_range('2020-01-01', periods=300))
        diag = fin.diagnose(price_data=price_series)
        print(f"\n  Status:  {diag['status']}")
        print(f"  Issues:  {diag['issues'] or 'None'}")
        print(f"  Details: {diag['details']}")

        validation = fin.validate_techniques(price_data=price_series)
        print(f"\n  Technique validation:")
        for check, passed in validation['checks'].items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {check}")
    except ImportError:
        print("  pandas not available, skipping financial diagnostics")


def main():
    print("=" * 60)
    print("  SIGMA-C UNIVERSAL DIAGNOSTICS SYSTEM")
    print("  diagnose() | validate_techniques() | explain()")
    print("=" * 60)

    demo_gpu_diagnostics()
    demo_quantum_diagnostics()
    demo_financial_diagnostics()

    print("\n" + "=" * 60)
    print("  Diagnostics demo complete.")
    print("  Every adapter provides diagnose(), validate_techniques(),")
    print("  auto_search(), and explain() out of the box.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
