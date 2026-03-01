#!/usr/bin/env python3
"""
Sigma-C GPU Demo: Cache Transition Detection
==============================================
Copyright (c) 2025 ForgottenForge.xyz

GPU performance degrades non-linearly when working sets exceed cache
boundaries. This creates measurable phase transitions at each level
of the memory hierarchy (L1, L2, L3/VRAM).

This demo uses Sigma-C to sweep workload intensity and detect the
critical point where performance drops sharply -- the cache thrashing
threshold. It then validates the trend with a Jonckheere-Terpstra test.

Runs in simulation mode if CuPy is not installed.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def main():
    print("=" * 60)
    print("  GPU CACHE TRANSITION DETECTION")
    print("  Finding the workload threshold for cache thrashing")
    print("=" * 60)

    gpu = Universe.gpu()

    # Use a small set of alpha levels for a quick demo
    alpha_levels = [0.0, 0.15, 0.3, 0.45, 0.6]
    print(f"\n  Testing {len(alpha_levels)} workload intensities...")
    print(f"  (alpha = memory pressure parameter)")

    result = gpu.auto_tune(
        alpha_levels=alpha_levels,
        epsilon_points=12
    )

    tuning = result['tuning_results']
    stats = result['statistics']

    # Display per-alpha results
    print(f"\n  {'Alpha':>7} {'sigma_c':>10} {'kappa':>8} {'Mean GFLOPS':>13}")
    print("  " + "-" * 42)
    for alpha in sorted(tuning.keys()):
        r = tuning[alpha]
        mean_gflops = np.mean(r['gflops'])
        print(f"  {alpha:7.2f} {r['sigma_c']:10.4f} {r['kappa']:8.2f} {mean_gflops:13.1f}")
    print("  " + "-" * 42)

    # Best configuration
    best_alpha = max(tuning.keys(), key=lambda k: tuning[k]['kappa'])
    print(f"\n  Sharpest transition at alpha = {best_alpha:.2f}")
    print(f"    sigma_c = {tuning[best_alpha]['sigma_c']:.4f}")
    print(f"    kappa   = {tuning[best_alpha]['kappa']:.2f}")

    # Statistical validation
    jt = stats['jonckheere_terpstra']
    print(f"\n  Jonckheere-Terpstra trend test:")
    print(f"    Statistic: {jt.get('statistic', jt.get('J', 'N/A'))}")
    print(f"    p-value:   {jt['p_value']:.4e}")
    if jt['p_value'] < 0.05:
        print(f"    Result:    Significant monotonic trend (p < 0.05)")
        print(f"               sigma_c increases with workload intensity")
    else:
        print(f"    Result:    No significant trend detected")

    # Roofline analysis
    roofline = gpu.analyze_roofline()
    print(f"\n  Roofline Model:")
    print(f"    Peak compute: {roofline['peak_gflops']:.0f} GFLOPS")
    print(f"    Peak bandwidth: {roofline['peak_bandwidth_gbs']:.0f} GB/s")
    print(f"    Ridge point: {roofline['ridge_point']:.1f} FLOP/Byte")
    print(f"    Current regime: {roofline['regime']}")

    # Thermal throttling prediction
    for temp in [60.0, 75.0, 82.0]:
        shift = gpu.predict_thermal_throttling(temp)
        print(f"    sigma_c shift at {temp:.0f}C: {shift:.3f}")

    print(f"\n  The critical point (sigma_c) tells you the exact workload")
    print(f"  intensity where your GPU transitions from efficient to")
    print(f"  cache-thrashing behavior. Stay below it for optimal perf.\n")


if __name__ == "__main__":
    main()
