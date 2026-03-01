#!/usr/bin/env python3
"""
Sigma-C Seismic Demo: Gutenberg-Richter Law Analysis
======================================================
Copyright (c) 2025 ForgottenForge.xyz

The Gutenberg-Richter law states that earthquake magnitudes follow:
  log10(N) = a - b * M

where b ~ 1.0 globally. Deviations from b = 1.0 indicate stress
accumulation (b < 1.0, more large quakes) or stress release (b > 1.0,
more small quakes). Sigma-C uses this as a criticality indicator.

This demo generates synthetic earthquake catalogs with different b-values
and demonstrates how Sigma-C detects the seismic regime.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def generate_earthquake_catalog(n_events=1000, b_value=1.0, m_min=2.0, seed=42):
    """Generate synthetic magnitudes following Gutenberg-Richter distribution."""
    rng = np.random.default_rng(seed)
    # Inverse CDF: M = m_min - ln(U) / (b * ln(10))
    u = rng.uniform(0, 1, n_events)
    magnitudes = m_min - np.log(u) / (b_value * np.log(10))
    return magnitudes


def main():
    print("=" * 60)
    print("  SEISMIC CRITICALITY: GUTENBERG-RICHTER ANALYSIS")
    print("  Detecting stress regimes from magnitude distributions")
    print("=" * 60)

    seis = Universe.seismic()

    scenarios = [
        ("Normal (b=1.0)", 1.0, "Typical global average"),
        ("Stressed (b=0.7)", 0.7, "Stress accumulation -- more large events"),
        ("Relaxed (b=1.3)", 1.3, "Post-mainshock -- more small events"),
    ]

    print(f"\n  {'Scenario':<22} {'b_true':>7} {'b_detected':>11} {'1/b (criticality)':>18}")
    print("  " + "-" * 60)

    for name, b_true, description in scenarios:
        magnitudes = generate_earthquake_catalog(n_events=2000, b_value=b_true, seed=int(b_true * 100))
        result = seis.analyze_gutenberg_richter(magnitudes)
        b_detected = result['b_value']
        crit = result['criticality']

        print(f"  {name:<22} {b_true:>7.2f} {b_detected:>11.3f} {crit:>18.3f}")

    # Detailed analysis of the stressed scenario
    print(f"\n--- Detailed Analysis: Stressed Region ---")
    stressed_mags = generate_earthquake_catalog(n_events=2000, b_value=0.7, seed=70)
    result = seis.analyze_gutenberg_richter(stressed_mags)

    print(f"  b-value:      {result['b_value']:.3f}")
    print(f"  M_min:        {result['m_min']:.2f}")
    print(f"  Criticality:  {result['criticality']:.3f}")

    # Statistical significance
    p_value = seis.compute_significance(result['b_value'], stressed_mags, n_surrogates=500)
    print(f"  p-value:      {p_value:.4f}")

    # Magnitude distribution
    print(f"\n  Magnitude distribution (stressed region):")
    print("  " + "-" * 45)
    bins = np.arange(2.0, 7.0, 0.5)
    counts, _ = np.histogram(stressed_mags, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * 35)
        print(f"  M {bins[i]:4.1f}-{bins[i+1]:4.1f} | {'#' * bar_len} ({counts[i]})")
    print("  " + "-" * 45)

    # Omori law on synthetic aftershock times
    print(f"\n--- Omori Aftershock Decay ---")
    rng = np.random.default_rng(99)
    # Simulate aftershock times with p ~ 1.1 decay
    aftershock_times = rng.pareto(1.1, 500) + 0.1
    omori = seis.analyze_omori_scaling(aftershock_times)
    print(f"  p-value (decay exponent): {omori['p_value']:.3f}")
    print(f"  (p ~ 1.0 is typical Omori law behavior)")

    print(f"\n  Key insight: b < 1.0 indicates stress accumulation and")
    print(f"  increased probability of large earthquakes. Sigma-C quantifies")
    print(f"  this departure from equilibrium as a criticality measure.\n")


if __name__ == "__main__":
    main()
