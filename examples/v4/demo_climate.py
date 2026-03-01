#!/usr/bin/env python3
"""
Sigma-C Climate Demo: Mesoscale Boundary Detection
====================================================
Copyright (c) 2025 ForgottenForge.xyz

Atmospheric energy spectra follow different power laws at different scales:
  E(k) ~ k^(-3)    at synoptic scales (> 500 km)
  E(k) ~ k^(-5/3)  at mesoscale (< 500 km)

The boundary between these regimes is a critical scale where weather
dynamics transition from quasi-2D (synoptic) to 3D turbulence (mesoscale).

This demo generates a synthetic atmospheric spectrum with the known
spectral break, then uses Sigma-C to detect it.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sigma_c import Universe


def generate_atmospheric_spectrum(n_wavenumbers=200, break_scale_km=500.0):
    """Generate synthetic atmospheric energy spectrum with spectral break."""
    # Wavenumber range: from planetary (10,000 km) to mesoscale (10 km)
    wavelengths = np.logspace(np.log10(10), np.log10(10000), n_wavenumbers)[::-1]
    wavenumbers = 2 * np.pi / wavelengths  # rad/km

    k_break = 2 * np.pi / break_scale_km
    energy = np.zeros(n_wavenumbers)

    for i, k in enumerate(wavenumbers):
        if k < k_break:
            # Synoptic scale: E(k) ~ k^(-3)
            energy[i] = 1e6 * k ** (-3.0)
        else:
            # Mesoscale: E(k) ~ k^(-5/3), matched at break point
            E_break = 1e6 * k_break ** (-3.0)
            energy[i] = E_break * (k / k_break) ** (-5.0/3.0)

        # Add some noise
        energy[i] *= np.exp(np.random.normal(0, 0.05))

    return wavenumbers, energy, wavelengths, break_scale_km


def main():
    print("=" * 60)
    print("  ATMOSPHERIC MESOSCALE BOUNDARY DETECTION")
    print("  Spectral break between synoptic and mesoscale regimes")
    print("=" * 60)

    np.random.seed(42)
    clim = Universe.climate()

    wavenumbers, energy, wavelengths, true_break = generate_atmospheric_spectrum()

    # Use the ClimateAdapter to find the spectral break
    result = clim.analyze_mesoscale_boundary(energy, wavenumbers)

    detected_scale = result['critical_wavelength_km']
    sigma_c = result['sigma_c']
    slope_synoptic = result['spectral_slope_synoptic']
    slope_mesoscale = result['spectral_slope_mesoscale']
    error_pct = abs(detected_scale - true_break) / true_break * 100

    print(f"\n  True spectral break:       {true_break:.0f} km")
    print(f"  Detected break (Sigma-C):  {detected_scale:.0f} km")
    print(f"  Error:                     {error_pct:.1f}%")
    print(f"  sigma_c (break/Rossby):    {sigma_c:.3f}")
    print(f"  Synoptic slope:            {slope_synoptic:.2f}  (expected: -3.0)")
    print(f"  Mesoscale slope:           {slope_mesoscale:.2f}  (expected: -1.67)")

    # Spectral visualization
    print(f"\n  Energy spectrum (log-log):")
    print("  " + "-" * 50)
    step = max(1, len(wavenumbers) // 15)
    for i in range(0, len(wavenumbers), step):
        wl = wavelengths[i]
        log_e = np.log10(energy[i])
        bar_len = max(0, int((log_e + 2) * 5))
        marker = " <-- break" if abs(wl - detected_scale) < 80 else ""
        print(f"  {wl:7.0f} km | {'#' * min(bar_len, 40)}{marker}")
    print("  " + "-" * 50)

    # Vertical structure analysis
    print(f"\n--- Vertical Stability (Tropopause Detection) ---")
    n_profiles = 5
    pressure_levels = np.array([1000, 850, 700, 500, 300, 200, 100, 50])  # hPa
    temperature_profiles = np.zeros((n_profiles, len(pressure_levels)))

    for p in range(n_profiles):
        # Standard atmosphere-like profile with tropopause around 200 hPa
        z = -7.0 * np.log(pressure_levels / 1000.0)
        temperature_profiles[p] = 288 - 6.5 * z  # Tropospheric lapse rate
        # Tropopause: temperature stops decreasing
        tropo_idx = np.searchsorted(z, 11.0)  # ~11 km
        temperature_profiles[p, tropo_idx:] = temperature_profiles[p, tropo_idx] + \
            np.random.normal(0, 1, len(z) - tropo_idx)

    vert = clim.analyze_vertical_structure(pressure_levels, temperature_profiles)
    print(f"  Mean tropopause height: {vert['mean_tropopause_height']:.1f} km")
    print(f"  (Expected: ~11 km for standard atmosphere)")

    print(f"\n  The spectral break at ~{detected_scale:.0f} km marks the transition")
    print(f"  from quasi-2D synoptic dynamics to 3D mesoscale turbulence.\n")


if __name__ == "__main__":
    main()
