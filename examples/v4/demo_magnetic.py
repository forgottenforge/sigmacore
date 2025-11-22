#!/usr/bin/env python3
"""
Sigma-C Magnetic Demo
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates how to use the MagneticAdapter to detect the Curie temperature
phase transition in a 2D Ising Model.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import sigma_c from local source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c import Universe

def main():
    print("🧲 Starting Magnetic Phase Transition Analysis...")
    
    # 1. Initialize Magnetic Adapter
    mag = Universe.magnetic()
    print("✓ Magnetic Adapter initialized")
    
    # 2. Run Simulation
    # We simulate a 20x20 lattice across a temperature range.
    # The theoretical critical temperature (Curie point) for 2D Ising is ~2.269.
    print("✓ Simulating 2D Ising Model (Metropolis-Hastings)...")
    
    res = mag.analyze_phase_transition(L=20, temp_range=np.linspace(1.5, 3.5, 21))
    
    sigma_c = res['sigma_c']
    kappa = res['kappa']
    
    print("\n📊 Magnetic Status:")
    print(f"   Detected Curie Temperature (Tc): {sigma_c:.3f}")
    print(f"   Theoretical Tc:                  2.269")
    print(f"   Criticality Score (κ):           {kappa:.2f}")
    
    # 3. Visualize
    try:
        plt.figure(figsize=(10, 5))
        
        temps = res['temperatures']
        mags = res['magnetization']
        
        plt.plot(temps, mags, 'b-o', label='Magnetization')
        plt.axvline(sigma_c, color='r', linestyle='--', label=f'Detected Tc {sigma_c:.3f}')
        plt.axvline(2.269, color='g', linestyle=':', label='Theoretical Tc 2.269')
        
        plt.xlabel("Temperature (T)")
        plt.ylabel("Magnetization (M)")
        plt.title("Magnetic Phase Transition (Ising Model)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "magnetic_results.png"
        plt.savefig(output_file)
        print(f"✓ Plot saved to {output_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
