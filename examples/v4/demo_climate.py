#!/usr/bin/env python3
"""
Sigma-C Climate Demo
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates how to use the ClimateAdapter to analyze spatial scaling properties
of temperature fields (ERA5 data).

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import sigma_c from local source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c import Universe

def main():
    print("🌍 Starting Climate Spatial Analysis...")
    
    # 1. Initialize Climate Adapter
    clim = Universe.climate()
    print("✓ Climate Adapter initialized")
    
    # 2. Generate Mock Data (for demo purposes)
    # In production, you would load real ERA5 parquet files.
    print("✓ Generating synthetic temperature field...")
    n_points = 5000
    data = pd.DataFrame({
        'lat': np.random.uniform(35, 70, n_points),
        'lon': np.random.uniform(-10, 40, n_points),
        # Create a synthetic field with a specific correlation length
        'value': np.sin(np.random.uniform(35, 70, n_points)/5) * 10 + 15 + np.random.normal(0, 2, n_points)
    })
    
    # 3. Run Spatial Scaling Analysis
    print("✓ Computing gradient variance across scales...")
    res = clim.analyze_spatial_scaling(data)
    
    sigma_c = res['sigma_c']
    kappa = res['kappa']
    
    print("\n📊 Climate Status:")
    print(f"   Critical Spatial Scale: {sigma_c:.1f} km")
    print(f"   Organization Strength (κ): {kappa:.2f}")
    
    # 4. Visualize
    try:
        plt.figure(figsize=(10, 5))
        
        scales = res['sigma_range']
        obs = res['observable']
        
        plt.plot(scales, obs, 'c-o', label='Gradient Variance')
        plt.axvline(sigma_c, color='r', linestyle='--', label=f'Critical Scale {sigma_c:.0f}km')
        
        plt.xscale('log')
        plt.xlabel("Spatial Scale (km)")
        plt.ylabel("Gradient Variance")
        plt.title("Climate Spatial Scaling Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "climate_results.png"
        plt.savefig(output_file)
        print(f"✓ Plot saved to {output_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
