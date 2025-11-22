#!/usr/bin/env python3
"""
Sigma-C Seismic Demo
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates how to use the SeismicAdapter to detect critical states in 
earthquake catalogs.

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
    print("🌋 Starting Seismic Criticality Analysis...")
    
    # 1. Initialize Seismic Adapter
    seis = Universe.seismic()
    print("✓ Seismic Adapter initialized")
    
    # 2. Generate Mock Catalog (for demo purposes)
    # In production, use seis.fetch_catalog(region='california')
    print("✓ Generating synthetic earthquake catalog...")
    n_events = 500
    catalog = pd.DataFrame({
        'latitude': np.random.uniform(32, 42, n_events),
        'longitude': np.random.uniform(-125, -114, n_events),
        'mag': np.random.exponential(1.0, n_events) + 2.5, # Gutenberg-Richter like
        'time': pd.date_range('2023-01-01', periods=n_events, freq='12H')
    })
    
    # 3. Run Criticality Analysis
    print("✓ Computing stress proxy and susceptibility...")
    res = seis.analyze_criticality(catalog)
    
    sigma_c = res['sigma_c']
    kappa = res['kappa']
    
    print("\n📊 Seismic Status:")
    print(f"   Critical Interaction Range: {sigma_c:.2f} km")
    print(f"   Criticality Score (κ):      {kappa:.2f}")
    
    # 4. Visualize
    try:
        plt.figure(figsize=(10, 5))
        
        scales = res['sigma_values']
        obs = res['observable']
        
        plt.plot(scales, obs, 'k-^', label='Stress Correlation')
        plt.axvline(sigma_c, color='r', linestyle='--', label=f'Critical Range {sigma_c:.1f}km')
        
        plt.xlabel("Interaction Range (km)")
        plt.ylabel("Stress Correlation")
        plt.title("Seismic Criticality Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "seismic_results.png"
        plt.savefig(output_file)
        print(f"✓ Plot saved to {output_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
