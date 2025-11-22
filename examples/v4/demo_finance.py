#!/usr/bin/env python3
"""
Sigma-C Financial Demo
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates how to use the FinancialAdapter to detect volatility regimes 
and critical market shifts using Sigma-C.

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
    print("📈 Starting Financial Regime Detection...")
    
    # 1. Initialize Financial Adapter
    fin = Universe.finance()
    print("✓ Financial Adapter initialized")
    
    # 2. Fetch Data (or use mock if offline)
    symbol = "SPY"
    print(f"✓ Analyzing {symbol} market data...")
    
    # The adapter handles data fetching. If yfinance fails (no internet), it mocks data.
    # We can also pass our own DataFrame:
    # data = pd.read_csv('my_data.csv')
    # res = fin.detect_regime(data=data)
    
    res = fin.detect_regime(symbol=symbol)
    
    # 3. Analyze Results
    sigma_c = res['sigma_c']
    kappa = res['kappa']
    regime = res['regime']
    
    print("\n📊 Market Status:")
    print(f"   Detected Regime:      {regime}")
    print(f"   Critical Scale (σ_c): {sigma_c:.2f} days")
    print(f"   Stability Score (κ):  {kappa:.2f}")
    
    if kappa > 2.0:
        print("   => Strong evidence of a critical phase transition (market shift).")
    else:
        print("   => Market is relatively stable or in a mixed state.")

    # 4. Visualize
    try:
        plt.figure(figsize=(10, 5))
        
        # Plot Volatility Clustering Observable
        sigma_vals = res['sigma_values']
        obs_vals = res['observable']
        
        plt.plot(sigma_vals, obs_vals, 'g-o', label='Volatility Clustering')
        plt.axvline(sigma_c, color='r', linestyle='--', label=f'Critical Scale {sigma_c:.1f}d')
        
        plt.xscale('log')
        plt.xlabel("Time Scale (Days)")
        plt.ylabel("Clustering Strength")
        plt.title(f"Financial Criticality Analysis: {symbol}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "finance_results.png"
        plt.savefig(output_file)
        print(f"✓ Plot saved to {output_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
