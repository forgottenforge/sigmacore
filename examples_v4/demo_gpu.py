#!/usr/bin/env python3
"""
Sigma-C GPU Demo
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates how to use the GPUAdapter to auto-tune kernel parameters 
by finding the critical performance threshold.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
import matplotlib.pyplot as plt

# Ensure we can import sigma_c from local source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c import Universe

def main():
    print("🎮 Starting GPU Kernel Auto-Tuning...")
    
    # 1. Initialize GPU Adapter
    # If cupy is not installed, it will run in simulation mode.
    gpu = Universe.gpu()
    print("✓ GPU Adapter initialized")
    
    # 2. Run Auto-Tuning
    # We sweep a parameter 'alpha' (e.g., memory access pattern or thread block size proxy)
    # to find the 'critical point' where performance degrades non-linearly (cache thrashing).
    print("✓ Running benchmark sweep...")
    
    res = gpu.auto_tune(
        alpha_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        epsilon_points=10
    )
    
    # 3. Analyze Results
    # The auto_tune method returns a dict with 'tuning_results' (per alpha) and 'statistics'
    tuning_results = res['tuning_results']
    stats = res['statistics']
    
    # Find best alpha (lowest sigma_c usually implies most stable, but here we want the one that pushes criticality furthest)
    # Actually, in this context, we want to see which alpha has the highest critical point (most robust)
    best_alpha = max(tuning_results.keys(), key=lambda k: tuning_results[k]['sigma_c'])
    
    print("\n📊 Tuning Results:")
    print(f"   Optimal Alpha: {best_alpha:.2f}")
    print(f"   P-Value (Jonckheere-Terpstra): {stats['jonckheere_terpstra']['p_value']:.4e}")
    
    if stats['jonckheere_terpstra']['p_value'] < 0.05:
        print("   => Statistically significant performance trend detected.")
    
    # 4. Visualize
    try:
        plt.figure(figsize=(10, 5))
        
        # Plot Performance vs Alpha
        alphas = sorted(tuning_results.keys())
        # We plot the critical point (sigma_c) for each alpha
        critical_points = [tuning_results[a]['sigma_c'] for a in alphas]
        
        plt.plot(alphas, critical_points, 'm-D', label='Critical Point (Robustness)')
        plt.axvline(best_alpha, color='g', linestyle='--', label=f'Optimal {best_alpha:.2f}')
        
        plt.xlabel("Tuning Parameter (Alpha)")
        plt.ylabel("Critical Noise Level (σ_c)")
        plt.title("GPU Kernel Criticality Tuning")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "gpu_results.png"
        plt.savefig(output_file)
        print(f"✓ Plot saved to {output_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
