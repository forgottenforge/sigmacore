#!/usr/bin/env python3
"""
Sigma-C Quantum Demo
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates how to use the QuantumAdapter to analyze critical susceptibility 
in a noisy quantum circuit (Grover's Algorithm).

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os

# Ensure we can import sigma_c from local source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c import Universe
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("🚀 Starting Quantum Criticality Analysis...")
    
    # 1. Initialize the Quantum Adapter
    # 'simulator' uses local Braket simulator. Use 'rigetti' for real hardware (requires AWS creds).
    qpu = Universe.quantum(device='simulator')
    print(f"✓ Quantum Adapter initialized: {qpu}")
    print(f"Attributes: {dir(qpu)}")

    # 2. Define the experiment parameters
    # We sweep 'epsilon' (noise level) to find the critical point where the algorithm breaks down.
    epsilon_range = np.linspace(0.0, 0.25, 20)
    
    print(f"✓ Running optimization loop over {len(epsilon_range)} noise levels...")
    
    # 3. Run the optimization
    # This builds circuits, injects noise, runs them, and computes susceptibility.
    results = qpu.run_optimization(
        circuit_type='grover', 
        epsilon_values=epsilon_range,
        shots=1000  # Number of shots per point
    )
    
    # 4. Analyze Results
    sigma_c = results['sigma_c']
    kappa = results['kappa']
    
    print("\n📊 Analysis Complete:")
    print(f"   Critical Noise Level (σ_c): {sigma_c:.4f}")
    print(f"   Peak Clarity (κ):           {kappa:.2f}")
    
    # 5. Visualize
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot Observable (Success Probability)
        plt.subplot(2, 1, 1)
        plt.plot(results['epsilon'], results['observable'], 'b-o', label='Success Prob')
        plt.axvline(sigma_c, color='r', linestyle='--', label=f'Critical Point {sigma_c:.3f}')
        plt.title("Quantum Phase Transition in Grover's Algorithm")
        plt.ylabel("Success Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Susceptibility (Rate of Change)
        plt.subplot(2, 1, 2)
        # Re-compute susceptibility for plotting (internal engine does this, but we want to show it)
        # We can access the engine directly or just plot the derivative of the observable
        grad = np.abs(np.gradient(results['observable'], results['epsilon']))
        plt.plot(results['epsilon'], grad, 'r-s', label='Susceptibility |χ|')
        plt.axvline(sigma_c, color='r', linestyle='--')
        plt.xlabel("Noise Level (ε)")
        plt.ylabel("|∂O/∂ε|")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "quantum_results.png"
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"✓ Plot saved to {output_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    main()
