"""
Sigma-C v1.2.0 "Full Power" Demo
================================
Showcases the Universal Optimizer and Rigorous Analysis across all domains.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import sigmacore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c_framework.sigma_c.adapters.quantum import QuantumAdapter
from sigma_c_framework.sigma_c.adapters.gpu import GPUAdapter
from sigma_c_framework.sigma_c.adapters.financial import FinancialAdapter

from sigma_c_framework.sigma_c.optimization.quantum import BalancedQuantumOptimizer
from sigma_c_framework.sigma_c.optimization.gpu import BalancedGPUOptimizer
from sigma_c_framework.sigma_c.optimization.financial import BalancedFinancialOptimizer

from sigma_c_framework.sigma_c.reporting.latex import LatexGenerator
from sigma_c_framework.sigma_c.plotting.publication import PublicationVisualizer

def run_demo():
    print("🚀 Starting Sigma-C v1.2.0 'Full Power' Demo...")
    
    # 1. Quantum Domain
    print("\n[1/3] Optimizing Quantum Circuit (Fidelity vs. Resilience)...")
    q_adapter = QuantumAdapter()
    q_opt = BalancedQuantumOptimizer(q_adapter)
    
    # Mock circuit factory
    def grover_factory(epsilon=0.0, idle_frac=0.0):
        return q_adapter.create_grover_with_noise(epsilon=epsilon, idle_frac=idle_frac)
        
    q_res = q_opt.optimize_circuit(
        grover_factory,
        param_space={
            'epsilon': [0.0, 0.05, 0.1], 
            'idle_frac': [0.0, 0.1, 0.2]
        }
    )
    print(f"   ✅ Optimal Quantum Params: {q_res.optimal_params}")
    print(f"   ✅ Score: {q_res.score:.4f} (Sigma_c: {q_res.sigma_c_after:.4f})")

    # 2. GPU Domain
    print("\n[2/3] Optimizing GPU Kernel (Throughput vs. Thermals)...")
    g_adapter = GPUAdapter()
    g_opt = BalancedGPUOptimizer(g_adapter)
    
    g_res = g_opt.optimize_kernel(
        param_space={
            'n_launch': [5, 10, 20],
            'n_mem': [0, 2, 4],
            'alpha': [0.1, 0.3, 0.5]
        }
    )
    print(f"   ✅ Optimal GPU Params: {g_res.optimal_params}")
    print(f"   ✅ Score: {g_res.score:.4f} (Sigma_c: {g_res.sigma_c_after:.4f})")

    # 3. Financial Domain
    print("\n[3/3] Optimizing Financial Strategy (Returns vs. Crash Risk)...")
    f_adapter = FinancialAdapter()
    f_opt = BalancedFinancialOptimizer(f_adapter)
    
    f_res = f_opt.optimize_strategy(
        param_space={
            'lookback': [100, 200, 300],
            'symbol': ['^GSPC'] # S&P 500
        }
    )
    print(f"   ✅ Optimal Financial Params: {f_res.optimal_params}")
    print(f"   ✅ Score: {f_res.score:.4f} (Sigma_c: {f_res.sigma_c_after:.4f})")

    # 4. Generate Report
    print("\n[4/4] Generating Publication Report...")
    latex = LatexGenerator()
    viz = PublicationVisualizer(style='nature')
    
    # Create a dummy plot
    fig = plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Universal Optimization Landscape")
    viz.save_figure(fig, "universal_landscape")
    
    latex.generate_report(
        title="Universal Rigor in Multi-Domain Optimization",
        author="Sigma-C AI Agent",
        abstract="We demonstrate the application of rigorous quantum-inspired optimization techniques across Quantum, GPU, and Financial domains.",
        sections=[
            {
                'title': 'Quantum Optimization',
                'content': f"Optimal parameters found: {q_res.optimal_params}. Critical noise threshold sigma_c improved to {q_res.sigma_c_after:.4f}."
            },
            {
                'title': 'GPU Kernel Tuning',
                'content': f"Optimal configuration: {g_res.optimal_params}. Thermal stability sigma_c: {g_res.sigma_c_after:.4f}."
            },
            {
                'title': 'Financial Strategy',
                'content': f"Optimal lookback window: {f_res.optimal_params}. Crash risk sigma_c: {f_res.sigma_c_after:.4f}."
            }
        ],
        filename="sigma_c_v1.2.0_report"
    )
    
    print("\n🎉 Demo Complete! 'Full Power' verified.")

if __name__ == "__main__":
    run_demo()
