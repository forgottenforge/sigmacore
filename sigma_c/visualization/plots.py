"""
Sigma-C Visualization Module
============================
Standardized plotting tools for optimization results.

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from ..optimization.universal import OptimizationResult

def plot_convergence(result: OptimizationResult, 
                    metric: str = 'score', 
                    title: str = 'Optimization Convergence',
                    save_path: Optional[str] = None):
    """
    Plot the convergence of a metric over optimization steps.
    """
    history = result.history
    steps = range(len(history))
    values = [step.get(metric, 0) for step in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_landscape(optimizer: Any, 
                  param_space: Dict[str, List[Any]], 
                  system: Any,
                  resolution: int = 20,
                  save_path: Optional[str] = None):
    """
    Plot 2D landscape of the objective function.
    Only works for 2 numeric parameters.
    """
    keys = list(param_space.keys())
    if len(keys) != 2:
        print("Warning: Landscape plot requires exactly 2 parameters.")
        return
        
    p1_name, p2_name = keys
    p1_vals = np.linspace(min(param_space[p1_name]), max(param_space[p1_name]), resolution)
    p2_vals = np.linspace(min(param_space[p2_name]), max(param_space[p2_name]), resolution)
    
    X, Y = np.meshgrid(p1_vals, p2_vals)
    Z = np.zeros_like(X)
    
    print(f"Generating landscape ({resolution}x{resolution})...")
    
    for i in range(resolution):
        for j in range(resolution):
            params = {p1_name: X[i, j], p2_name: Y[i, j]}
            # Use optimizer's internal evaluation logic
            # Note: This might be slow as it re-evaluates everything
            try:
                perf = optimizer._evaluate_performance(system, params)
                stab = optimizer._evaluate_stability(system, params)
                Z[i, j] = optimizer.calculate_score(perf, stab)
            except:
                Z[i, j] = np.nan
                
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Composite Score')
    plt.title(f'Optimization Landscape: {p1_name} vs {p2_name}', fontsize=14)
    plt.xlabel(p1_name, fontsize=12)
    plt.ylabel(p2_name, fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pareto_frontier(result: OptimizationResult, save_path: Optional[str] = None):
    """
    Plot Performance vs Stability to visualize the trade-off.
    """
    history = result.history
    perf = [step['performance'] for step in history]
    stab = [step['stability'] for step in history]
    scores = [step['score'] for step in history]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(perf, stab, c=scores, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Composite Score')
    
    # Highlight best point
    best_idx = np.argmax(scores)
    plt.scatter(perf[best_idx], stab[best_idx], color='red', s=150, marker='*', label='Optimal')
    
    plt.title('Performance vs Stability Trade-off', fontsize=14)
    plt.xlabel('Performance', fontsize=12)
    plt.ylabel('Stability (Sigma_c)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
