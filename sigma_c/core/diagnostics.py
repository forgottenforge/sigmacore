#!/usr/bin/env python3
"""
Sigma-C Diagnostics Engine
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Helper utilities for the Universal Diagnostics System (v1.1.0).

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats

class DiagnosticsEngine:
    """
    Helper class for diagnostics across all domains.
    """
    
    @staticmethod
    def check_data_quality(data: np.ndarray, min_length: int = 10) -> Dict[str, Any]:
        """
        Basic data quality checks.
        
        Returns:
            {
                'sufficient_length': bool,
                'has_nans': bool,
                'has_infs': bool,
                'is_constant': bool,
                'length': int
            }
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        return {
            'sufficient_length': len(data) >= min_length,
            'has_nans': np.any(np.isnan(data)),
            'has_infs': np.any(np.isinf(data)),
            'is_constant': np.std(data) < 1e-10,
            'length': len(data)
        }
    
    @staticmethod
    def detect_outliers(data: np.ndarray, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using z-score method.
        
        Returns:
            {
                'has_outliers': bool,
                'outlier_indices': List[int],
                'outlier_fraction': float
            }
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > threshold
        
        return {
            'has_outliers': np.any(outlier_mask),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'outlier_fraction': np.mean(outlier_mask)
        }
    
    @staticmethod
    def check_stationarity(data: np.ndarray, window_size: int = 50) -> Dict[str, Any]:
        """
        Simple stationarity check using rolling statistics.
        
        Returns:
            {
                'is_stationary': bool,
                'mean_stable': bool,
                'variance_stable': bool
            }
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) < window_size * 2:
            return {
                'is_stationary': False,
                'mean_stable': False,
                'variance_stable': False,
                'note': 'Insufficient data for stationarity test'
            }
        
        # Split into windows
        n_windows = len(data) // window_size
        window_means = []
        window_vars = []
        
        for i in range(n_windows):
            window = data[i*window_size:(i+1)*window_size]
            window_means.append(np.mean(window))
            window_vars.append(np.var(window))
        
        # Check stability
        mean_cv = np.std(window_means) / (np.mean(window_means) + 1e-10)
        var_cv = np.std(window_vars) / (np.mean(window_vars) + 1e-10)
        
        mean_stable = mean_cv < 0.1
        variance_stable = var_cv < 0.2
        
        return {
            'is_stationary': mean_stable and variance_stable,
            'mean_stable': mean_stable,
            'variance_stable': variance_stable,
            'mean_cv': mean_cv,
            'variance_cv': var_cv
        }
    
    @staticmethod
    def optimize_parameter_grid(param_ranges: Dict[str, Tuple[float, float]], 
                                 n_points: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate parameter grid for systematic search.
        
        Args:
            param_ranges: Dict of {param_name: (min, max)}
            n_points: Number of points per parameter
        
        Returns:
            Dict of {param_name: np.ndarray of values}
        """
        grid = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            grid[param_name] = np.linspace(min_val, max_val, n_points)
        return grid
    
    @staticmethod
    def rank_results(results: List[Dict[str, Any]], 
                     metric: str = 'kappa',
                     ascending: bool = False) -> List[Dict[str, Any]]:
        """
        Rank results by a specific metric.
        
        Args:
            results: List of result dictionaries
            metric: Key to rank by
            ascending: If True, lower is better
        
        Returns:
            Sorted list of results
        """
        return sorted(results, 
                     key=lambda x: x.get(metric, -np.inf if not ascending else np.inf),
                     reverse=not ascending)
    
    @staticmethod
    def visualize_parameter_sweep(results: List[Dict[str, Any]], 
                                   param_name: str,
                                   metric: str = 'sigma_c') -> plt.Figure:
        """
        Create visualization of parameter sweep results.
        
        Args:
            results: List of result dictionaries
            param_name: Parameter that was varied
            metric: Metric to plot
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = [r[param_name] for r in results]
        values = [r[metric] for r in results]
        
        ax.plot(params, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs {param_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = np.argmax(values)
        ax.plot(params[best_idx], values[best_idx], 'r*', markersize=20, 
                label=f'Best: {param_name}={params[best_idx]:.3f}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_recommendation(issues: List[str], 
                                 domain: str = 'generic') -> str:
        """
        Generate actionable recommendations based on issues.
        
        Args:
            issues: List of detected issues
            domain: Domain name for context
        
        Returns:
            Markdown-formatted recommendation string
        """
        if not issues:
            return "✅ No issues detected. Analysis looks good!"
        
        rec = f"## Recommendations for {domain.capitalize()} Domain\n\n"
        
        for i, issue in enumerate(issues, 1):
            rec += f"{i}. **Issue:** {issue}\n"
            
            # Generic recommendations based on common patterns
            if 'insufficient' in issue.lower() or 'length' in issue.lower():
                rec += "   - **Action:** Collect more data points\n"
            elif 'nan' in issue.lower() or 'missing' in issue.lower():
                rec += "   - **Action:** Clean data or impute missing values\n"
            elif 'outlier' in issue.lower():
                rec += "   - **Action:** Investigate outliers or apply robust methods\n"
            elif 'stationary' in issue.lower():
                rec += "   - **Action:** Apply differencing or detrending\n"
            else:
                rec += "   - **Action:** Review domain-specific requirements\n"
            rec += "\n"
        
        return rec
