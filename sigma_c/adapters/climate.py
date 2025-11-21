#!/usr/bin/env python3
"""
Sigma-C Climate Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Climate Data and Spatial Scaling Analysis.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import warnings

class ClimateAdapter(SigmaCAdapter):
    """
    Adapter for ERA5 Climate Data.
    Ported from erasi3.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache_dir = Path(self.config.get('cache_dir', 'era5_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_observable(self, data: pd.DataFrame, **kwargs) -> float:
        """
        Compute spatial variance of temperature gradients.
        """
        sigma_km = kwargs.get('sigma_km', 100.0)
        
        # Sample for speed if needed
        if len(data) > 20000:
            data = data.sample(n=20000, random_state=42)
            
        # Create grid
        lat_min, lat_max = data['lat'].min(), data['lat'].max()
        lon_min, lon_max = data['lon'].min(), data['lon'].max()
        
        n_cells = 30
        lat_edges = np.linspace(lat_min, lat_max, n_cells + 1)
        lon_edges = np.linspace(lon_min, lon_max, n_cells + 1)
        
        lat_idx = np.digitize(data['lat'], lat_edges) - 1
        lon_idx = np.digitize(data['lon'], lon_edges) - 1
        
        grid = np.full((n_cells, n_cells), np.nan)
        for i in range(n_cells):
            for j in range(n_cells):
                mask = (lat_idx == i) & (lon_idx == j)
                if mask.sum() > 0:
                    grid[i, j] = data.loc[mask, 'value'].mean()
                    
        # Interpolate NaN
        if np.any(np.isnan(grid)):
            mask = np.isnan(grid)
            grid[mask] = np.nanmean(grid)
            
        # Smooth
        km_per_cell = (lat_max - lat_min) * 111.0 / n_cells
        sigma_cells = sigma_km / km_per_cell
        sigma_cells = max(0.3, sigma_cells)
        
        grid_smooth = gaussian_filter(grid, sigma=sigma_cells, mode='reflect')
        
        # Gradients
        grad_y, grad_x = np.gradient(grid_smooth)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        variance = np.nanvar(gradient_magnitude)
        return variance if not np.isnan(variance) else 0.0

    def analyze_spatial_scaling(self, data: pd.DataFrame, sigma_range=None):
        """
        Run spatial scaling analysis (H1).
        """
        if sigma_range is None:
            sigma_range = np.logspace(np.log10(20), np.log10(5000), 20)
            
        observables = []
        for sigma in sigma_range:
            obs = self.get_observable(data, sigma_km=sigma)
            observables.append(obs)
            
        analysis = self.compute_susceptibility(sigma_range, np.array(observables))
        
        return {
            'sigma_c': analysis['sigma_c'],
            'kappa': analysis['kappa'],
            'sigma_range': sigma_range,
            'observable': observables
        }

    # ========== v1.1.0: Climate Diagnostics ==========
    
    def _domain_specific_diagnose(self, data: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Climate-specific diagnostics."""
        issues, recommendations, details = [], [], {}
        
        if data is None:
            return {'status': 'error', 'issues': ['No data'], 'recommendations': ['Provide climate data'], 'auto_fix': None, 'details': {}}
        
        details['data_shape'] = data.shape
        if data.ndim != 2:
            issues.append(f"Expected 2D grid, got {data.ndim}D")
            recommendations.append("Provide spatial grid data")
        elif min(data.shape) < 10:
            issues.append(f"Grid too small: {data.shape}")
            recommendations.append("Use larger spatial grid")
        
        status = 'ok' if not issues else 'warning'
        return {'status': status, 'issues': issues, 'recommendations': recommendations, 'auto_fix': None, 'details': details}
    
    def _domain_specific_auto_search(self, param_ranges: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Auto-search optimal climate parameters."""
        grid_sizes = [32, 64, 128] if param_ranges is None else list(range(*param_ranges['grid_size']))
        results = []
        
        for size in grid_sizes:
            try:
                result = self.analyze_climate_criticality(grid_size=size)
                results.append({'grid_size': size, 'sigma_c': result['sigma_c'], 'kappa': result['kappa'], 'success': True})
            except Exception as e:
                results.append({'grid_size': size, 'sigma_c': 0, 'kappa': 0, 'success': False, 'error': str(e)})
        
        successful = [r for r in results if r.get('success', False)]
        if not successful:
            return {'best_params': {}, 'all_results': results, 'convergence_data': {}, 'recommendation': 'No successful runs'}
        
        best = max(successful, key=lambda x: x['kappa'])
        return {'best_params': {'grid_size': best['grid_size']}, 'all_results': results, 'convergence_data': {'n_successful': len(successful)}, 'recommendation': f"Use grid_size={best['grid_size']} (κ={best['kappa']:.2f})"}
    
    def _domain_specific_validate(self, data: Optional[np.ndarray] = None, **kwargs) -> Dict[str, bool]:
        """Validate climate techniques."""
        if data is None:
            return {'data_provided': False, 'valid_grid': False}
        return {'data_provided': True, 'valid_grid': data.ndim == 2 and min(data.shape) >= 10}
    
    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """Climate-specific explanation."""
        return f"# Climate Criticality Analysis\n\nσ_c: {result.get('sigma_c', 'N/A')}\nκ: {result.get('kappa', 'N/A')}\n\nHigher κ = stronger spatial correlations"
