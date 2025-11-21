#!/usr/bin/env python3
"""
Sigma-C Magnetic Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Magnetic Systems (Ising Model).

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
import numpy as np
from typing import Any, Dict, Optional, List

class MagneticAdapter(SigmaCAdapter):
    """
    Adapter for Magnetic Systems (Ising Model).
    Simulates a 2D Ising model to detect the Curie temperature phase transition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Compute magnetization per spin.
        """
        # data is the lattice configuration (spin +/- 1)
        return float(np.abs(np.mean(data)))

    def simulate_ising(self, L: int, T: float, steps: int = 1000) -> np.ndarray:
        """
        Metropolis-Hastings simulation of 2D Ising Model.
        """
        # Initialize random spins
        spins = np.random.choice([-1, 1], size=(L, L))
        
        for _ in range(steps):
            # Pick random site
            i, j = np.random.randint(0, L, 2)
            s = spins[i, j]
            
            # Calculate energy change (periodic boundary conditions)
            neighbors = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
            dE = 2 * s * neighbors
            
            # Metropolis criterion
            if dE < 0 or np.random.random() < np.exp(-dE / T):
                spins[i, j] *= -1
                
        return spins

    def analyze_phase_transition(self, L: int = 20, temp_range: Optional[List[float]] = None):
        """
        Analyze the magnetic phase transition (Curie point).
        """
        if temp_range is None:
            # Critical temp for 2D Ising is ~2.269
            temp_range = np.linspace(1.5, 3.5, 20)
            
        observables = []
        magnetizations = []
        
        for T in temp_range:
            # Run simulation
            lattice = self.simulate_ising(L, T, steps=L*L*100) # Sufficient equilibration
            mag = self.get_observable(lattice)
            observables.append(mag)
            magnetizations.append(lattice)
            
        # Compute Susceptibility
        analysis = self.compute_susceptibility(np.array(temp_range), np.array(observables))
        
        return {
            'sigma_c': analysis['sigma_c'], # Should be close to 2.27
            'kappa': analysis['kappa'],
            'temperatures': temp_range,
            'magnetization': observables
        }

    # ========== v1.1.0: Magnetic Diagnostics ==========
    
    def _domain_specific_diagnose(self, lattice_size: int = None, **kwargs) -> Dict[str, Any]:
        """Magnetic-specific diagnostics."""
        issues, recommendations, details = [], [], {}
        
        if lattice_size is None:
            lattice_size = self.config.get('lattice_size', 32)
        
        details['lattice_size'] = lattice_size
        if lattice_size < 16:
            issues.append(f"Small lattice: {lattice_size}x{lattice_size}")
            recommendations.append("Use larger lattice (>= 32) for finite-size scaling")
        elif lattice_size > 128:
            issues.append(f"Large lattice: {lattice_size}x{lattice_size} (slow)")
            recommendations.append("Consider smaller lattice for faster simulation")
        
        status = 'ok' if not issues else 'warning'
        return {'status': status, 'issues': issues, 'recommendations': recommendations, 'auto_fix': None, 'details': details}
    
    def _domain_specific_auto_search(self, param_ranges: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Auto-search optimal magnetic parameters."""
        temps = np.linspace(1.5, 3.5, 8) if param_ranges is None else np.linspace(*param_ranges['temperature'], 8)
        results = []
        
        for temp in temps:
            try:
                result = self.simulate_ising(temperature=temp)
                results.append({'temperature': float(temp), 'sigma_c': result['sigma_c'], 'kappa': result['kappa'], 'success': True})
            except Exception as e:
                results.append({'temperature': float(temp), 'sigma_c': 0, 'kappa': 0, 'success': False, 'error': str(e)})
        
        successful = [r for r in results if r.get('success', False)]
        if not successful:
            return {'best_params': {}, 'all_results': results, 'convergence_data': {}, 'recommendation': 'No successful runs'}
        
        best = max(successful, key=lambda x: x['kappa'])
        return {'best_params': {'temperature': best['temperature']}, 'all_results': results, 'convergence_data': {'n_successful': len(successful)}, 'recommendation': f"Use T={best['temperature']:.2f} (κ={best['kappa']:.2f})"}
    
    def _domain_specific_validate(self, data: Optional[Any] = None, **kwargs) -> Dict[str, bool]:
        """Validate magnetic techniques."""
        lattice_size = kwargs.get('lattice_size', self.config.get('lattice_size', 32))
        return {'lattice_size_valid': 16 <= lattice_size <= 256, 'numpy_available': True}
    
    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """Magnetic-specific explanation."""
        return f"# Magnetic Phase Transition Analysis\n\nσ_c: {result.get('sigma_c', 'N/A')}\nκ: {result.get('kappa', 'N/A')}\n\nHigher κ = sharper phase transition (near T_c)"
