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
