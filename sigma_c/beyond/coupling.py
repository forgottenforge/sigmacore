"""
Sigma-C Cross-Domain Coupling
=============================
Copyright (c) 2025 ForgottenForge.xyz

Analyzes interactions between different critical systems to detect
cascade risks (e.g., Financial crash triggering Economic instability).
"""

import numpy as np
from typing import Dict, List, Tuple

class CouplingMatrix:
    """
    Manages the interaction matrix J_ij between N domains.
    System stability is determined by the largest eigenvalue of J.
    """
    
    def __init__(self, domains: List[str]):
        self.domains = domains
        self.n = len(domains)
        self.matrix = np.zeros((self.n, self.n))
        self.domain_map = {d: i for i, d in enumerate(domains)}
        
    def set_coupling(self, source: str, target: str, strength: float):
        """
        Sets the coupling strength J_ij (source -> target).
        Positive: Ferromagnetic (Sync)
        Negative: Antiferromagnetic (Anti-sync)
        """
        if source not in self.domain_map or target not in self.domain_map:
            raise ValueError("Unknown domain")
        
        i = self.domain_map[target] # Row index (Target)
        j = self.domain_map[source] # Col index (Source)
        self.matrix[i, j] = strength
        
    def analyze_stability(self) -> Dict[str, float]:
        """
        Analyzes the stability of the coupled system.
        If max(Re(eigenvalues)) > 0 (for continuous) or |lambda| > 1 (discrete),
        perturbations grow -> Cascade Risk.
        """
        eigenvals = np.linalg.eigvals(self.matrix)
        max_eig = np.max(np.abs(eigenvals))
        
        # Criticality of the meta-system
        # If max_eig approaches 1, the coupled system is critical
        meta_sigma_c = 1.0 / (max_eig + 1e-9)
        
        return {
            'max_eigenvalue': float(max_eig),
            'meta_sigma_c': float(meta_sigma_c),
            'stability': 'stable' if max_eig < 0.95 else 'critical' if max_eig < 1.05 else 'unstable',
            'cascade_risk': float(max(0, max_eig - 0.8) / 0.2) # 0 to 1 scale starting at 0.8
        }

    def simulate_cascade(self, initial_perturbation: Dict[str, float], steps: int = 10) -> List[Dict[str, float]]:
        """
        Simulates the propagation of a perturbation.
        x(t+1) = J * x(t)
        """
        state = np.zeros(self.n)
        for d, v in initial_perturbation.items():
            if d in self.domain_map:
                state[self.domain_map[d]] = v
                
        history = []
        for _ in range(steps):
            state = self.matrix @ state
            # Non-linearity (saturation)
            state = np.tanh(state)
            
            snapshot = {d: state[i] for d, i in self.domain_map.items()}
            history.append(snapshot)
            
        return history
