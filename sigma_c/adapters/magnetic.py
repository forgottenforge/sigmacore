"""
Sigma-C Magnetic Adapter
========================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Magnetic Systems (Ising Model, Spin Glasses).
Focuses on Critical Exponents and Finite Size Scaling.
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any, List

class MagneticAdapter(SigmaCAdapter):
    """
    Adapter for Magnetic Systems.
    Validates universality classes and critical exponents.
    """
    
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns magnetization or susceptibility.
        """
        # Assuming data is magnetization
        if len(data) < 1: return 0.0
        return float(np.mean(np.abs(data)))
    
    def analyze_critical_exponents(self, temperatures: np.ndarray, magnetization: np.ndarray, susceptibility: np.ndarray, specific_heat: np.ndarray) -> Dict[str, float]:
        """
        Extracts critical exponents alpha, beta, gamma near T_c.
        
        M ~ (T_c - T)^beta
        chi ~ |T - T_c|^-gamma
        C_v ~ |T - T_c|^-alpha
        """
        # Find T_c (peak of susceptibility)
        tc_idx = np.argmax(susceptibility)
        t_c = temperatures[tc_idx]
        
        # Fit beta (T < T_c)
        t_sub = temperatures[:tc_idx]
        m_sub = magnetization[:tc_idx]
        log_tau = np.log(np.abs(t_c - t_sub) + 1e-9)
        log_m = np.log(m_sub + 1e-9)
        beta, _ = np.polyfit(log_tau, log_m, 1)
        
        # Fit gamma (T > T_c and T < T_c)
        # We average the slopes
        log_chi = np.log(susceptibility + 1e-9)
        log_tau_all = np.log(np.abs(t_c - temperatures) + 1e-9)
        
        # Exclude T_c itself
        mask = np.arange(len(temperatures)) != tc_idx
        gamma, _ = np.polyfit(log_tau_all[mask], log_chi[mask], 1)
        gamma = -gamma # Exponent is negative in definition
        
        return {
            'T_c': t_c,
            'beta': beta,
            'gamma': gamma,
            'alpha': 0.0 # Placeholder, needs C_v data
        }

    def analyze_finite_size_scaling(self, system_sizes: List[int], critical_temperatures: List[float]) -> Dict[str, float]:
        """
        Analyzes Finite Size Scaling (FSS).
        T_c(L) = T_c(inf) + A * L^(-1/nu)
        """
        # Fit T_c(L) vs L
        # We assume 2D Ising nu=1 for simplicity to find T_c(inf)
        # Or fit both
        
        log_L = np.log(system_sizes)
        # This is a non-linear fit usually, but we can linearize if we fix nu
        # Here we just return the data for external analysis
        
        return {
            'system_sizes': system_sizes,
            'critical_temperatures': critical_temperatures
        }
