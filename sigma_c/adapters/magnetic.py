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
        Returns mean absolute magnetization.
        """
        if len(data) < 1:
            return 0.0
        return float(np.mean(np.abs(data)))

    def analyze_critical_exponents(self, temperatures: np.ndarray, magnetization: np.ndarray,
                                    susceptibility: np.ndarray, specific_heat: np.ndarray) -> Dict[str, float]:
        """
        Extracts critical exponents beta, gamma, alpha near T_c.

        M ~ (T_c - T)^beta
        chi ~ |T - T_c|^-gamma
        C_v ~ |T - T_c|^-alpha
        """
        tc_idx = int(np.argmax(susceptibility))
        t_c = temperatures[tc_idx]

        # Fit beta (T < T_c)
        t_sub = temperatures[:tc_idx]
        m_sub = np.abs(magnetization[:tc_idx])
        if len(t_sub) >= 2 and np.any(m_sub > 0):
            valid = m_sub > 1e-12
            if np.sum(valid) >= 2:
                log_tau = np.log(np.abs(t_c - t_sub[valid]) + 1e-9)
                log_m = np.log(m_sub[valid])
                beta, _ = np.polyfit(log_tau, log_m, 1)
            else:
                beta = float('nan')
        else:
            beta = float('nan')

        # Fit gamma (exclude T_c)
        mask = np.arange(len(temperatures)) != tc_idx
        valid_chi = susceptibility[mask] > 1e-12
        if np.sum(valid_chi) >= 2:
            log_chi = np.log(susceptibility[mask][valid_chi])
            log_tau_all = np.log(np.abs(t_c - temperatures[mask][valid_chi]) + 1e-9)
            gamma, _ = np.polyfit(log_tau_all, log_chi, 1)
            gamma = -gamma
        else:
            gamma = float('nan')

        # Fit alpha from specific heat
        if specific_heat is not None and len(specific_heat) > 2:
            valid_cv = specific_heat[mask] > 1e-12
            if np.sum(valid_cv) >= 2:
                log_cv = np.log(specific_heat[mask][valid_cv])
                log_tau_cv = np.log(np.abs(t_c - temperatures[mask][valid_cv]) + 1e-9)
                alpha_fit, _ = np.polyfit(log_tau_cv, log_cv, 1)
                alpha = -alpha_fit
            else:
                alpha = 0.0
        else:
            alpha = 0.0

        return {
            'T_c': float(t_c),
            'beta': float(beta),
            'gamma': float(gamma),
            'alpha': float(alpha)
        }

    def analyze_finite_size_scaling(self, system_sizes: List[int], critical_temperatures: List[float]) -> Dict[str, float]:
        """
        Analyzes Finite Size Scaling (FSS).
        T_c(L) = T_c(inf) + A * L^(-1/nu)
        """
        sizes = np.array(system_sizes, dtype=float)
        tc_vals = np.array(critical_temperatures, dtype=float)

        if len(sizes) < 2:
            return {
                'system_sizes': system_sizes,
                'critical_temperatures': critical_temperatures
            }

        inv_L = 1.0 / sizes
        slope, intercept = np.polyfit(inv_L, tc_vals, 1)

        return {
            'system_sizes': system_sizes,
            'critical_temperatures': critical_temperatures,
            'T_c_extrapolated': float(intercept),
            'scaling_coefficient': float(slope)
        }
