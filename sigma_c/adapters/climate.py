"""
Sigma-C Climate Adapter
=======================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Atmospheric and Climate Criticality.
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any, List

class ClimateAdapter(SigmaCAdapter):
    """
    Adapter for Climate Systems.
    Focuses on Mesoscale/Synoptic boundaries and extreme event precursors.
    """
    
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns the spectral slope or relevant climate observable.
        """
        # Assuming data is an energy spectrum
        if len(data) < 2: return 0.0
        return float(np.mean(data))
    
    def analyze_mesoscale_boundary(self, energy_spectrum: np.ndarray, wavenumbers: np.ndarray) -> Dict[str, float]:
        """
        Analyzes the transition from Mesoscale to Synoptic scale.
        Paper: sigma_c corresponds to ~500km scale.
        
        Spectral Energy Density E(k) ~ k^-3 (Synoptic) vs k^-5/3 (Mesoscale).
        The break point is the critical scale.
        """
        # Find the kink in the log-log spectrum
        log_k = np.log(wavenumbers)
        log_E = np.log(energy_spectrum)
        
        # We look for the intersection of two lines
        # This is a simplified implementation finding the max curvature
        # In reality, we'd fit a piecewise linear function
        
        # Approximate 2nd derivative
        d2 = np.gradient(np.gradient(log_E, log_k), log_k)
        kink_idx = np.argmax(np.abs(d2))
        
        k_crit = wavenumbers[kink_idx]
        wavelength_crit = 2 * np.pi / k_crit # in km
        
        # Sigma_c here is related to the ratio of this scale to the Rossby radius
        rossby_radius = 1000.0 # km (approx)
        sigma_c = wavelength_crit / rossby_radius
        
        return {
            'critical_wavelength_km': wavelength_crit,
            'sigma_c': sigma_c,
            'spectral_slope_synoptic': (log_E[kink_idx] - log_E[0]) / (log_k[kink_idx] - log_k[0]),
            'spectral_slope_mesoscale': (log_E[-1] - log_E[kink_idx]) / (log_k[-1] - log_k[kink_idx])
        }

    def analyze_vertical_structure(self, pressure_levels: np.ndarray, temperature_profiles: np.ndarray) -> Dict[str, float]:
        """
        Analyzes vertical stability (Tropopause detection).
        sigma_c(z) profile.
        """
        # Lapse rate = -dT/dz
        # We approximate z from pressure using scale height H ~ 7-8km
        z = -7.0 * np.log(pressure_levels / 1000.0)
        
        lapse_rates = -np.gradient(temperature_profiles, z, axis=1)
        
        # Tropopause is where lapse rate drops below 2 K/km
        tropopause_indices = np.argmax(lapse_rates < 2.0, axis=1)
        tropopause_heights = z[tropopause_indices]
        
        return {
            'mean_tropopause_height': np.mean(tropopause_heights),
            'stability_profile': np.mean(lapse_rates, axis=0).tolist()
        }
