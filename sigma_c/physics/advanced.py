"""
Advanced Physics Module
=======================
Implements advanced theoretical tools:
- Renormalization Group (RG) Flow
- Harmonic Analysis (Log-periodic oscillations)
- Chaos Quantification (Lyapunov exponents)

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import curve_fit

class RenormalizationGroup:
    """
    Analyzes scaling behavior and RG flow near critical points.
    """
    
    def analyze_flow(self, sigma_values: np.ndarray, observables: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the flow of the observable under rescaling of sigma.
        """
        # Calculate beta function: beta(g) = d(g)/d(ln sigma)
        # where g is the coupling constant (observable)
        
        log_sigma = np.log(sigma_values)
        dg_dlnsigma = np.gradient(observables, log_sigma)
        
        # Find fixed points where beta(g) = 0
        zero_crossings = np.where(np.diff(np.sign(dg_dlnsigma)))[0]
        fixed_points = []
        
        for idx in zero_crossings:
            sigma_star = sigma_values[idx]
            g_star = observables[idx]
            stability = "Stable" if dg_dlnsigma[idx] > dg_dlnsigma[idx+1] else "Unstable"
            fixed_points.append({
                'sigma': sigma_star,
                'coupling': g_star,
                'stability': stability
            })
            
        return {
            'beta_function': dg_dlnsigma,
            'fixed_points': fixed_points,
            'flow_type': 'Asymptotic Freedom' if dg_dlnsigma[-1] < 0 else 'Strong Coupling'
        }

class HarmonicAnalyzer:
    """
    Detects log-periodic oscillations characteristic of discrete scale invariance.
    """
    
    def detect_log_periodicity(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit to: y(x) = A + B(x_c - x)^z [1 + C cos(omega ln(x_c - x) + phi)]
        """
        # Simplified detection using FFT on log-time
        # We assume x is approaching x_c from below
        
        try:
            x_c = np.max(x) * 1.01 # Guess critical point slightly beyond range
            log_dist = np.log(x_c - x)
            
            # Detrend
            p = np.polyfit(log_dist, y, 1)
            trend = np.polyval(p, log_dist)
            residuals = y - trend
            
            # FFT
            fft_vals = np.fft.rfft(residuals)
            freqs = np.fft.rfftfreq(len(residuals))
            
            peak_idx = np.argmax(np.abs(fft_vals[1:])) + 1
            dominant_freq = freqs[peak_idx]
            amplitude = np.abs(fft_vals[peak_idx])
            
            return {
                'has_log_periodicity': amplitude > np.std(residuals) * 2,
                'frequency': dominant_freq,
                'amplitude': amplitude,
                'scaling_factor': np.exp(1/dominant_freq) if dominant_freq > 0 else 0
            }
        except Exception:
            return {'has_log_periodicity': False}

class ChaosQuantifier:
    """
    Quantifies chaos using Lyapunov exponents and entropy.
    """
    
    def compute_lyapunov(self, time_series: np.ndarray, dt: float = 1.0) -> float:
        """
        Estimate maximal Lyapunov exponent (Rosenstein algorithm simplified).
        """
        # Very simplified placeholder for 1D series
        # In reality, needs embedding dimension and phase space reconstruction
        
        # Check divergence of nearby points
        n = len(time_series)
        if n < 10:
            return 0.0
            
        # Naive approach: fit exponential divergence
        diffs = np.abs(np.diff(time_series))
        # Avoid log(0)
        diffs = diffs[diffs > 0]
        if len(diffs) < 2:
            return 0.0
            
        log_diffs = np.log(diffs)
        # Slope of log diffs over time roughly indicates expansion rate
        lambda_max = np.polyfit(np.arange(len(log_diffs)), log_diffs, 1)[0]
        
        return lambda_max / dt
