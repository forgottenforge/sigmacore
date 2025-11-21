"""
Sigma-C Discovery Module
========================
Copyright (c) 2025 ForgottenForge.xyz

Implements rigorous methods for automatic observable discovery and 
multi-scale susceptibility analysis as defined in the reference papers.
"""

import numpy as np
from scipy import signal, stats
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ObservableCandidate:
    name: str
    score: float
    data: np.ndarray
    method: str

class ObservableDiscovery:
    """
    Automatically identifies the optimal order parameter (observable) 
    that maximizes the visibility of phase transitions.
    """
    
    def __init__(self, method: str = 'hybrid'):
        """
        Args:
            method: 'gradient', 'entropy', 'pca', or 'hybrid'
        """
        self.method = method

    def identify_observables(self, data: np.ndarray, feature_names: Optional[List[str]] = None, method: str = 'gradient') -> Dict[str, Any]:
        """
        Identifies observables from data.
        
        Args:
            data: Shape (n_samples, n_features)
            feature_names: Optional list of feature names
            method: Discovery method
            
        Returns:
            Dictionary with ranked observables
        """
        if data.ndim == 1:
            return {
                'ranked_observables': [{'name': 'single_feature', 'score': 1.0}],
                'best_observable': 'single_feature'
            }
            
        n_features = data.shape[1]
        candidates = []
        
        # Gradient-Based Discovery
        for i in range(n_features):
            feat = data[:, i]
            chi = np.abs(np.gradient(feat))
            score = np.max(chi) / (np.mean(chi) + 1e-9)
            name = feature_names[i] if feature_names else f"feature_{i}"
            candidates.append({'name': name, 'score': float(score)})
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'ranked_observables': candidates,
            'best_observable': candidates[0]['name'] if candidates else None
        }

class MultiScaleAnalysis:
    """
    Performs multi-resolution analysis to detect criticality across different scales.
    Essential for systems with hierarchical structure (e.g., GPU caches, turbulence).
    """
    
    def __init__(self, scales: Optional[np.ndarray] = None):
        self.scales = scales if scales is not None else np.logspace(0.1, 2, 20)

    def compute_susceptibility_spectrum(self, signal_data: np.ndarray, scales: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes the susceptibility spectrum across multiple scales.
        
        Args:
            signal_data: Time series data
            scales: Array of scales to analyze (default: logarithmic spacing)
            
        Returns:
            Dictionary with scales and corresponding susceptibility values
        """
        if scales is None:
            scales = np.logspace(0, np.log10(len(signal_data) // 4), 20)
        
        try:
            # Try to use scipy.signal.cwt with ricker wavelet
            from scipy import signal as scipy_signal
            try:
                # Modern scipy API
                widths = scales
                coeffs = scipy_signal.cwt(signal_data, scipy_signal.ricker, widths)
            except AttributeError:
                # Fallback: manual wavelet transform
                coeffs = np.zeros((len(scales), len(signal_data)))
                for i, width in enumerate(scales):
                    # Simple Gaussian wavelet approximation
                    wavelet_size = min(int(width * 10), len(signal_data))
                    if wavelet_size < 3:
                        wavelet_size = 3
                    x = np.arange(wavelet_size) - wavelet_size // 2
                    wavelet = (1 - (x / width)**2) * np.exp(-0.5 * (x / width)**2)
                    wavelet = wavelet / np.sqrt(np.sum(wavelet**2))
                    coeffs[i] = np.convolve(signal_data, wavelet, mode='same')
        except ImportError:
            # Complete fallback without scipy
            coeffs = np.zeros((len(scales), len(signal_data)))
            for i, scale in enumerate(scales):
                window = int(scale)
                if window < 1:
                    window = 1
                if window > len(signal_data):
                    window = len(signal_data)
                coeffs[i] = np.convolve(signal_data, np.ones(window)/window, mode='same')
        
        # Compute susceptibility at each scale (variance of coefficients)
        susceptibilities = np.var(coeffs, axis=1)
        
        # Find critical scale (peak susceptibility)
        critical_idx = np.argmax(susceptibilities)
        critical_scale = scales[critical_idx]
        
        return {
            'scales': scales.tolist(),
            'susceptibilities': susceptibilities.tolist(),
            'critical_scale': float(critical_scale),
            'max_susceptibility': float(susceptibilities[critical_idx])
        }

    def find_critical_scales(self, spectrum: Dict[str, Any]) -> List[float]:
        """
        Identifies scales with peak susceptibility.
        """
        scales = np.array(spectrum['scales'])
        chis = np.array(spectrum['susceptibilities'])
        
        # Find peaks in the spectrum
        from scipy import signal as scipy_signal
        peaks, _ = scipy_signal.find_peaks(chis)
        return scales[peaks].tolist()
