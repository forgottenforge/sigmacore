"""
Sigma-C Discovery Module
========================
Copyright (c) 2025 ForgottenForge.xyz

Implements methods for automatic observable discovery and
multi-scale susceptibility analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional

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

    def identify_observables(self, data: np.ndarray, feature_names: Optional[List[str]] = None, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Identifies observables from data.

        Args:
            data: Shape (n_samples, n_features)
            feature_names: Optional list of feature names
            method: Discovery method (defaults to self.method)

        Returns:
            Dictionary with ranked observables
        """
        if method is None:
            method = self.method

        if data.ndim == 1:
            return {
                'ranked_observables': [{'name': 'single_feature', 'score': 1.0}],
                'best_observable': 'single_feature'
            }

        n_features = data.shape[1]
        candidates = []

        for i in range(n_features):
            feat = data[:, i]
            chi = np.abs(np.gradient(feat))
            score = np.max(chi) / (np.mean(chi) + 1e-9)
            name = feature_names[i] if feature_names else f"feature_{i}"
            candidates.append({'name': name, 'score': float(score)})

        candidates.sort(key=lambda x: x['score'], reverse=True)

        return {
            'ranked_observables': candidates,
            'best_observable': candidates[0]['name'] if candidates else None
        }

class MultiScaleAnalysis:
    """
    Performs multi-resolution analysis to detect criticality across different scales.
    """

    def __init__(self, scales: Optional[np.ndarray] = None):
        self.scales = scales

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
            scales = self.scales
        if scales is None:
            max_scale = max(len(signal_data) // 4, 2)
            scales = np.logspace(0, np.log10(max_scale), 20)

        try:
            from scipy import signal as scipy_signal
            try:
                widths = scales
                coeffs = scipy_signal.cwt(signal_data, scipy_signal.ricker, widths)
            except AttributeError:
                coeffs = np.zeros((len(scales), len(signal_data)))
                for i, width in enumerate(scales):
                    wavelet_size = min(int(width * 10), len(signal_data))
                    if wavelet_size < 3:
                        wavelet_size = 3
                    x = np.arange(wavelet_size) - wavelet_size // 2
                    wavelet = (1 - (x / width)**2) * np.exp(-0.5 * (x / width)**2)
                    wavelet = wavelet / np.sqrt(np.sum(wavelet**2))
                    coeffs[i] = np.convolve(signal_data, wavelet, mode='same')
        except ImportError:
            coeffs = np.zeros((len(scales), len(signal_data)))
            for i, scale in enumerate(scales):
                window = max(1, min(int(scale), len(signal_data)))
                coeffs[i] = np.convolve(signal_data, np.ones(window)/window, mode='same')

        susceptibilities = np.var(coeffs, axis=1)

        critical_idx = int(np.argmax(susceptibilities))
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

        from scipy import signal as scipy_signal
        peaks, _ = scipy_signal.find_peaks(chis)
        return scales[peaks].tolist()
