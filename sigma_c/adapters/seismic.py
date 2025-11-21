"""
Sigma-C Seismic Adapter
=======================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Seismology and Earthquake Prediction.
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any

class SeismicAdapter(SigmaCAdapter):
    """
    Adapter for Seismic Systems.
    Focuses on Gutenberg-Richter deviation and Omori scaling.
    """
    
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns the b-value or strain observable.
        """
        # Assuming data is magnitudes
        if len(data) < 2: return 0.0
        return float(np.mean(data))
    
    def analyze_gutenberg_richter(self, magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Analyzes deviation from Gutenberg-Richter Law: log10(N) = a - b*M.
        Near criticality (sigma_c), b-value fluctuates.
        """
        # Calculate b-value using Maximum Likelihood
        m_min = np.min(magnitudes)
        mean_m = np.mean(magnitudes)
        b_value = np.log10(np.e) / (mean_m - m_min)
        
        # Check for deviation in the tail (large magnitudes)
        # We compare theoretical N vs actual N for M > M_threshold
        
        return {
            'b_value': b_value,
            'm_min': m_min,
            'criticality': 1.0 / b_value # Lower b-value -> Higher stress -> Higher criticality
        }

    def analyze_omori_scaling(self, event_times: np.ndarray) -> Dict[str, float]:
        """
        Analyzes Omori Law for aftershocks: n(t) = K / (c + t)^p.
        p-value changes near sigma_c.
        """
        # We need time differences between mainshock and aftershocks
        # Assuming event_times is sorted and t=0 is mainshock
        
        # Fit p-value
        # log(n(t)) ~ -p * log(t)
        
        # Histogram of times
        counts, bins = np.histogram(event_times, bins='auto')
        centers = (bins[:-1] + bins[1:]) / 2
        
        # Filter zeros
        valid = counts > 0
        log_t = np.log(centers[valid])
        log_n = np.log(counts[valid])
        
        slope, intercept = np.polyfit(log_t, log_n, 1)
        p_value = -slope
        
        return {
            'p_value': p_value,
            'decay_constant': np.exp(intercept)
        }
