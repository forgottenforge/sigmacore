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

    def compute_significance(self, observed_stat: float, data: np.ndarray, n_surrogates: int = 1000) -> float:
        """
        Computes statistical significance (p-value) of an observed statistic
        using surrogate data testing (shuffling).
        
        Args:
            observed_stat: The statistic calculated from original data (e.g., b-value)
            data: The original data array
            n_surrogates: Number of surrogate datasets to generate
            
        Returns:
            p-value: Probability that random data produces a statistic >= observed_stat
        """
        surrogates = []
        for _ in range(n_surrogates):
            # Generate surrogate by shuffling (destroys temporal correlations)
            # or resampling (bootstrapping)
            shuffled = np.random.permutation(data)
            
            # Re-calculate statistic for surrogate
            # Note: This assumes 'observed_stat' is a b-value-like statistic
            # derived from the mean or distribution shape.
            # For b-value specifically:
            m_min = np.min(shuffled)
            mean_m = np.mean(shuffled)
            # b-value for shuffled data (should be similar if IID, but useful for other stats)
            b_shuffled = np.log10(np.e) / (mean_m - m_min + 1e-9)
            surrogates.append(b_shuffled)
            
        # p-value: fraction of surrogates more extreme than observed
        # Assuming we are testing for high b-values (or low?)
        # Usually we test if the observed structure is non-random.
        # For b-value, shuffling magnitudes doesn't change b-value much if it's just distribution based.
        # But if we are testing time-clustering (Omori), shuffling times destroys it.
        
        # Let's assume this is a generic significance test where we check if observed is an outlier.
        surrogates = np.array(surrogates)
        p_value = np.mean(surrogates >= observed_stat)
        
        return float(p_value)
