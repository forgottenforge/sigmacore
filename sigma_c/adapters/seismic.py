"""
Sigma-C Seismic Adapter
=======================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Seismology and Earthquake Analysis.
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
        if len(data) < 2:
            return 0.0
        return float(np.mean(data))

    def analyze_gutenberg_richter(self, magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Analyzes deviation from Gutenberg-Richter Law: log10(N) = a - b*M.
        Uses Maximum Likelihood Estimation for b-value.
        """
        m_min = np.min(magnitudes)
        mean_m = np.mean(magnitudes)
        denom = mean_m - m_min
        if denom < 1e-9:
            return {
                'b_value': float('nan'),
                'm_min': float(m_min),
                'criticality': float('nan'),
                'warning': 'All magnitudes are identical; b-value undefined'
            }

        b_value = np.log10(np.e) / denom

        if abs(b_value) < 1e-12:
            criticality = float('inf')
        else:
            criticality = 1.0 / b_value

        return {
            'b_value': float(b_value),
            'm_min': float(m_min),
            'criticality': float(criticality)
        }

    def analyze_omori_scaling(self, event_times: np.ndarray) -> Dict[str, float]:
        """
        Analyzes Omori Law for aftershocks: n(t) = K / (c + t)^p.
        """
        counts, bins = np.histogram(event_times, bins='auto')
        centers = (bins[:-1] + bins[1:]) / 2

        valid = counts > 0
        if np.sum(valid) < 2:
            return {'p_value': float('nan'), 'decay_constant': float('nan')}

        log_t = np.log(centers[valid])
        log_n = np.log(counts[valid])

        slope, intercept = np.polyfit(log_t, log_n, 1)
        p_value = -slope

        return {
            'p_value': float(p_value),
            'decay_constant': float(np.exp(intercept))
        }

    def compute_significance(self, observed_stat: float, data: np.ndarray, n_surrogates: int = 1000) -> float:
        """
        Computes statistical significance using bootstrap resampling.

        Args:
            observed_stat: The statistic calculated from original data
            data: The original data array
            n_surrogates: Number of bootstrap samples

        Returns:
            p-value: Probability that a bootstrap sample produces a statistic >= observed_stat
        """
        n = len(data)
        surrogates = []
        for _ in range(n_surrogates):
            resampled = np.random.choice(data, size=n, replace=True)
            m_min = np.min(resampled)
            mean_m = np.mean(resampled)
            denom = mean_m - m_min
            if denom < 1e-9:
                continue
            b_resampled = np.log10(np.e) / denom
            surrogates.append(b_resampled)

        if len(surrogates) == 0:
            return 1.0

        surrogates = np.array(surrogates)
        p_value = np.mean(np.abs(surrogates - np.mean(surrogates)) >= np.abs(observed_stat - np.mean(surrogates)))

        return float(p_value)
