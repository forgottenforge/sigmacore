"""
Sigma-C Edge Adapter
====================
Copyright (c) 2025 ForgottenForge.xyz

Optimizes performance under energy and resource constraints.
Focuses on Energy Efficiency as the critical observable.
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any

class EdgeAdapter(SigmaCAdapter):
    """
    Adapter for Edge Computing (IoT, Embedded, Mobile).
    Optimizes for Performance/Watt.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.target_hardware = self.config.get('hardware', 'generic_mcu')

    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns Energy Efficiency (Performance / Power).
        """
        data = np.asarray(data)
        if data.ndim < 2 or data.shape[1] < 2:
            return 0.0
        perf = data[:, 0]
        power = data[:, 1]
        efficiency = perf / (power + 1e-9)
        return float(np.mean(efficiency))

    def analyze_power_efficiency(self, frequency: np.ndarray, power: np.ndarray,
                                  performance: np.ndarray) -> Dict[str, float]:
        """
        Analyzes the trade-off between frequency, power, and performance.
        Finds the 'Knee Point' where dPerf/dPower drops below a threshold.
        """
        d_perf = np.gradient(performance)
        d_power = np.gradient(power)
        marginal_efficiency = d_perf / (d_power + 1e-9)

        critical_idx = int(np.argmax(marginal_efficiency < 0.5 * np.max(marginal_efficiency)))

        return {
            'critical_frequency': float(frequency[critical_idx]),
            'max_efficiency': float(np.max(performance / (power + 1e-9))),
            'sigma_c_power': float(power[critical_idx] / np.max(power))
        }

    def _domain_specific_validate(self, data: Any = None, **kwargs) -> Dict[str, bool]:
        """Validate edge computing configuration."""
        checks = {
            'hardware_configured': self.target_hardware != 'generic_mcu',
            'basic_validation': True
        }
        if data is not None:
            data = np.asarray(data)
            checks['data_has_columns'] = data.ndim >= 2 and data.shape[1] >= 2
        return checks
