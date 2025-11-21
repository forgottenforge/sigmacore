"""
Sigma-C Edge Adapter
====================
Copyright (c) 2025 ForgottenForge.xyz

Optimizes performance under energy and resource constraints.
Focuses on Energy Efficiency as the critical observable.
"""

from ..core.base import SigmaCAdapter
import numpy as np
from typing import Dict, Any, List

class EdgeAdapter(SigmaCAdapter):
    """
    Adapter for Edge Computing (IoT, Embedded, Mobile).
    Optimizes for Performance/Watt.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        if config is None:
            config = {}
        self.target_hardware = config.get('hardware', 'generic_mcu')
        
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Returns Energy Efficiency (Performance / Power).
        """
        # Expecting data columns: [performance, power]
        if data.shape[1] < 2:
            return 0.0
        perf = data[:, 0]
        power = data[:, 1]
        efficiency = perf / (power + 1e-9)
        return float(np.mean(efficiency))
    
    def analyze_power_efficiency(self, frequency: np.ndarray, power: np.ndarray, performance: np.ndarray) -> Dict[str, float]:
        """
        Analyzes the trade-off between frequency, power, and performance.
        Finds the 'Knee Point' where dPerf/dPower drops below a threshold.
        """
        # dPerf / dPower
        d_perf = np.gradient(performance)
        d_power = np.gradient(power)
        marginal_efficiency = d_perf / (d_power + 1e-9)
        
        # Critical point is where marginal efficiency drops significantly
        # This is often where dynamic voltage frequency scaling (DVFS) should stop
        critical_idx = np.argmax(marginal_efficiency < 0.5 * np.max(marginal_efficiency))
        
        return {
            'critical_frequency': float(frequency[critical_idx]),
            'max_efficiency': float(np.max(performance / (power + 1e-9))),
            'sigma_c_power': float(power[critical_idx] / np.max(power)) # Normalized power at critical point
        }

    def _domain_specific_validate(self, result: Dict[str, Any]) -> bool:
        return result.get('sigma_c_power', 0.0) > 0.0
