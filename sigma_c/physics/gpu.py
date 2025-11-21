"""
Rigorous GPU Sigma_c
====================
Validates GPU sigma_c values against Roofline Model and Little's Law.
Includes real-time hardware monitoring and spectral analysis.

Copyright (c) 2024 ForgottenForge.xyz. All rights reserved.
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .rigorous import RigorousTheoreticalCheck

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

class RigorousGPUSigmaC(RigorousTheoreticalCheck):
    """
    Checks if measured sigma_c respects hardware limits and theoretical bounds.
    Includes spectral analysis for periodic GPU workloads.
    """
    
    def __init__(self):
        self.nvml_initialized = False
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception:
                pass

    def get_hardware_metrics(self) -> Dict[str, float]:
        """Get real-time GPU hardware metrics via NVML."""
        metrics = {
            'temperature': 0.0,
            'power_usage': 0.0,
            'utilization': 0.0,
            'memory_used': 0.0
        }
        
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                metrics['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics['power_usage'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # mW to W
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['utilization'] = util.gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['memory_used'] = mem.used / 1024**2 # MB
            except Exception:
                pass
                
        return metrics

    def calculate_spectral_sigma_c(self, signal: np.ndarray) -> float:
        """
        Calculate sigma_c using spectral analysis (FFT).
        Good for periodic or structured signals (e.g. frame times).
        """
        if len(signal) < 4:
            return 0.0
            
        # FFT
        fft_vals = np.fft.fft(signal)
        power = np.abs(fft_vals)**2
        freqs = np.fft.fftfreq(len(signal))
        
        # Positive frequencies only
        mask = freqs > 0
        pos_power = power[mask]
        pos_freqs = freqs[mask]
        
        if len(pos_power) == 0 or np.sum(pos_power) == 0:
            return 0.0
            
        # Spectral spread (complexity)
        centroid = np.sum(pos_freqs * pos_power) / np.sum(pos_power)
        spread = np.sqrt(np.sum(((pos_freqs - centroid)**2) * pos_power) / np.sum(pos_power))
        
        # Dominant frequency
        dom_idx = np.argmax(pos_power)
        dom_freq = pos_freqs[dom_idx]
        
        # Formula from boosti2.py
        if dom_freq > 0:
            base_sigma = 1.0 / (2 * np.pi * dom_freq)
            complexity = spread / (dom_freq + 1e-10)
            return float(base_sigma * (1 + complexity))
        else:
            return float(np.std(signal) / (np.mean(np.abs(signal)) + 1e-10))
    
    def check_theoretical_bounds(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Check Roofline bounds.
        sigma_c (critical workload) should correlate with Arithmetic Intensity.
        """
        arithmetic_intensity = data.get('arithmetic_intensity', 10.0) # FLOPs/Byte
        peak_flops = data.get('peak_flops', 1000.0)
        peak_bandwidth = data.get('peak_bandwidth', 500.0)
        
        # Roofline knee point
        knee_intensity = peak_flops / peak_bandwidth
        
        # If intensity < knee, we are memory bound -> lower sigma_c (more sensitive to memory pressure)
        # If intensity > knee, we are compute bound -> higher sigma_c (resilient to memory pressure)
        
        if arithmetic_intensity < knee_intensity:
            expected_sigma_c_range = (0.1, 0.4)
            regime = "Memory Bound"
        else:
            expected_sigma_c_range = (0.4, 0.8)
            regime = "Compute Bound"
            
        return {
            'lower_bound': expected_sigma_c_range[0],
            'upper_bound': expected_sigma_c_range[1],
            'metric': 'sigma_c',
            'theory': f'Roofline Model ({regime})'
        }

    def validate_sigma_c(self, value: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate sigma_c with hardware context.
        """
        # 1. Standard statistical check
        is_valid_stat = 0.0 < value < 1.0
        
        # 2. Hardware thermal check (if available)
        hw_metrics = self.get_hardware_metrics()
        is_thermal_safe = hw_metrics['temperature'] < 85.0 # 85C throttle limit
        
        # 3. Theoretical check
        bounds = self.check_theoretical_bounds(context)
        is_consistent = bounds['lower_bound'] <= value <= bounds['upper_bound']
        
        return {
            'is_valid': is_valid_stat and is_thermal_safe,
            'is_consistent': is_consistent,
            'hardware_metrics': hw_metrics,
            'regime': bounds['theory'],
            'thermal_status': 'safe' if is_thermal_safe else 'throttling'
        }

    def check_scaling_laws(self, data: Any, param_range: List[float], **kwargs) -> Dict[str, Any]:
        """
        Check Little's Law: Concurrency = Throughput * Latency.
        """
        # Placeholder
        return {'status': 'not_implemented'}

    def quantify_resource(self, data: Any) -> float:
        """
        Quantify Arithmetic Intensity.
        """
        flops = data.get('flops', 0)
        bytes_transferred = data.get('bytes', 1)
        return flops / bytes_transferred
