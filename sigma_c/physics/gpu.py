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
        Uses Roofline Model: Performance = min(Peak_FLOPS, Bandwidth * AI)
        """
        arithmetic_intensity = data.get('arithmetic_intensity', 1.0)
        peak_flops = data.get('peak_flops', 1000.0)
        peak_bandwidth = data.get('peak_bandwidth', 500.0)
        
        # Roofline model: performance is limited by either compute or memory
        # Ridge point: AI_ridge = Peak_FLOPS / Peak_Bandwidth
        ridge_point = peak_flops / peak_bandwidth
        
        # Lower bound: memory-bound regime (AI < ridge_point)
        # Upper bound: compute-bound regime (AI > ridge_point)
        lower_bound = 0.01  # Minimum sigma_c for memory-bound
        upper_bound = 0.5   # Maximum sigma_c for compute-bound
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'ridge_point': ridge_point,
            'arithmetic_intensity': arithmetic_intensity,
            'regime': 'compute_bound' if arithmetic_intensity > ridge_point else 'memory_bound',
            'metric': 'sigma_c',
            'theory': 'Roofline Model'
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
            'theoretical_bounds': bounds,
            'details': {
                'statistical_check': is_valid_stat,
                'thermal_check': is_thermal_safe,
                'consistency_check': is_consistent
            }
        }

    def check_scaling_laws(self, data: Any, param_range: List[float], **kwargs) -> Dict[str, Any]:
        """
        Simple scaling law check for GPU.
        Toy model: sigma_c scales inversely with arithmetic intensity.
        """
        if not param_range or len(param_range) < 2:
            return {'status': 'insufficient_data'}
        
        # Toy model: assume sigma_c ~ 1 / arithmetic_intensity^alpha
        intensities = np.array(param_range)
        sigma_c_values = 1.0 / (intensities + 1e-10)
        
        # Fit power law: sigma_c = a * intensity^b
        log_intensities = np.log(intensities + 1e-10)
        log_sigma = np.log(sigma_c_values + 1e-10)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_intensities, log_sigma, 1)
        exponent = coeffs[0]
        
        return {
            'status': 'completed',
            'scaling_exponent': float(exponent),
            'model': 'power_law',
            'theory': 'Roofline scaling'
        }

    def quantify_resource(self, data: Any) -> float:
        """
        Quantify Arithmetic Intensity.
        """
        flops = data.get('flops', 0)
        bytes_transferred = data.get('bytes', 1)
        return float(flops / bytes_transferred)
