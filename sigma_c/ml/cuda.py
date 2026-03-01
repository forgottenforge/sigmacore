"""
Sigma-C CUDA Integration
=========================
Copyright (c) 2025 ForgottenForge.xyz

CUDA kernel monitoring and GPU-accelerated susceptibility computation.
Uses CuPy when available, falls back to NumPy otherwise.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Dict, Any, List
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None

# C++ CUDA header for reference (used in native builds)
CUDA_HEADER = """
// sigma_c_cuda.cuh -- C++ CUDA monitoring header
// Include in your .cu files for automatic kernel criticality tracking.
//
// Usage:
//   #include "sigma_c_cuda.cuh"
//
//   __global__ void my_kernel(float* data, int n) {
//       int idx = blockIdx.x * blockDim.x + threadIdx.x;
//       if (idx < n) data[idx] *= 2.0f;
//   }
//
//   // In host code:
//   sigma_c::KernelMonitor monitor;
//   monitor.start();
//   my_kernel<<<blocks, threads>>>(data, n);
//   monitor.stop();
//   auto metrics = monitor.get_metrics();

#ifndef SIGMA_C_CUDA_H
#define SIGMA_C_CUDA_H

#include <cuda_runtime.h>

namespace sigma_c {

struct KernelMetrics {
    float execution_time_ms;
    float sigma_c;
    int n_threads;
    int n_blocks;
};

class KernelMonitor {
private:
    cudaEvent_t start_, stop_;
    KernelMetrics metrics_;

public:
    KernelMonitor() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~KernelMonitor() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { cudaEventRecord(start_); }

    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&metrics_.execution_time_ms, start_, stop_);
    }

    KernelMetrics get_metrics() const { return metrics_; }
};

} // namespace sigma_c
#endif
"""


class CUDAMonitor:
    """
    Monitor CUDA kernel executions and track criticality from timing patterns.

    Uses CuPy events for precise GPU timing. Criticality is estimated from
    the coefficient of variation of kernel execution times: high variance
    relative to mean indicates the system is near a performance transition.

    Usage:
        from sigma_c.ml.cuda import CUDAMonitor

        monitor = CUDAMonitor()

        with monitor.track("my_kernel"):
            # Any GPU operation
            result = cp.dot(a, b)

        print(monitor.get_metrics())
        print(f"sigma_c = {monitor.compute_criticality():.3f}")
    """

    def __init__(self):
        if not _HAS_CUPY:
            raise ImportError("CuPy not installed. Run: pip install cupy-cuda12x")

        self._timings: List[Dict[str, Any]] = []

    def track(self, label: str = "kernel"):
        """
        Context manager for kernel timing.

        Args:
            label: Name for this kernel execution.
        """
        return _KernelTimer(self, label)

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Return all recorded kernel timings."""
        return list(self._timings)

    def compute_criticality(self) -> float:
        """
        Compute criticality from kernel timing distribution.

        Uses coefficient of variation: sigma_c = std(t) / mean(t).
        High CV indicates unstable performance (near transition).

        Returns:
            Criticality score in [0, 1].
        """
        if len(self._timings) < 2:
            return 0.0

        times = np.array([t['elapsed_ms'] for t in self._timings])
        mean_t = np.mean(times)
        std_t = np.std(times)

        if mean_t < 1e-9:
            return 0.0

        cv = std_t / mean_t
        return float(np.clip(cv, 0, 1))

    def reset(self):
        """Clear all recorded timings."""
        self._timings.clear()


class _KernelTimer:
    """Context manager for timing a single GPU kernel execution."""

    def __init__(self, monitor: CUDAMonitor, label: str):
        self._monitor = monitor
        self._label = label
        self._start = cp.cuda.Event()
        self._stop = cp.cuda.Event()

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *args):
        self._stop.record()
        self._stop.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(self._start, self._stop)

        self._monitor._timings.append({
            'label': self._label,
            'elapsed_ms': float(elapsed_ms),
        })


class CUDAAccelerator:
    """
    GPU-accelerated susceptibility computation using CuPy.

    Transfers epsilon/observable arrays to GPU, computes the derivative
    and peak detection on-device, then transfers results back.

    Falls back to NumPy when CuPy is unavailable.

    Usage:
        from sigma_c.ml.cuda import CUDAAccelerator

        accel = CUDAAccelerator()
        result = accel.compute_susceptibility(epsilon, observable)
        print(f"sigma_c = {result['sigma_c']:.4f}")
    """

    def __init__(self, device: int = 0):
        self._use_gpu = _HAS_CUPY
        if self._use_gpu:
            cp.cuda.Device(device).use()

    def compute_susceptibility(self, epsilon: np.ndarray, observable: np.ndarray,
                               kernel_sigma: float = 0.6) -> Dict[str, Any]:
        """
        Compute susceptibility on GPU (or CPU fallback).

        Args:
            epsilon: Control parameter array.
            observable: Observable values.
            kernel_sigma: Gaussian smoothing width.

        Returns:
            Dictionary with sigma_c, kappa, chi, chi_max, smoothed.
        """
        if self._use_gpu:
            return self._gpu_compute(epsilon, observable, kernel_sigma)
        else:
            return self._cpu_compute(epsilon, observable, kernel_sigma)

    def _gpu_compute(self, epsilon: np.ndarray, observable: np.ndarray,
                     kernel_sigma: float) -> Dict[str, Any]:
        """GPU path using CuPy."""
        eps_d = cp.asarray(epsilon, dtype=cp.float64)
        obs_d = cp.asarray(observable, dtype=cp.float64)

        # Gaussian smoothing on GPU
        from cupyx.scipy.ndimage import gaussian_filter1d
        smoothed_d = gaussian_filter1d(obs_d, kernel_sigma)

        # Susceptibility
        chi_d = cp.abs(cp.gradient(smoothed_d, eps_d))
        idx = int(cp.argmax(chi_d))
        chi_max = float(cp.max(chi_d))
        baseline = float(cp.mean(chi_d))
        kappa = chi_max / baseline if baseline > 1e-12 else 0.0

        # Transfer back to host
        return {
            'chi': cp.asnumpy(chi_d),
            'sigma_c': float(eps_d[idx]),
            'kappa': kappa,
            'chi_max': chi_max,
            'smoothed': cp.asnumpy(smoothed_d),
            'baseline': baseline,
        }

    def _cpu_compute(self, epsilon: np.ndarray, observable: np.ndarray,
                     kernel_sigma: float) -> Dict[str, Any]:
        """CPU fallback using NumPy/SciPy."""
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(observable.astype(np.float64), kernel_sigma)
        chi = np.abs(np.gradient(smoothed, epsilon))
        idx = int(np.argmax(chi))
        chi_max = float(np.max(chi))
        baseline = float(np.mean(chi)) if len(chi) > 0 else 1.0
        kappa = chi_max / baseline if baseline > 1e-12 else 0.0

        return {
            'chi': chi,
            'sigma_c': float(epsilon[idx]),
            'kappa': kappa,
            'chi_max': chi_max,
            'smoothed': smoothed,
            'baseline': baseline,
        }
