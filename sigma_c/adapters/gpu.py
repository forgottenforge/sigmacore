#!/usr/bin/env python3
"""
Sigma-C GPU Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for GPU Kernel Optimization and Benchmarking.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
from ..stats import jonckheere_terpstra_test, isotonic_regression_with_ci, pool_adjacent_violators
import numpy as np
import time
from typing import Any, Dict, List, Optional
from tqdm import tqdm

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

class GPUAdapter(SigmaCAdapter):
    """
    Adapter for GPU Kernel Optimization.
    Ported from rev_gpu.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not _HAS_CUPY:
            print("⚠️ CuPy not found. GPU adapter running in simulation mode.")
        else:
            self._warmup()
            
    def _warmup(self):
        size = 1024
        A = cp.random.random((size, size), dtype=cp.float32)
        B = cp.random.random((size, size), dtype=cp.float32)
        _ = cp.dot(A, B)
        cp.cuda.runtime.deviceSynchronize()
            
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Compute performance degradation observable.
        """
        # data is assumed to be GFLOPS
        perf_norm = data / (np.max(data) + 1e-10)
        return 1.0 - perf_norm

    def run_benchmark(self, size=1024, n_launch=10, n_mem=0):
        """
        Run a single benchmark point.
        """
        if not _HAS_CUPY:
            # Simulation mode
            base_time = 0.1
            overhead = n_mem * 0.01
            return (2 * size**3 * n_launch) / ((base_time + overhead) * 1e9)
            
        # Allocate overhead
        mem = [cp.random.random((size//2, size//2), dtype=cp.float32)
               for _ in range(n_mem)]
        for m in mem:
            _ = cp.sum(m[::32, ::32])
            
        # Workload
        A = cp.random.random((size, size), dtype=cp.float32)
        B = cp.random.random((size, size), dtype=cp.float32)
        
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        for _ in range(n_launch):
            C = cp.dot(A, B)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - start
        
        del mem
        flops = 2 * size**3 * n_launch
        return flops / (elapsed * 1e9)

    def auto_tune(self, alpha_levels=None, epsilon_points=24):
        """
        Run auto-tuning loop (E2 experiment).
        """
        if alpha_levels is None:
            alpha_levels = [0.0, 0.15, 0.3, 0.45, 0.6]
            
        epsilon = np.linspace(0.0, 0.6, epsilon_points)
        all_sigma_c = []
        
        results = {}
        
        for alpha in alpha_levels:
            gflops = []
            for eps in tqdm(epsilon, desc=f"Tuning alpha={alpha}", leave=False):
                eps_tilde = 1 - (1 - eps)**(1 + 3*alpha)
                n_mem = int(4 + 32*alpha + 6*eps_tilde + 10*eps_tilde**2)
                n_launch = max(1, int(2 + 8*alpha + 4*eps_tilde + 6*eps_tilde**2))
                
                # Run multiple reps
                reps = []
                for _ in range(5):
                    reps.append(self.run_benchmark(n_launch=n_launch, n_mem=n_mem))
                gflops.append(np.mean(reps))
            
            # Compute Susceptibility
            obs = self.get_observable(np.array(gflops))
            analysis = self.compute_susceptibility(epsilon, obs)
            
            all_sigma_c.append(analysis['sigma_c'])
            results[alpha] = {
                'sigma_c': analysis['sigma_c'],
                'kappa': analysis['kappa'],
                'gflops': gflops
            }
            
        # Statistical Tests
        jt_test = jonckheere_terpstra_test(all_sigma_c, alpha_levels)
        iso_reg = isotonic_regression_with_ci(np.array(alpha_levels), np.array(all_sigma_c))
        
        return {
            'tuning_results': results,
            'statistics': {
                'jonckheere_terpstra': jt_test,
                'isotonic_regression': {
                    'fitted': iso_reg['fitted'].tolist(),
                    'ci_lower': iso_reg['ci_lower'].tolist(),
                    'ci_upper': iso_reg['ci_upper'].tolist()
                }
            }
        }
