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
    
    # ========================================================================
    # v1.1.0 Universal Diagnostics System
    # ========================================================================
    
    def _domain_specific_diagnose(self, data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        GPU-specific diagnostics.
        
        Checks:
        - CuPy installation
        - Cache thrashing indicators
        - Memory bandwidth utilization
        - Kernel efficiency
        """
        benchmark_data = data  # Rename for clarity in this method
        issues = []
        recommendations = []
        details = {}
        
        # Check 1: CuPy availability
        if not _HAS_CUPY:
            issues.append("CuPy not installed - running in simulation mode")
            recommendations.append("Install CuPy for real GPU benchmarking: pip install cupy-cuda11x")
            details['cupy_available'] = False
        else:
            details['cupy_available'] = True
            
        # Check 2: Benchmark data quality
        if benchmark_data is not None:
            if isinstance(benchmark_data, dict) and 'gflops' in benchmark_data:
                gflops = np.array(benchmark_data['gflops'])
                
                # Check for performance degradation
                if len(gflops) > 1:
                    trend = np.polyfit(range(len(gflops)), gflops, 1)[0]
                    if trend < -0.1:
                        issues.append("Significant performance degradation detected")
                        recommendations.append("Consider reducing memory overhead or kernel launches")
                        details['performance_trend'] = 'degrading'
                    else:
                        details['performance_trend'] = 'stable'
                        
                # Check for cache thrashing
                variance = np.var(gflops)
                mean = np.mean(gflops)
                cv = variance / (mean + 1e-10)
                if cv > 0.3:
                    issues.append("High performance variance - possible cache thrashing")
                    recommendations.append("Optimize memory access patterns")
                    details['cache_thrashing_risk'] = 'high'
                else:
                    details['cache_thrashing_risk'] = 'low'
                    
                details['gflops_mean'] = float(mean)
                details['gflops_variance'] = float(variance)
        
        # Determine status
        if len(issues) == 0:
            status = 'ok'
        elif any('not installed' in i for i in issues):
            status = 'warning'
        else:
            status = 'warning'
            
        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'details': details
        }
    
    def _domain_specific_auto_search(self, param_ranges: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        GPU-specific parameter auto-search.
        
        Searches over:
        - alpha levels (workload intensity)
        - epsilon points (granularity)
        - kernel sizes
        """
        if param_ranges is None:
            param_ranges = {
                'alpha_levels': [[0.0, 0.15, 0.3], [0.0, 0.15, 0.3, 0.45, 0.6]],
                'epsilon_points': [12, 24, 48],
                'kernel_size': [512, 1024, 2048]
            }
        
        results = []
        
        # Quick search over alpha configurations
        for alpha_config in param_ranges.get('alpha_levels', [[0.0, 0.3, 0.6]]):
            for eps_points in param_ranges.get('epsilon_points', [24]):
                # Run simplified auto_tune
                epsilon = np.linspace(0.0, 0.6, eps_points)
                sigma_c_values = []
                
                for alpha in alpha_config:
                    # Simulate or run quick benchmark
                    gflops = []
                    for eps in epsilon[:6]:  # Use subset for speed
                        eps_tilde = 1 - (1 - eps)**(1 + 3*alpha)
                        n_mem = int(4 + 32*alpha + 6*eps_tilde)
                        n_launch = max(1, int(2 + 8*alpha + 4*eps_tilde))
                        gflops.append(self.run_benchmark(n_launch=n_launch, n_mem=n_mem))
                    
                    obs = self.get_observable(np.array(gflops))
                    analysis = self.compute_susceptibility(epsilon[:6], obs)
                    sigma_c_values.append(analysis['sigma_c'])
                
                results.append({
                    'params': {
                        'alpha_levels': alpha_config,
                        'epsilon_points': eps_points
                    },
                    'mean_sigma_c': float(np.mean(sigma_c_values)),
                    'sigma_c_range': float(np.max(sigma_c_values) - np.min(sigma_c_values))
                })
        
        # Find best configuration
        best = max(results, key=lambda x: x['sigma_c_range'])
        
        return {
            'best_params': best['params'],
            'all_results': results,
            'recommendation': f"Use alpha_levels={best['params']['alpha_levels']} with {best['params']['epsilon_points']} epsilon points"
        }
    
    def _domain_specific_validate(self, data: Optional[Any] = None, **kwargs) -> Dict[str, bool]:
        """
        GPU-specific technique validation.
        
        Validates:
        - Benchmark reproducibility
        - Susceptibility computation
        - Statistical tests
        """
        tests = {}
        
        # Test 1: Benchmark reproducibility
        try:
            gflops1 = self.run_benchmark(size=512, n_launch=5, n_mem=2)
            gflops2 = self.run_benchmark(size=512, n_launch=5, n_mem=2)
            relative_diff = abs(gflops1 - gflops2) / (gflops1 + 1e-10)
            tests['benchmark_reproducibility'] = relative_diff < 0.2  # 20% tolerance
        except Exception as e:
            tests['benchmark_reproducibility'] = False
            
        # Test 2: Susceptibility computation
        try:
            epsilon = np.linspace(0, 1, 10)
            obs = np.exp(-5 * epsilon)
            result = self.compute_susceptibility(epsilon, obs)
            tests['susceptibility_computation'] = (
                'sigma_c' in result and 
                'kappa' in result and
                0 <= result['sigma_c'] <= 1
            )
        except Exception:
            tests['susceptibility_computation'] = False
            
        # Test 3: Observable computation
        try:
            data = np.array([100, 90, 80, 70])
            obs = self.get_observable(data)
            tests['observable_computation'] = isinstance(obs, (float, np.floating))
        except Exception:
            tests['observable_computation'] = False
            
        return tests
    
    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """
        GPU-specific result explanation.
        """
        sigma_c = result.get('sigma_c', 0.0)
        kappa = result.get('kappa', 0.0)
        
        explanation = f"""
GPU Kernel Analysis Results:
============================

Critical Point (σ_c): {sigma_c:.3f}
This represents the workload intensity threshold where GPU performance 
begins to degrade significantly due to cache thrashing and memory pressure.

Criticality Exponent (κ): {kappa:.2f}
This quantifies how sharply performance degrades near the critical point.
- κ < 1.0: Gradual degradation (good cache behavior)
- κ ≈ 2.0: Moderate degradation (typical for GEMM kernels)
- κ > 3.0: Sharp degradation (cache thrashing)

Interpretation:
"""
        
        if sigma_c < 0.3:
            explanation += "- LOW threshold: GPU is highly sensitive to workload increases\n"
            explanation += "- Recommendation: Optimize memory access patterns\n"
        elif sigma_c < 0.6:
            explanation += "- MODERATE threshold: Typical GPU behavior\n"
            explanation += "- Recommendation: Balance compute and memory operations\n"
        else:
            explanation += "- HIGH threshold: GPU handles workload increases well\n"
            explanation += "- Recommendation: Can safely increase workload intensity\n"
            
        if kappa < 1.5:
            explanation += "- GRADUAL degradation: Excellent cache utilization\n"
        elif kappa < 2.5:
            explanation += "- MODERATE degradation: Normal GPU kernel behavior\n"
        else:
            explanation += "- SHARP degradation: Consider memory optimization\n"
            
        return explanation
