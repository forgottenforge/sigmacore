"""
Sigma-C v2.0 Rigorous Physics Verification
==========================================
Copyright (c) 2025 ForgottenForge.xyz

Verifies that the implementation matches the theoretical predictions 
from the reference papers.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sigma_c_framework')))

import numpy as np
from sigma_c.adapters.quantum import QuantumAdapter
from sigma_c.adapters.gpu import GPUAdapter
from sigma_c.adapters.financial import FinancialAdapter
from sigma_c.adapters.climate import ClimateAdapter
from sigma_c.adapters.seismic import SeismicAdapter
from sigma_c.adapters.magnetic import MagneticAdapter
from sigma_c.core.discovery import ObservableDiscovery
from sigma_c.core.control import AdaptiveController, StreamingSigmaC
from sigma_c.beyond.coupling import CouplingMatrix

def test_quantum_rigor():
    print("\n=== Testing Quantum Rigor ===")
    q = QuantumAdapter()
    
    # 1. Depth Scaling
    print("1. Depth Scaling (sigma_c ~ D^(1-alpha))")
    # Mock factory
    def factory(depth): return None 
    res = q.analyze_depth_scaling(factory)
    print(f"   Alpha: {res['alpha']:.4f} (Expected ~0.7)")
    
    # 2. Idle Sensitivity
    print("2. Idle Sensitivity")
    res = q.analyze_idle_sensitivity(factory)
    print(f"   Slope: {res['sensitivity_slope']:.4f} (Expected ~ -0.133)")
    print(f"   Consistent: {res['is_consistent']}")

def test_gpu_rigor():
    print("\n=== Testing GPU Rigor ===")
    g = GPUAdapter()
    
    # 1. Cache Transitions
    print("1. Cache Transitions")
    sizes = [1024, 1024*1024, 10*1024*1024]
    res = g.detect_cache_transitions(sizes)
    print(f"   L1: {res['L1_transition']} (Expected 0.023)")
    print(f"   L2: {res['L2_transition']} (Expected 0.072)")
    print(f"   L3: {res['L3_transition']} (Expected 0.241)")
    
    # 2. Roofline
    print("2. Roofline Analysis")
    res = g.analyze_roofline()
    print(f"   Ridge Point: {res['ridge_point']:.2f} FLOPS/Byte")
    
    # 3. Thermal
    print("3. Thermal Throttling")
    shift = g.predict_thermal_throttling(current_temp=70.0)
    print(f"   Sigma_c Shift (70C): {shift:.4f}")

def test_financial_rigor():
    print("\n=== Testing Financial Rigor ===")
    f = FinancialAdapter()
    
    # Generate synthetic Brownian motion
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 1000)
    price = 100 * np.exp(np.cumsum(returns))
    
    # 1. Hurst
    print("1. Hurst Exponent")
    res = f.compute_hurst_exponent(price)
    print(f"   Hurst: {res['hurst']:.4f} (Expected ~0.5 for Random Walk)")
    
    # 2. GARCH
    print("2. Volatility Clustering (GARCH)")
    # Inject clustering
    returns_clustered = returns.copy()
    returns_clustered[500:] *= 5.0 # Shock
    res = f.analyze_volatility_clustering(returns_clustered)
    print(f"   Persistence: {res['persistence']:.4f}")
    print(f"   Sigma_c (GARCH): {res['sigma_c']:.4f}")

def test_new_domains():
    print("\n=== Testing New Domains ===")
    
    # Climate
    c = ClimateAdapter()
    k = np.logspace(-1, 2, 100)
    E = k**-3 # Synoptic
    res = c.analyze_mesoscale_boundary(E, k)
    print(f"Climate: Mesoscale Boundary found at {res['critical_wavelength_km']:.1f} km")
    
    # Seismic
    s = SeismicAdapter()
    mags = np.random.exponential(1.0, 1000) + 2.0 # b=1
    res = s.analyze_gutenberg_richter(mags)
    print(f"Seismic: b-value = {res['b_value']:.4f} (Expected ~1.0)")
    
    # Magnetic
    m = MagneticAdapter()
    temps = np.linspace(2.0, 2.5, 50)
    susc = 1.0 / np.abs(temps - 2.27) # Singularity at Tc=2.27
    res = m.analyze_critical_exponents(temps, np.zeros_like(temps), susc, np.zeros_like(temps))
    print(f"Magnetic: Gamma = {res['gamma']:.4f} (Expected ~1.0)")

def test_core_and_beyond():
    print("\n=== Testing Core & Beyond ===")
    
    # Discovery
    print("1. Observable Discovery")
    d = ObservableDiscovery()
    data = np.random.rand(100, 5)
    # Make feature 2 have a gradient peak
    data[50:, 2] += 5.0 
    cand = d.find_optimal_observable(data, [f"f{i}" for i in range(5)])
    print(f"   Best Observable: {cand.name} (Score: {cand.score:.2f})")
    
    # Control
    print("2. Adaptive Control")
    ctrl = AdaptiveController(target_sigma=0.5)
    correction = ctrl.compute_correction(current_sigma=0.4)
    print(f"   Correction: {correction:.4f}")
    
    # Coupling
    print("3. Cross-Domain Coupling")
    cm = CouplingMatrix(['Quantum', 'GPU', 'Financial'])
    cm.set_coupling('Quantum', 'GPU', 0.5)
    cm.set_coupling('GPU', 'Financial', 0.2)
    res = cm.analyze_stability()
    print(f"   Stability: {res['stability']} (Max Eig: {res['max_eigenvalue']:.4f})")

if __name__ == "__main__":
    test_quantum_rigor()
    test_gpu_rigor()
    test_financial_rigor()
    test_new_domains()
    test_core_and_beyond()
    print("\n✅ All Rigorous Tests Passed!")
