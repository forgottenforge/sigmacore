"""
Comprehensive Adapter Functionality Test
=========================================
Tests that ALL adapters actually work, not just exist.
"""

import sys
import numpy as np

def test_quantum_adapter():
    """Test QuantumAdapter with real circuit creation."""
    print("\n=== Testing Quantum Adapter ===")
    try:
        from sigma_c.adapters.quantum import QuantumAdapter
        
        adapter = QuantumAdapter()
        
        # Test 1: Create circuit
        circuit = adapter.create_grover_with_noise(n_qubits=2, epsilon=0.05)
        assert circuit is not None, "Circuit creation failed"
        print("✓ Circuit creation works")
        
        # Test 2: Run optimization
        result = adapter.run_optimization(
            circuit_type='grover',
            epsilon_values=[0.0, 0.05, 0.1],
            shots=10
        )
        assert 'sigma_c' in result, "Optimization failed"
        print(f"✓ Optimization works (sigma_c={result['sigma_c']:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Quantum Adapter FAILED: {e}")
        return False

def test_gpu_adapter():
    """Test GPUAdapter with real metrics."""
    print("\n=== Testing GPU Adapter ===")
    try:
        from sigma_c.adapters.gpu import GPUAdapter
        
        adapter = GPUAdapter()
        
        # Test 1: Get observable
        data = np.array([100.0, 90.0, 80.0, 70.0])  # GFLOPS values
        obs_array = adapter.get_observable(data)
        # get_observable returns an array, take mean
        obs = float(np.mean(obs_array)) if isinstance(obs_array, np.ndarray) else float(obs_array)
        assert obs >= 0, "Observable calculation failed"
        print(f"✓ Observable calculation works (obs={obs:.4f})")
        
        # Test 2: Roofline analysis
        roofline = adapter.analyze_roofline()
        assert 'ridge_point' in roofline, "Roofline analysis failed"
        print(f"✓ Roofline analysis works (ridge={roofline['ridge_point']:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ GPU Adapter FAILED: {e}")
        return False

def test_financial_adapter():
    """Test FinancialAdapter with real market data."""
    print("\n=== Testing Financial Adapter ===")
    try:
        from sigma_c.adapters.financial import FinancialAdapter
        
        adapter = FinancialAdapter()
        
        # Test 1: Hurst exponent
        prices = np.cumsum(np.random.randn(100)) + 100
        hurst_result = adapter.compute_hurst_exponent(prices)
        hurst = hurst_result['hurst']  # Returns dict, not float
        assert 0 <= hurst <= 1, "Hurst exponent out of range"
        print(f"✓ Hurst exponent works (H={hurst:.4f})")
        
        # Test 2: GARCH analysis
        returns = np.diff(prices) / prices[:-1]
        garch = adapter.analyze_volatility_clustering(returns)
        assert 'persistence' in garch, "GARCH analysis failed"
        print(f"✓ GARCH analysis works (persistence={garch['persistence']:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Financial Adapter FAILED: {e}")
        return False

def test_climate_adapter():
    """Test ClimateAdapter."""
    print("\n=== Testing Climate Adapter ===")
    try:
        from sigma_c.adapters.climate import ClimateAdapter
        
        adapter = ClimateAdapter()
        
        # Test: Mesoscale boundary detection
        wavenumbers = np.logspace(-3, -1, 50)  # 1/km
        energy_spectrum = wavenumbers ** (-3) + 0.1 * np.random.randn(50)
        result = adapter.analyze_mesoscale_boundary(energy_spectrum, wavenumbers)
        assert 'sigma_c' in result or 'boundary_sigma_c' in result, "Boundary detection failed"
        sigma_c_key = 'sigma_c' if 'sigma_c' in result else 'boundary_sigma_c'
        print(f"✓ Mesoscale boundary detection works (sigma_c={result[sigma_c_key]:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Climate Adapter FAILED: {e}")
        return False

def test_seismic_adapter():
    """Test SeismicAdapter."""
    print("\n=== Testing Seismic Adapter ===")
    try:
        from sigma_c.adapters.seismic import SeismicAdapter
        
        adapter = SeismicAdapter()
        
        # Test: Gutenberg-Richter analysis
        magnitudes = np.random.exponential(2.0, 100) + 1.0
        result = adapter.analyze_gutenberg_richter(magnitudes)
        assert 'b_value' in result, "G-R analysis failed"
        print(f"✓ Gutenberg-Richter works (b={result['b_value']:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Seismic Adapter FAILED: {e}")
        return False

def test_magnetic_adapter():
    """Test MagneticAdapter."""
    print("\n=== Testing Magnetic Adapter ===")
    try:
        from sigma_c.adapters.magnetic import MagneticAdapter
        
        adapter = MagneticAdapter()
        
        # Test: Critical exponents
        temps = np.linspace(0.8, 1.2, 20)
        magnetization = np.abs(1 - temps) ** 0.3
        # Create susceptibility and specific heat
        susceptibility = 1.0 / (np.abs(temps - 1.0) + 0.01) ** 1.2
        specific_heat = 1.0 / (np.abs(temps - 1.0) + 0.01) ** 0.1
        
        result = adapter.analyze_critical_exponents(temps, magnetization, susceptibility, specific_heat)
        assert 'gamma' in result, "Critical exponent analysis failed"
        print(f"✓ Critical exponents work (gamma={result['gamma']:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Magnetic Adapter FAILED: {e}")
        return False

def test_edge_adapter():
    """Test EdgeAdapter."""
    print("\n=== Testing Edge Adapter ===")
    try:
        from sigma_c.adapters.edge import EdgeAdapter
        
        adapter = EdgeAdapter()
        
        # Test: Power efficiency analysis
        freq = np.linspace(0.5, 2.0, 20)
        power = freq ** 2
        perf = freq * 0.8
        result = adapter.analyze_power_efficiency(freq, power, perf)
        assert 'critical_frequency' in result, "Power efficiency analysis failed"
        print(f"✓ Power efficiency works (f_crit={result['critical_frequency']:.2f} GHz)")
        
        return True
    except Exception as e:
        print(f"✗ Edge Adapter FAILED: {e}")
        return False

def test_llm_cost_adapter():
    """Test LLMCostAdapter."""
    print("\n=== Testing LLM Cost Adapter ===")
    try:
        from sigma_c.adapters.llm_cost import LLMCostAdapter
        
        adapter = LLMCostAdapter()
        
        # Test: Cost-safety analysis
        models = [
            {'name': 'Basic', 'cost': 0.001, 'hallucination_rate': 0.15},
            {'name': 'Pro', 'cost': 0.01, 'hallucination_rate': 0.05},
            {'name': 'Ultra', 'cost': 0.1, 'hallucination_rate': 0.01}
        ]
        result = adapter.analyze_cost_safety(models)
        assert 'best_model' in result, "Cost-safety analysis failed"
        print(f"✓ Cost-safety analysis works (best={result['best_model']})")
        
        return True
    except Exception as e:
        print(f"✗ LLM Cost Adapter FAILED: {e}")
        return False

def test_ml_adapter():
    """Test MLAdapter."""
    print("\n=== Testing ML Adapter ===")
    try:
        from sigma_c.adapters.ml import MLAdapter
        
        adapter = MLAdapter()
        
        # Test: Get observable
        data = np.random.rand(10, 2)  # [accuracy, robustness]
        obs = adapter.get_observable(data)
        assert obs > 0, "Observable calculation failed"
        print(f"✓ ML observable works (obs={obs:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ ML Adapter FAILED: {e}")
        return False

def test_connectors():
    """Test key connectors."""
    print("\n=== Testing Connectors ===")
    
    results = {}
    
    # Test Universal Bridge
    try:
        from sigma_c.connectors.bridge import SigmaCBridge
        
        @SigmaCBridge.wrap_any_function
        def test_func(x):
            return x ** 2
        
        result = test_func(5)
        assert result == 25, "Function wrapping failed"
        print("✓ Universal Bridge works")
        results['bridge'] = True
    except Exception as e:
        print(f"✗ Universal Bridge FAILED: {e}")
        results['bridge'] = False
    
    # Test PyTorch integration
    try:
        from sigma_c.ml.pytorch import CriticalModule, SigmaCLoss
        print("✓ PyTorch integration imports successfully")
        results['pytorch'] = True
    except Exception as e:
        print(f"✗ PyTorch integration FAILED: {e}")
        results['pytorch'] = False
    
    # Test REST API
    try:
        from sigma_c.api.rest import SigmaCAPI
        api = SigmaCAPI()
        result = api.compute([0.0, 0.1, 0.2], [1.0, 0.8, 0.5])
        assert 'sigma_c' in result, "API computation failed"
        print(f"✓ REST API works (sigma_c={result['sigma_c']:.4f})")
        results['api'] = True
    except Exception as e:
        print(f"✗ REST API FAILED: {e}")
        results['api'] = False
    
    return results

def main():
    """Run all adapter tests."""
    print("=" * 60)
    print("COMPREHENSIVE ADAPTER FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = {
        'Quantum': test_quantum_adapter(),
        'GPU': test_gpu_adapter(),
        'Financial': test_financial_adapter(),
        'Climate': test_climate_adapter(),
        'Seismic': test_seismic_adapter(),
        'Magnetic': test_magnetic_adapter(),
        'Edge': test_edge_adapter(),
        'LLM Cost': test_llm_cost_adapter(),
        'ML': test_ml_adapter()
    }
    
    connector_results = test_connectors()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nDomain Adapters:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:15} {status}")
    
    print("\nConnectors:")
    for name, passed in connector_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:15} {status}")
    
    total_tests = len(results) + len(connector_results)
    passed_tests = sum(results.values()) + sum(connector_results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✅ ALL ADAPTERS FUNCTIONAL!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} ADAPTERS FAILED!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
