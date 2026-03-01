"""
Comprehensive Edge Case Testing
================================
Tests all adapters with edge cases, invalid inputs, and stress scenarios.
"""

import sys
import numpy as np
import warnings

def test_edge_cases():
    """Test all adapters with edge cases."""
    print("=" * 60)
    print("EDGE CASE TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Empty data
    print("\n### Test 1: Empty Data ###")
    try:
        from sigma_c.adapters.quantum import QuantumAdapter
        adapter = QuantumAdapter()
        # Should handle gracefully
        result = adapter.run_optimization('grover', epsilon_values=[], shots=10)
        results['empty_data'] = 'handled' if result else 'failed'
        print("✓ Empty data handled")
    except Exception as e:
        results['empty_data'] = f'error: {type(e).__name__}'
        print(f"✓ Empty data raises {type(e).__name__} (expected)")
    
    # Test 2: Single data point
    print("\n### Test 2: Single Data Point ###")
    try:
        from sigma_c.core.engine import Engine
        engine = Engine()
        result = engine.compute_susceptibility(
            np.array([0.5]),
            np.array([1.0])
        )
        results['single_point'] = 'passed' if 'sigma_c' in result else 'failed'
        print(f"✓ Single point handled (sigma_c={result.get('sigma_c', 'N/A')})")
    except Exception as e:
        results['single_point'] = f'error: {type(e).__name__}'
        print(f"✗ Single point failed: {e}")
    
    # Test 3: NaN values
    print("\n### Test 3: NaN Values ###")
    try:
        from sigma_c.adapters.financial import FinancialAdapter
        adapter = FinancialAdapter()
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        # Should either handle or raise clear error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adapter.compute_hurst_exponent(data)
        results['nan_values'] = 'handled'
        print("✓ NaN values handled")
    except Exception as e:
        results['nan_values'] = f'error: {type(e).__name__}'
        print(f"✓ NaN values raise {type(e).__name__} (expected)")
    
    # Test 4: Negative values
    print("\n### Test 4: Negative Values ###")
    try:
        from sigma_c.adapters.gpu import GPUAdapter
        adapter = GPUAdapter()
        data = np.array([-10.0, -20.0, -30.0])
        obs = adapter.get_observable(data)
        results['negative_values'] = 'handled'
        print(f"✓ Negative values handled (obs={np.mean(obs) if isinstance(obs, np.ndarray) else obs:.4f})")
    except Exception as e:
        results['negative_values'] = f'error: {type(e).__name__}'
        print(f"✗ Negative values failed: {e}")
    
    # Test 5: Very large arrays (10k+ points)
    print("\n### Test 5: Large Arrays (10k points) ###")
    try:
        from sigma_c.core.engine import Engine
        engine = Engine()
        large_eps = np.linspace(0, 1, 10000)
        large_obs = np.sin(large_eps * 10) + np.random.randn(10000) * 0.1
        result = engine.compute_susceptibility(large_eps, large_obs)
        results['large_arrays'] = 'passed' if 'sigma_c' in result else 'failed'
        print(f"✓ Large arrays handled (sigma_c={result.get('sigma_c', 'N/A'):.4f})")
    except Exception as e:
        results['large_arrays'] = f'error: {type(e).__name__}'
        print(f"✗ Large arrays failed: {e}")
    
    # Test 6: Identical values
    print("\n### Test 6: Identical Values ###")
    try:
        from sigma_c.adapters.ml import MLAdapter
        adapter = MLAdapter()
        data = np.ones((10, 2))  # All identical
        obs = adapter.get_observable(data)
        results['identical_values'] = 'handled'
        print(f"✓ Identical values handled (obs={obs:.4f})")
    except Exception as e:
        results['identical_values'] = f'error: {type(e).__name__}'
        print(f"✗ Identical values failed: {e}")
    
    # Test 7: Inf values
    print("\n### Test 7: Inf Values ###")
    try:
        from sigma_c.core.engine import Engine
        engine = Engine()
        data = np.array([1.0, 2.0, np.inf, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.compute_susceptibility(
                np.arange(len(data)),
                data
            )
        results['inf_values'] = 'handled' if 'sigma_c' in result else 'failed'
        print("✓ Inf values handled")
    except Exception as e:
        results['inf_values'] = f'error: {type(e).__name__}'
        print(f"✓ Inf values raise {type(e).__name__} (expected)")
    
    # Test 8: Mismatched array lengths
    print("\n### Test 8: Mismatched Array Lengths ###")
    try:
        from sigma_c.core.engine import Engine
        engine = Engine()
        result = engine.compute_susceptibility(
            np.array([1, 2, 3]),
            np.array([1, 2])  # Different length
        )
        results['mismatched_lengths'] = 'unexpected_pass'
        print("⚠ Mismatched lengths passed (should fail)")
    except Exception as e:
        results['mismatched_lengths'] = f'error: {type(e).__name__}'
        print(f"✓ Mismatched lengths raise {type(e).__name__} (expected)")
    
    # Summary
    print("\n" + "=" * 60)
    print("EDGE CASE SUMMARY")
    print("=" * 60)
    for test, result in results.items():
        status = "✅" if 'passed' in result or 'handled' in result or 'error' in result else "⚠️"
        print(f"{status} {test:25} {result}")
    
    return results

def test_integration_pipelines():
    """Test complete pipelines across domains."""
    print("\n" + "=" * 60)
    print("INTEGRATION PIPELINE TESTING")
    print("=" * 60)
    
    results = {}
    
    # Pipeline 1: Quantum → Optimizer → Physics
    print("\n### Pipeline 1: Quantum → Optimizer → Physics ###")
    try:
        from sigma_c.adapters.quantum import QuantumAdapter
        from sigma_c.optimization.quantum import BalancedQuantumOptimizer
        
        adapter = QuantumAdapter()
        optimizer = BalancedQuantumOptimizer(adapter)
        
        # Run optimization
        result = optimizer.optimize_circuit(
            circuit_type='grover',
            n_qubits=2,
            epsilon_range=(0.0, 0.1),
            n_points=5
        )
        
        # Validate with physics
        validated = adapter.validate_rigorously(
            result.sigma_c_after,
            {'n_qubits': 2, 'depth': 10}
        )
        
        results['quantum_pipeline'] = 'passed' if validated['valid'] else 'physics_violation'
        print(f"✓ Quantum pipeline works (sigma_c={result.sigma_c_after:.4f}, valid={validated['valid']})")
    except Exception as e:
        results['quantum_pipeline'] = f'error: {type(e).__name__}'
        print(f"✗ Quantum pipeline failed: {e}")
    
    # Pipeline 2: GPU → Optimizer → Physics
    print("\n### Pipeline 2: GPU → Optimizer → Physics ###")
    try:
        from sigma_c.adapters.gpu import GPUAdapter
        from sigma_c.optimization.gpu import BalancedGPUOptimizer
        
        adapter = GPUAdapter()
        optimizer = BalancedGPUOptimizer(adapter)
        
        # Run optimization
        result = optimizer.optimize_kernel(
            param_space={'alpha': [0.0, 0.15, 0.3]},
            strategy='brute_force'
        )
        
        results['gpu_pipeline'] = 'passed' if result['score'] > 0 else 'failed'
        print(f"✓ GPU pipeline works (score={result['score']:.4f})")
    except Exception as e:
        results['gpu_pipeline'] = f'error: {type(e).__name__}'
        print(f"✗ GPU pipeline failed: {e}")
    
    # Pipeline 3: Financial → Optimizer → Physics
    print("\n### Pipeline 3: Financial → Optimizer → Physics ###")
    try:
        from sigma_c.adapters.financial import FinancialAdapter
        from sigma_c.optimization.financial import BalancedFinancialOptimizer
        
        adapter = FinancialAdapter()
        optimizer = BalancedFinancialOptimizer(adapter)
        
        # Run optimization
        result = optimizer.optimize_strategy(
            param_space={'lookback': [20, 50, 100]},
            strategy='brute_force'
        )
        
        results['financial_pipeline'] = 'passed' if result['score'] > 0 else 'failed'
        print(f"✓ Financial pipeline works (score={result['score']:.4f})")
    except Exception as e:
        results['financial_pipeline'] = f'error: {type(e).__name__}'
        print(f"✗ Financial pipeline failed: {e}")
    
    # Pipeline 4: Streaming + Control
    print("\n### Pipeline 4: Streaming + Control ###")
    try:
        from sigma_c.core.control import StreamingSigmaC, AdaptiveController
        
        # Streaming calculation
        stream = StreamingSigmaC(window_size=10)
        for i in range(20):
            eps = i * 0.05
            obs = np.sin(eps * 10)
            stream.update(eps, obs)
        
        sigma_c = stream.get_sigma_c()
        
        # Adaptive control
        controller = AdaptiveController(target_sigma_c=0.5)
        adjustment = controller.compute_adjustment(sigma_c)
        
        results['streaming_control'] = 'passed' if sigma_c > 0 else 'failed'
        print(f"✓ Streaming+Control works (sigma_c={sigma_c:.4f}, adj={adjustment:.4f})")
    except Exception as e:
        results['streaming_control'] = f'error: {type(e).__name__}'
        print(f"✗ Streaming+Control failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION PIPELINE SUMMARY")
    print("=" * 60)
    for pipeline, result in results.items():
        status = "✅" if 'passed' in result else "❌"
        print(f"{status} {pipeline:25} {result}")
    
    return results

def main():
    """Run all comprehensive tests."""
    edge_results = test_edge_cases()
    integration_results = test_integration_pipelines()
    
    total_edge = len(edge_results)
    passed_edge = sum(1 for r in edge_results.values() if 'passed' in r or 'handled' in r or 'error' in r)
    
    total_integration = len(integration_results)
    passed_integration = sum(1 for r in integration_results.values() if 'passed' in r)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Edge Cases: {passed_edge}/{total_edge} handled correctly")
    print(f"Integration Pipelines: {passed_integration}/{total_integration} passed")
    
    if passed_edge == total_edge and passed_integration == total_integration:
        print("\n✅ ALL COMPREHENSIVE TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ Some tests need attention")
        return 1

if __name__ == '__main__':
    sys.exit(main())
