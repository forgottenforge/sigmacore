#!/usr/bin/env python3
"""
Comprehensive test of all 6 domain adapters for v1.1.0 diagnostics.
"""

import numpy as np
import pandas as pd
from sigma_c import Universe

def test_adapter(name, adapter, test_data=None):
    """Test all 4 diagnostic methods for an adapter."""
    print(f"\n{'='*60}")
    print(f"Testing {name} Adapter")
    print('='*60)
    
    try:
        # Test 1: diagnose()
        print(f"  1. diagnose()...", end=' ')
        if test_data is not None:
            diag = adapter.diagnose(test_data)
        else:
            diag = adapter.diagnose()
        assert 'status' in diag
        assert 'issues' in diag
        assert 'recommendations' in diag
        print(f"✓ (status: {diag['status']})")
        
        # Test 2: validate_techniques()
        print(f"  2. validate_techniques()...", end=' ')
        validation = adapter.validate_techniques()
        assert 'all_passed' in validation
        assert 'tests' in validation
        print(f"✓ (passed: {validation['all_passed']})")
        
        # Test 3: compute_susceptibility()
        print(f"  3. compute_susceptibility()...", end=' ')
        epsilon = np.linspace(0, 1, 10)
        observable = np.exp(-5 * epsilon)
        result = adapter.compute_susceptibility(epsilon, observable)
        assert 'sigma_c' in result
        assert 'kappa' in result
        print(f"✓ (σ_c={result['sigma_c']:.3f}, κ={result['kappa']:.2f})")
        
        # Test 4: explain()
        print(f"  4. explain()...", end=' ')
        explanation = adapter.explain(result)
        assert isinstance(explanation, str)
        assert len(explanation) > 50
        print(f"✓ ({len(explanation)} chars)")
        
        print(f"\n  ✅ {name} adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n  ❌ {name} adapter FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("""
==========================================================================
   Sigma-C Framework v1.1.0 - Complete Diagnostics Verification
   
   Testing all 6 domain adapters
==========================================================================
    """)
    
    results = {}
    
    # 1. GPU Adapter
    gpu = Universe.gpu()
    results['GPU'] = test_adapter('GPU', gpu)
    
    # 2. Quantum Adapter
    from sigma_c.adapters.quantum import Circuit
    quantum = Universe.quantum()
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    results['Quantum'] = test_adapter('Quantum', quantum, circuit)
    
    # 3. Financial Adapter
    finance = Universe.finance()
    # Create sample price data
    dates = pd.date_range('2024-01-01', periods=100)
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    results['Financial'] = test_adapter('Financial', finance, prices)
    
    # 4. Climate Adapter
    climate = Universe.climate()
    # Create sample climate data (lat, lon, temp)
    climate_data = np.random.randn(50, 3)
    climate_data[:, 0] = np.random.uniform(-90, 90, 50)  # lat
    climate_data[:, 1] = np.random.uniform(-180, 180, 50)  # lon
    climate_data[:, 2] = np.random.uniform(250, 310, 50)  # temp
    results['Climate'] = test_adapter('Climate', climate, climate_data)
    
    # 5. Seismic Adapter
    seismic = Universe.seismic()
    # Create sample earthquake catalog
    catalog = pd.DataFrame({
        'magnitude': np.random.uniform(2, 7, 100),
        'depth': np.random.uniform(0, 100, 100),
        'latitude': np.random.uniform(-90, 90, 100),
        'longitude': np.random.uniform(-180, 180, 100)
    })
    results['Seismic'] = test_adapter('Seismic', seismic, catalog)
    
    # 6. Magnetic Adapter
    magnetic = Universe.magnetic()
    results['Magnetic'] = test_adapter('Magnetic', magnetic)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for domain, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {domain:12s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL 6 ADAPTERS PASSED!")
        print("v1.1.0 Universal Diagnostics System is COMPLETE")
    else:
        print("⚠️  Some adapters failed - review errors above")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())
