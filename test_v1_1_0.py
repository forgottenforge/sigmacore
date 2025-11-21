#!/usr/bin/env python3
"""
Comprehensive v1.1.0 Verification Test
======================================
Tests all diagnostics features across all 6 domains.
"""

import sys
import traceback
from sigma_c import Universe
import numpy as np
import pandas as pd

def test_quantum_diagnostics():
    """Test Quantum adapter diagnostics."""
    print("\n" + "="*60)
    print("TESTING: Quantum Adapter Diagnostics")
    print("="*60)
    
    try:
        universe = Universe()
        
        # Test 1: diagnose() exists and works
        print("✓ Testing diagnose() method...")
        circuit = universe.quantum.create_grover_with_noise(n_qubits=2, epsilon=0.01)
        diag = universe.quantum.diagnose(circuit)
        assert 'status' in diag, "Missing 'status' in diagnose result"
        assert 'issues' in diag, "Missing 'issues' in diagnose result"
        assert 'recommendations' in diag, "Missing 'recommendations' in diagnose result"
        print(f"  Status: {diag['status']}")
        print(f"  Issues: {len(diag['issues'])}")
        
        # Test 2: auto_search() exists and works
        print("✓ Testing auto_search() method...")
        search = universe.quantum.auto_search(circuit_type='grover', n_qubits=2, n_shots=20)
        assert 'best_params' in search, "Missing 'best_params' in search result"
        assert 'recommendation' in search, "Missing 'recommendation' in search result"
        print(f"  Best params: {search['best_params']}")
        
        # Test 3: validate_techniques() exists and works
        print("✓ Testing validate_techniques() method...")
        validation = universe.quantum.validate_techniques(circuit)
        assert isinstance(validation, dict), "validate_techniques should return dict"
        print(f"  Validation checks: {len(validation)}")
        
        # Test 4: explain() exists and works
        print("✓ Testing explain() method...")
        result = universe.quantum.run_optimization(circuit_type='grover', n_qubits=2, shots=30)
        explanation = universe.quantum.explain(result)
        assert isinstance(explanation, str), "explain should return string"
        assert len(explanation) > 0, "explain should return non-empty string"
        print(f"  Explanation length: {len(explanation)} chars")
        
        print("✅ Quantum adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Quantum adapter FAILED: {e}")
        traceback.print_exc()
        return False

def test_gpu_diagnostics():
    """Test GPU adapter diagnostics."""
    print("\n" + "="*60)
    print("TESTING: GPU Adapter Diagnostics")
    print("="*60)
    
    try:
        universe = Universe()
        
        print("✓ Testing diagnose() method...")
        diag = universe.gpu.diagnose()
        assert 'status' in diag
        print(f"  Status: {diag['status']}")
        
        print("✓ Testing auto_search() method...")
        search = universe.gpu.auto_search(epsilon_points=10)
        assert 'best_params' in search
        print(f"  Best params: {search['best_params']}")
        
        print("✓ Testing validate_techniques() method...")
        validation = universe.gpu.validate_techniques()
        assert isinstance(validation, dict)
        print(f"  Validation: {validation}")
        
        print("✅ GPU adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ GPU adapter FAILED: {e}")
        traceback.print_exc()
        return False

def test_financial_diagnostics():
    """Test Financial adapter diagnostics."""
    print("\n" + "="*60)
    print("TESTING: Financial Adapter Diagnostics")
    print("="*60)
    
    try:
        universe = Universe()
        
        # Create synthetic data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))
        price_data = pd.Series(prices)
        
        print("✓ Testing diagnose() method...")
        diag = universe.finance().diagnose(price_data=price_data)
        assert 'status' in diag
        print(f"  Status: {diag['status']}")
        print(f"  Issues: {len(diag['issues'])}")
        
        print("✓ Testing validate_techniques() method...")
        validation = universe.finance().validate_techniques(price_data=price_data)
        assert isinstance(validation, dict)
        print(f"  Validation: {validation}")
        
        print("✅ Financial adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Financial adapter FAILED: {e}")
        traceback.print_exc()
        return False

def test_climate_diagnostics():
    """Test Climate adapter diagnostics."""
    print("\n" + "="*60)
    print("TESTING: Climate Adapter Diagnostics")
    print("="*60)
    
    try:
        universe = Universe()
        
        # Create synthetic grid
        grid = np.random.randn(64, 64)
        
        print("✓ Testing diagnose() method...")
        diag = universe.climate().diagnose(data=grid)
        assert 'status' in diag
        print(f"  Status: {diag['status']}")
        
        print("✓ Testing validate_techniques() method...")
        validation = universe.climate().validate_techniques(data=grid)
        assert isinstance(validation, dict)
        print(f"  Validation: {validation}")
        
        print("✅ Climate adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Climate adapter FAILED: {e}")
        traceback.print_exc()
        return False

def test_seismic_diagnostics():
    """Test Seismic adapter diagnostics."""
    print("\n" + "="*60)
    print("TESTING: Seismic Adapter Diagnostics")
    print("="*60)
    
    try:
        universe = Universe()
        
        # Create synthetic catalog
        np.random.seed(42)
        catalog = pd.DataFrame({
            'latitude': np.random.uniform(30, 40, 200),
            'longitude': np.random.uniform(-120, -110, 200),
            'magnitude': np.random.exponential(2.0, 200) + 2.0,
            'depth': np.random.uniform(0, 50, 200)
        })
        
        print("✓ Testing diagnose() method...")
        diag = universe.seismic().diagnose(catalog=catalog)
        assert 'status' in diag
        print(f"  Status: {diag['status']}")
        
        print("✓ Testing validate_techniques() method...")
        validation = universe.seismic().validate_techniques(catalog=catalog)
        assert isinstance(validation, dict)
        print(f"  Validation: {validation}")
        
        print("✅ Seismic adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Seismic adapter FAILED: {e}")
        traceback.print_exc()
        return False

def test_magnetic_diagnostics():
    """Test Magnetic adapter diagnostics."""
    print("\n" + "="*60)
    print("TESTING: Magnetic Adapter Diagnostics")
    print("="*60)
    
    try:
        universe = Universe()
        
        print("✓ Testing diagnose() method...")
        diag = universe.magnetic().diagnose(lattice_size=32)
        assert 'status' in diag
        print(f"  Status: {diag['status']}")
        
        print("✓ Testing validate_techniques() method...")
        validation = universe.magnetic().validate_techniques(lattice_size=32)
        assert isinstance(validation, dict)
        print(f"  Validation: {validation}")
        
        print("✅ Magnetic adapter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Magnetic adapter FAILED: {e}")
        traceback.print_exc()
        return False

def test_base_api():
    """Test that base API methods exist on all adapters."""
    print("\n" + "="*60)
    print("TESTING: Base API Consistency")
    print("="*60)
    
    universe = Universe()
    adapters = {
        'quantum': universe.quantum(),
        'gpu': universe.gpu(),
        'finance': universe.finance(),
        'climate': universe.climate(),
        'seismic': universe.seismic(),
        'magnetic': universe.magnetic()
    }
    
    required_methods = ['diagnose', 'auto_search', 'validate_techniques', 'explain']
    
    all_passed = True
    for name, adapter in adapters.items():
        print(f"\n✓ Checking {name} adapter...")
        for method in required_methods:
            if not hasattr(adapter, method):
                print(f"  ❌ Missing method: {method}")
                all_passed = False
            else:
                print(f"  ✓ Has method: {method}")
    
    if all_passed:
        print("\n✅ Base API: ALL ADAPTERS CONSISTENT")
    else:
        print("\n❌ Base API: INCONSISTENCIES FOUND")
    
    return all_passed

def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         Sigma-C Framework v1.1.0                            ║
║         Comprehensive Verification Test Suite               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        'Base API': test_base_api(),
        'Quantum': test_quantum_diagnostics(),
        'GPU': test_gpu_diagnostics(),
        'Financial': test_financial_diagnostics(),
        'Climate': test_climate_diagnostics(),
        'Seismic': test_seismic_diagnostics(),
        'Magnetic': test_magnetic_diagnostics()
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for domain, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{domain:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - v1.1.0 READY FOR RELEASE!")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
