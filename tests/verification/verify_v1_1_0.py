#!/usr/bin/env python3
"""
Simple v1.1.0 Diagnostics Verification
======================================
Verifies all 6 adapters have diagnostics methods.
"""

from sigma_c import Universe
import numpy as np
import pandas as pd

print("""
╔══════════════════════════════════════════════════════════════╗
║         Sigma-C Framework v1.1.0 - Quick Verification       ║
╚══════════════════════════════════════════════════════════════╝
""")

# Test 1: Check all adapters have required methods
print("TEST 1: Checking API Consistency")
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
all_ok = True

for name, adapter in adapters.items():
    print(f"\n{name.upper()} Adapter:")
    for method in required_methods:
        has_method = hasattr(adapter, method)
        status = "✅" if has_method else "❌"
        print(f"  {status} {method}()")
        if not has_method:
            all_ok = False

if all_ok:
    print("\n✅ All adapters have required methods!")
else:
    print("\n❌ Some methods missing!")
    exit(1)

# Test 2: Quick functional test
print("\n\nTEST 2: Quick Functional Test")
print("="*60)

try:
    # Quantum
    print("\n✓ Quantum: diagnose()")
    circuit = universe.quantum().create_grover_with_noise(n_qubits=2, epsilon=0.01)
    diag = universe.quantum().diagnose(circuit)
    print(f"  Status: {diag['status']}")
    
    # GPU
    print("\n✓ GPU: diagnose()")
    diag = universe.gpu().diagnose()
    print(f"  Status: {diag['status']}")
    
    # Finance
    print("\n✓ Finance: diagnose()")
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.02)))
    diag = universe.finance().diagnose(price_data=prices)
    print(f"  Status: {diag['status']}")
    
    # Climate
    print("\n✓ Climate: diagnose()")
    grid = np.random.randn(64, 64)
    diag = universe.climate().diagnose(data=grid)
    print(f"  Status: {diag['status']}")
    
    # Seismic
    print("\n✓ Seismic: diagnose()")
    catalog = pd.DataFrame({
        'latitude': np.random.uniform(30, 40, 200),
        'longitude': np.random.uniform(-120, -110, 200),
        'magnitude': np.random.exponential(2.0, 200) + 2.0
    })
    diag = universe.seismic().diagnose(catalog=catalog)
    print(f"  Status: {diag['status']}")
    
    # Magnetic
    print("\n✓ Magnetic: diagnose()")
    diag = universe.magnetic().diagnose(lattice_size=32)
    print(f"  Status: {diag['status']}")
    
    print("\n✅ All functional tests passed!")
    
except Exception as e:
    print(f"\n❌ Functional test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("🎉 v1.1.0 VERIFICATION COMPLETE - ALL TESTS PASSED!")
print("="*60)
