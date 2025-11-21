#!/usr/bin/env python3
"""
Final v1.1.0 Verification Test
===============================
Complete verification of diagnostics implementation.
"""

from sigma_c import Universe
import numpy as np
import pandas as pd

print("""
╔══════════════════════════════════════════════════════════════╗
║         Sigma-C Framework v1.1.0 - Final Verification       ║
╚══════════════════════════════════════════════════════════════╝
""")

# Check 1: All methods exist
print("CHECK 1: API Consistency")
print("="*60)

universe = Universe()
adapters = [
    ('Quantum', universe.quantum()),
    ('GPU', universe.gpu()),
    ('Finance', universe.finance()),
    ('Climate', universe.climate()),
    ('Seismic', universe.seismic()),
    ('Magnetic', universe.magnetic())
]

required_methods = ['diagnose', 'auto_search', 'validate_techniques', 'explain']
api_ok = True

for name, adapter in adapters:
    print(f"\n{name}:")
    for method in required_methods:
        has = hasattr(adapter, method)
        print(f"  {'✅' if has else '❌'} {method}()")
        if not has:
            api_ok = False

if not api_ok:
    print("\n❌ API INCOMPLETE!")
    exit(1)

print("\n✅ All adapters have complete API!")

# Check 2: Functional tests
print("\n\nCHECK 2: Functional Tests")
print("="*60)

try:
    # Quantum
    print("\n1. Quantum Adapter:")
    q = universe.quantum()
    circuit = q.create_grover_with_noise(n_qubits=2, epsilon=0.01)
    diag = q.diagnose(circuit)
    print(f"   diagnose(): {diag['status']}")
    val = q.validate_techniques(circuit)
    print(f"   validate_techniques(): {len(val)} checks")
    
    # GPU  
    print("\n2. GPU Adapter:")
    g = universe.gpu()
    diag = g.diagnose(benchmark_data=None)  # Pass None explicitly
    print(f"   diagnose(): {diag['status']}")
    val = g.validate_techniques()
    print(f"   validate_techniques(): {len(val)} checks")
    
    # Finance
    print("\n3. Finance Adapter:")
    f = universe.finance()
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.02)))
    diag = f.diagnose(price_data=prices)
    print(f"   diagnose(): {diag['status']}")
    val = f.validate_techniques(price_data=prices)
    print(f"   validate_techniques(): {len(val)} checks")
    
    # Climate
    print("\n4. Climate Adapter:")
    c = universe.climate()
    grid = np.random.randn(64, 64)
    diag = c.diagnose(data=grid)
    print(f"   diagnose(): {diag['status']}")
    val = c.validate_techniques(data=grid)
    print(f"   validate_techniques(): {len(val)} checks")
    
    # Seismic
    print("\n5. Seismic Adapter:")
    s = universe.seismic()
    catalog = pd.DataFrame({
        'latitude': np.random.uniform(30, 40, 200),
        'longitude': np.random.uniform(-120, -110, 200),
        'magnitude': np.random.exponential(2.0, 200) + 2.0
    })
    diag = s.diagnose(catalog=catalog)
    print(f"   diagnose(): {diag['status']}")
    val = s.validate_techniques(catalog=catalog)
    print(f"   validate_techniques(): {len(val)} checks")
    
    # Magnetic
    print("\n6. Magnetic Adapter:")
    m = universe.magnetic()
    diag = m.diagnose(lattice_size=32)
    print(f"   diagnose(): {diag['status']}")
    val = m.validate_techniques(lattice_size=32)
    print(f"   validate_techniques(): {len(val)} checks")
    
    print("\n✅ All functional tests passed!")
    
except Exception as e:
    print(f"\n❌ Functional test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "="*60)
print("🎉 v1.1.0 VERIFICATION COMPLETE!")
print("="*60)
print("\n✅ All 6 adapters have complete diagnostics")
print("✅ All methods are functional")
print("\n📦 Ready for release!")
