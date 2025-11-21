#!/usr/bin/env python3
"""Quick test of all 6 adapters."""

import sys
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from sigma_c import Universe

print("Testing v1.1.0 Diagnostics System")
print("="*50)

passed = []
failed = []

# Test 1: GPU
try:
    print("\n1. GPU...", end=' ')
    gpu = Universe.gpu()
    diag = gpu.diagnose()
    val = gpu.validate_techniques()
    eps = np.linspace(0, 1, 10)
    obs = np.exp(-5*eps)
    res = gpu.compute_susceptibility(eps, obs)
    exp = gpu.explain(res)
    assert 'status' in diag and 'all_passed' in val
    print("PASS")
    passed.append('GPU')
except Exception as e:
    print(f"FAIL: {e}")
    failed.append('GPU')

# Test 2: Quantum
try:
    print("2. Quantum...", end=' ')
    from sigma_c.adapters.quantum import Circuit
    quantum = Universe.quantum()
    circuit = Circuit()
    circuit.h(0)
    diag = quantum.diagnose(circuit)
    val = quantum.validate_techniques(circuit)
    eps = np.linspace(0, 1, 10)
    obs = np.exp(-5*eps)
    res = quantum.compute_susceptibility(eps, obs)
    exp = quantum.explain(res)
    assert 'status' in diag and 'all_passed' in val
    print("PASS")
    passed.append('Quantum')
except Exception as e:
    print(f"FAIL: {e}")
    failed.append('Quantum')

# Test 3: Financial
try:
    print("3. Financial...", end=' ')
    finance = Universe.finance()
    dates = pd.date_range('2024-01-01', periods=100)
    prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)
    diag = finance.diagnose(prices)
    val = finance.validate_techniques(prices)
    eps = np.linspace(0, 1, 10)
    obs = np.exp(-5*eps)
    res = finance.compute_susceptibility(eps, obs)
    exp = finance.explain(res)
    assert 'status' in diag and 'all_passed' in val
    print("PASS")
    passed.append('Financial')
except Exception as e:
    print(f"FAIL: {e}")
    failed.append('Financial')

# Test 4: Climate
try:
    print("4. Climate...", end=' ')
    climate = Universe.climate()
    data = np.random.randn(50, 3)
    diag = climate.diagnose(data)
    val = climate.validate_techniques(data)
    eps = np.linspace(0, 1, 10)
    obs = np.exp(-5*eps)
    res = climate.compute_susceptibility(eps, obs)
    exp = climate.explain(res)
    assert 'status' in diag and 'all_passed' in val
    print("PASS")
    passed.append('Climate')
except Exception as e:
    print(f"FAIL: {e}")
    failed.append('Climate')

# Test 5: Seismic
try:
    print("5. Seismic...", end=' ')
    seismic = Universe.seismic()
    catalog = pd.DataFrame({
        'magnitude': np.random.uniform(2, 7, 100),
        'depth': np.random.uniform(0, 100, 100)
    })
    diag = seismic.diagnose(catalog)
    val = seismic.validate_techniques(catalog)
    eps = np.linspace(0, 1, 10)
    obs = np.exp(-5*eps)
    res = seismic.compute_susceptibility(eps, obs)
    exp = seismic.explain(res)
    assert 'status' in diag and 'all_passed' in val
    print("PASS")
    passed.append('Seismic')
except Exception as e:
    print(f"FAIL: {e}")
    failed.append('Seismic')

# Test 6: Magnetic
try:
    print("6. Magnetic...", end=' ')
    magnetic = Universe.magnetic()
    diag = magnetic.diagnose()
    val = magnetic.validate_techniques()
    eps = np.linspace(0, 1, 10)
    obs = np.exp(-5*eps)
    res = magnetic.compute_susceptibility(eps, obs)
    exp = magnetic.explain(res)
    assert 'status' in diag and 'all_passed' in val
    print("PASS")
    passed.append('Magnetic')
except Exception as e:
    print(f"FAIL: {e}")
    failed.append('Magnetic')

# Summary
print("\n" + "="*50)
print(f"PASSED: {len(passed)}/6")
print(f"FAILED: {len(failed)}/6")

if len(passed) == 6:
    print("\nALL ADAPTERS READY FOR v1.1.0!")
    sys.exit(0)
else:
    print(f"\nFailed adapters: {', '.join(failed)}")
    sys.exit(1)
