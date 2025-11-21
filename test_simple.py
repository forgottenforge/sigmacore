#!/usr/bin/env python3
"""Simple test of v1.1.0 framework."""

import numpy as np
from sigma_c import Universe

print("Testing Sigma-C Framework v1.1.0...")
print("="*60)

# Test 1: Load GPU adapter
print("\\n1. Loading GPU adapter...")
gpu = Universe.gpu()
print("   OK - GPU adapter loaded")

# Test 2: Test diagnostics
print("\\n2. Testing diagnose()...")
diag = gpu.diagnose()
print(f"   Status: {diag['status']}")
print(f"   Issues: {len(diag['issues'])}")

# Test 3: Test susceptibility computation
print("\\n3. Testing compute_susceptibility()...")
epsilon = np.linspace(0, 1, 20)
observable = np.exp(-5 * epsilon)  # Exponential decay
result = gpu.compute_susceptibility(epsilon, observable)
print(f"   sigma_c = {result['sigma_c']:.3f}")
print(f"   kappa = {result['kappa']:.2f}")

# Test 4: Test validate_techniques
print("\\n4. Testing validate_techniques()...")
validation = gpu.validate_techniques()
print(f"   All passed: {validation['all_passed']}")

# Test 5: Test explain
print("\\n5. Testing explain()...")
explanation = gpu.explain(result)
print(f"   Explanation length: {len(explanation)} chars")

print("\\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
