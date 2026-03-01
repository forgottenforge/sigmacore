#!/usr/bin/env python3
"""Test validate_techniques for GPU and Magnetic."""

from sigma_c import Universe

print("Testing validate_techniques()...")
print("="*50)

# Test GPU
print("\n1. GPU validate_techniques():")
try:
    gpu = Universe.gpu()
    result = gpu.validate_techniques()
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test Magnetic
print("\n2. Magnetic validate_techniques():")
try:
    magnetic = Universe.magnetic()
    result = magnetic.validate_techniques()
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
