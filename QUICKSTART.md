# Sigma-C Framework - Quick Start Guide

**Version 1.2.0** | **Copyright (c) 2025 ForgottenForge.xyz**

## For Developers: Getting Started in 5 Minutes

### 1. Installation

```bash
# Clone and install
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework
pip install .
```

### 2. Your First Analysis

```python
from sigma_c import Universe

# Example: Detect GPU performance critical point
gpu = Universe.gpu()
result = gpu.auto_tune(alpha_levels=[0.1, 0.3, 0.5, 0.7, 0.9])

print(f"Critical threshold: {result['sigma_c']:.3f}")
print(f"Stability score: {result['statistics']['kappa']:.2f}")
```

### 3. Available Domains

| Domain | Use Case | Method |
|--------|----------|--------|
| **Quantum** | Noise resilience in quantum circuits | `Universe.quantum()` |
| **GPU** | Kernel optimization, cache tuning | `Universe.gpu()` |
| **Financial** | Market regime detection, crash prediction | `Universe.finance()` |
| **Climate** | Spatial scaling in weather data | `Universe.climate()` |
| **Seismic** | Earthquake criticality analysis | `Universe.seismic()` |
| **Magnetic** | Phase transitions (Ising model) | `Universe.magnetic()` |

### 4. Run the Examples

```bash
# All examples are in examples_v4/
python examples_v4/demo_quantum.py
python examples_v4/demo_gpu.py
python examples_v4/demo_finance.py
python examples_v4/demo_climate.py
python examples_v4/demo_seismic.py
python examples_v4/demo_magnetic.py
```

Each demo generates a plot showing the critical point detection.

### 5. Core Concept: Critical Susceptibility

**What it does:** Finds the exact parameter value (σ_c) where a system transitions from stable to unstable.

**Why it's better than traditional metrics:**
- Detects **precursors** to failure, not just the failure itself
- Works across completely different domains (quantum, finance, physics)
- Statistically robust (bootstrap + permutation tests)

**Key outputs:**
- `sigma_c`: The critical scale/threshold
- `kappa`: Stability score (higher = more critical)
- `p_value`: Statistical significance

### 6. Typical Workflow

```python
from sigma_c import Universe
import numpy as np

# 1. Initialize adapter
adapter = Universe.gpu()

# 2. Prepare your data
# x = control parameter (e.g., noise level, cache size)
# y = observable (e.g., performance, success rate)

# 3. Compute susceptibility
result = adapter.compute_susceptibility(x, y)

# 4. Interpret results
if result['kappa'] > 10:
    print(f"⚠️ Critical behavior detected at σ_c = {result['sigma_c']}")
else:
    print("✓ System is stable")
```

### 7. Advanced: Custom Observables

```python
class MyAdapter(SigmaCAdapter):
    def get_observable(self, data, **kwargs):
        # Your custom metric here
        return np.std(data) / np.mean(data)

# Use it
from sigma_c.adapters.factory import AdapterFactory
AdapterFactory.register('custom', MyAdapter)
my_adapter = AdapterFactory.create('custom')
```

### 8. C++ Integration (High Performance)

```cpp
#include <sigma_c/susceptibility.hpp>

std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
std::vector<double> y = {0.1, 0.3, 0.8, 1.5, 2.0};

auto result = sigma_c::compute_susceptibility(x, y);
std::cout << "Critical point: " << result.sigma_c << std::endl;
```

Link with: `-lsigma_c_core`

### 9. Licensing

**Dual License:**
- **Open Source:** AGPL-3.0 (see `license_AGPL.txt`)
- **Commercial:** Proprietary use without copyleft (contact: nfo@forgottenforge.xyz)

### 10. Support & Documentation

- **Full Docs:** See `DOCUMENTATION.md` (English + German)
- **Examples:** `examples_v4/` directory
- **Issues:** GitHub Issues
- **Contact:** nfo@forgottenforge.xyz

---

## Common Use Cases

### Quantum Computing
```python
qpu = Universe.quantum(device='simulator')
result = qpu.run_optimization(
    circuit_type='grover',
    noise_levels=np.linspace(0, 0.1, 20)
)
# Finds the noise threshold where Grover's algorithm fails
```

### Financial Markets
```python
fin = Universe.finance()
result = fin.detect_regime(symbol='SPY')
# Detects if market is approaching a critical transition
```

### GPU Optimization
```python
gpu = Universe.gpu()
result = gpu.auto_tune(alpha_levels=[0.1, 0.5, 0.9])
# Finds optimal cache/memory configuration
```

---

**Ready to use? Start with the examples and adapt them to your data!** 🚀
