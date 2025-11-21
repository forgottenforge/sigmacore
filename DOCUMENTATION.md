# Sigma-C Framework Documentation / Dokumentation

**Version:** 1.2.0  
**Date:** November 2025  
**Copyright:** (c) 2025 ForgottenForge.xyz

---

## 🇬🇧 English

### 1. Introduction
**Sigma-C Framework** is a high-performance framework for detecting **critical phase transitions** in complex systems. Unlike traditional metrics (like standard deviation or simple thresholds), Sigma-C uses **Critical Susceptibility ($\chi$)** theory to identify the precise scale or parameter value where a system undergoes a fundamental structural change.

**Why is it superior?**
*   **Sensitivity:** Detects precursors to instability (crashes, failures) *before* they happen.
*   **Universality:** The same math works for Quantum Noise, GPU Thrashing, Financial Crashes, and Seismic Activity.
*   **Robustness:** Uses bootstrap and permutation testing to filter out false positives.

### 2. Installation

**Prerequisites:** Python 3.8+, C++17 compiler (MSVC, GCC, or Clang).

#### Windows
```powershell
# 1. Clone the repository
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework

# 2. Install dependencies and build C++ core
pip install .
```

#### Linux / macOS
```bash
# 1. Clone the repository
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework

# 2. Install dependencies and build C++ core
pip install .
```

### 3. Integration

#### Python (Recommended)
The `Universe` class is your single entry point.

```python
from sigma_c import Universe

# Initialize an adapter
gpu = Universe.gpu()

# Run analysis
result = gpu.auto_tune(alpha_levels=[0.1, 0.5, 0.9])
print(f"Critical Point: {result['sigma_c']}")
```

#### C++ (High Performance)
Link against `sigma_c_core.lib` / `libsigma_c_core.so`.

```cpp
#include <sigma_c/susceptibility.hpp>

std::vector<double> x = { ... };
std::vector<double> y = { ... };
auto result = sigma_c::compute_susceptibility(x, y);
std::cout << "Sigma-C: " << result.sigma_c << std::endl;
```

### 4. Domain Examples

See the `examples_v4/` directory for runnable scripts.

*   **Quantum:** `demo_quantum.py` - Detects when noise breaks Grover's algorithm.
*   **Finance:** `demo_finance.py` - Identifies market regimes and crash risks.
*   **GPU:** `demo_gpu.py` - Auto-tunes kernels to avoid cache thrashing.
*   **Climate:** `demo_climate.py` - Finds characteristic scales of weather systems.
*   **Seismic:** `demo_seismic.py` - Detects critical stress correlation lengths.
*   **Magnetic:** `demo_magnetic.py` - Finds Curie temperature in 2D Ising model.
*   **Diagnostics:** `demo_diagnostics.py` - **NEW in v1.1.0** - Comprehensive diagnostics showcase.

### 5. v1.1.0: Universal Diagnostics System 🆕

**New in v1.1.0:** Every adapter now includes intelligent diagnostics, automated parameter search, and human-readable explanations.

#### Diagnostics API

All adapters support four core diagnostic methods:

```python
from sigma_c import Universe

# 1. diagnose() - Health check for your data/configuration
adapter = Universe.quantum()
circuit = adapter.create_grover_with_noise(n_qubits=2, epsilon=0.01)
diag = adapter.diagnose(circuit)

print(f"Status: {diag['status']}")  # 'ok', 'warning', or 'error'
print(f"Issues: {diag['issues']}")
print(f"Recommendations: {diag['recommendations']}")

# 2. auto_search() - Find optimal parameters automatically
search = adapter.auto_search(circuit_type='grover', n_qubits=2)
print(f"Best params: {search['best_params']}")
print(f"Recommendation: {search['recommendation']}")

# 3. validate_techniques() - Verify domain-specific requirements
validation = adapter.validate_techniques(circuit)
print(f"All checks passed: {validation['all_passed']}")

# 4. explain() - Human-readable result interpretation
result = adapter.run_optimization(circuit_type='grover', n_qubits=2)
explanation = adapter.explain(result)
print(explanation)  # Markdown-formatted explanation
```

#### Domain-Specific Diagnostics

**Quantum Domain:**
- Circuit depth and complexity validation
- Gate fidelity estimation
- Noise model checks
- Auto-search: optimal noise levels, idle fractions

**GPU Domain:**
- Cache thrashing detection
- Memory bandwidth analysis
- Kernel efficiency checks
- Auto-search: optimal cache pressure (alpha)

**Financial Domain:**
- Data quality and length checks
- Stationarity tests
- Volatility clustering detection
- Auto-search: optimal lookback windows

**Climate Domain:**
- Spatial grid validation
- Coverage checks
- Auto-search: optimal grid resolutions

**Seismic Domain:**
- Catalog completeness checks
- Magnitude range validation
- Auto-search: optimal spatial binning

**Magnetic Domain:**
- Lattice size validation
- Equilibration checks
- Auto-search: optimal temperature ranges

#### Complete Workflow Example

```python
from sigma_c import Universe

# Step 1: Diagnose your setup
adapter = Universe.finance()
diag = adapter.diagnose(symbol='^GSPC')

if diag['status'] != 'ok':
    print("⚠️ Issues detected:")
    for issue in diag['issues']:
        print(f"  - {issue}")
    print("\n💡 Recommendations:")
    for rec in diag['recommendations']:
        print(f"  - {rec}")
else:
    print("✅ All checks passed!")

# Step 2: Auto-search for optimal parameters
print("\n🔍 Searching for optimal parameters...")
search = adapter.auto_search(symbol='^GSPC')
print(f"Best parameters: {search['best_params']}")

# Step 3: Run analysis with optimal parameters
result = adapter.detect_regime(symbol='^GSPC', **search['best_params'])

# Step 4: Get human-readable explanation
explanation = adapter.explain(result)
print(explanation)
```

For a complete demonstration across all 6 domains, run:
```bash
python examples_v4/demo_diagnostics.py
```

### 6. Troubleshooting

*   **Build Errors:** Ensure you have a C++17-compatible compiler installed.
*   **Import Errors:** Verify that the package is installed in your active Python environment.
*   **Performance Issues:** Check that you're using the correct adapter for your domain.
*   **Diagnostics Warnings:** Run `adapter.diagnose()` to get specific recommendations.

---

## 🇩🇪 Deutsch

### 1. Einführung
**Sigma-C Framework** ist ein Hochleistungs-Framework zur Erkennung von **kritischen Phasenübergängen** in komplexen Systemen. Im Gegensatz zu herkömmlichen Metriken (wie Standardabweichung) nutzt Sigma-C die Theorie der **Kritischen Suszeptibilität ($\chi$)**, um den genauen Punkt zu identifizieren, an dem sich die Struktur eines Systems grundlegend ändert.

**Warum ist es überlegen?**
*   **Sensitivität:** Erkennt Vorboten von Instabilität (Abstürze, Ausfälle), *bevor* sie eintreten.
*   **Universalität:** Die gleiche Mathematik funktioniert für Quantenrauschen, GPU-Thrashing, Finanzcrashs und Erdbeben.
*   **Robustheit:** Nutzt Bootstrap- und Permutationstests, um Fehlalarme auszuschließen.

### 2. Installation

**Voraussetzungen:** Python 3.8+, C++17 Compiler.

#### Windows
```powershell
# 1. Repository klonen
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework

# 2. Installieren
pip install .
```

#### Linux / macOS
```bash
# 1. Repository klonen
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework

# 2. Installieren
pip install .
```

### 3. Integration

#### Python (Empfohlen)
Die `Universe`-Klasse ist der zentrale Einstiegspunkt.

```python
from sigma_c import Universe

# Adapter initialisieren
gpu = Universe.gpu()

# Analyse starten
result = gpu.auto_tune(alpha_levels=[0.1, 0.5, 0.9])
print(f"Kritischer Punkt: {result['sigma_c']}")
```

#### C++ (Hochleistung)
Gegen `sigma_c_core.lib` / `libsigma_c_core.so` linken.

```cpp
#include <sigma_c/susceptibility.hpp>

std::vector<double> x = { ... };
std::vector<double> y = { ... };
auto result = sigma_c::compute_susceptibility(x, y);
std::cout << "Sigma-C: " << result.sigma_c << std::endl;
```

### 4. Anwendungsbeispiele

Siehe `examples_v4/` für ausführbare Skripte.

*   **Quantum:** `demo_quantum.py` - Erkennt, wann Rauschen den Grover-Algorithmus bricht.
*   **Finance:** `demo_finance.py` - Identifiziert Marktregimes und Crash-Risiken.
*   **GPU:** `demo_gpu.py` - Optimiert Kernel, um Cache-Thrashing zu vermeiden.
*   **Climate:** `demo_climate.py` - Findet charakteristische Skalen von Wettersystemen.
*   **Seismic:** `demo_seismic.py` - Erkennt kritische Spannungskorrelationen.
*   **Magnetic:** `demo_magnetic.py` - Findet Curie-Temperatur im 2D-Ising-Modell.
