[![PyPI version](https://badge.fury.io/py/sigma-c-framework.svg)](https://badge.fury.io/py/sigma-c-framework)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-green.svg)](https://github.com/forgottenforge/sigmacore/blob/main/license_COMMERCIAL.txt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# Sigma-C Framework v1.1.0

**Copyright (c) 2025 ForgottenForge.xyz**

**Critical Susceptibility Framework** for Quantum, GPU, Financial, Climate, Seismic, and Magnetic analysis.

## 🚀 Quick Start

```bash
# Install the package
pip install sigma-c-framework

# Run examples
python -m sigma_c.examples.demo_quantum
```

Or clone and install from source:
```bash
git clone https://github.com/forgottenforge/sigmacore.git
cd sigmacore/sigma_c_framework
pip install .
```

## 🎯 What is Sigma-C?

Sigma-C detects **critical phase transitions** in complex systems using Critical Susceptibility (χ) theory. Unlike traditional metrics, it identifies the precise scale where systems undergo fundamental structural changes.

**Use Cases:**
- 🔬 **Quantum Computing**: Find noise thresholds that break quantum algorithms
- 🎮 **GPU Optimization**: Auto-tune kernels to avoid cache thrashing
- 💰 **Finance**: Predict market crashes before they happen
- 🌍 **Climate Science**: Identify characteristic scales of weather systems
- 🌋 **Seismology**: Detect critical stress states in earthquake catalogs
- 🧲 **Magnetism**: Analyze phase transitions (Curie temperature)

## 📦 Features

- **6 Domain Adapters** ready for production use
- **🆕 Universal Diagnostics System (v1.1.0)** - Auto-search, validation, and recommendations
- **High-Performance C++ Core** with Python bindings
- **Statistical Robustness** via bootstrap and permutation tests
- **Comprehensive Documentation** in English and German
- **Dual License**: AGPL-3.0 or Commercial

## 📚 Documentation

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) (5 minutes)
- **Full Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md) (English + German)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)

## 💡 Example

### Basic Usage (v1.0.0)
```python
from sigma_c import Universe

# Detect GPU performance critical point
gpu = Universe.gpu()
result = gpu.auto_tune(alpha_levels=[0.1, 0.5, 0.9])

print(f"Critical threshold: {result['sigma_c']:.3f}\"")
print(f"Stability score: {result['statistics']['kappa']:.2f}")
```

### With Diagnostics (v1.1.0) 🆕
```python
from sigma_c import Universe

# Step 1: Diagnose your setup
gpu = Universe.gpu()
diag = gpu.diagnose()

if diag['status'] == 'ok':
    # Step 2: Auto-search optimal parameters
    search = gpu.auto_search()
    print(f"Optimal alpha: {search['best_params']['alpha']:.2f}")
    
    # Step 3: Get human-readable explanation
    result = gpu.auto_tune(alpha_levels=[search['best_params']['alpha']])
    print(gpu.explain(result))
```

## 📄 License

Dual-licensed under AGPL-3.0 or Commercial License.

- **Open Source**: See [license_AGPL.txt](license_AGPL.txt)
- **Commercial**: Contact nfo@forgottenforge.xyz

For commercial licensing without AGPL-3.0 obligations, contact: **nfo@forgottenforge.xyz**

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 🙏 Acknowledgments

The Sigma-C Framework builds upon the excellent work of the open-source community. We gratefully acknowledge the following projects:

### Core Dependencies
- **[NumPy](https://numpy.org/)** - Fundamental package for scientific computing
- **[SciPy](https://scipy.org/)** - Scientific computing library for optimization and statistics
- **[pandas](https://pandas.pydata.org/)** - Data analysis and manipulation library
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library for statistical analysis
- **[pybind11](https://pybind11.readthedocs.io/)** - C++/Python interoperability

### Domain-Specific Libraries
- **[CuPy](https://cupy.dev/)** - GPU-accelerated computing (optional for GPU adapter)
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Financial market data (for Financial adapter)
- **[tqdm](https://tqdm.github.io/)** - Progress bars for long-running computations

### Visualization & Analysis
- **[matplotlib](https://matplotlib.org/)** - Plotting and visualization
- **[seaborn](https://seaborn.pydata.org/)** - Statistical data visualization

### Build & Development Tools
- **[CMake](https://cmake.org/)** - Cross-platform build system
- **[setuptools](https://setuptools.pypa.io/)** - Python package building
- **[wheel](https://wheel.readthedocs.io/)** - Python package distribution format

We are deeply grateful to the maintainers and contributors of these projects for making the Sigma-C Framework possible.

## 📧 Contact

- **Email**: nfo@forgottenforge.xyz
- **GitHub**: [github.com/forgottenforge/sigmacore](https://github.com/forgottenforge/sigmacore)
- **Issues**: [github.com/forgottenforge/sigmacore/issues](https://github.com/forgottenforge/sigmacore/issues)

---

**Made with ❤️ by ForgottenForge**

