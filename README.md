[![PyPI version](https://badge.fury.io/py/sigma-c-framework.svg)](https://badge.fury.io/py/sigma-c-framework)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-green.svg)](https://github.com/forgottenforge/sigmacore/blob/main/license_COMMERCIAL.txt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# Sigma-C Framework v1.2.0

**Copyright (c) 2025 ForgottenForge.xyz**

**Critical Susceptibility Framework** for Quantum, GPU, Financial, Climate, Seismic, and Magnetic analysis.

## 🚀 Quick Start

```bash
# Install the package
pip install sigma-c-framework

# Run the new v1.2.0 "Full Power" demo
python -m sigma_c.examples_v1_2.demo_universal_rigor
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
- **🆕 Universal Optimization (v1.2.0)** - Balanced optimizers for Fidelity vs. Resilience
- **🆕 Publication-Ready Reporting (v1.2.0)** - Automated LaTeX reports and Nature-style plots
- **Universal Diagnostics System** - Auto-search, validation, and recommendations
- **High-Performance C++ Core** with Python bindings
- **Statistical Robustness** via bootstrap and permutation tests
- **Comprehensive Documentation** in English and German
- **Dual License**: AGPL-3.0 or Commercial

## 📚 Documentation

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) (5 minutes)
- **Full Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md) (English + German)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)

## 💡 Example

### Universal Optimization (v1.2.0) 🆕
```python
from sigma_c.optimization.quantum import BalancedQuantumOptimizer
from sigma_c.adapters.quantum import QuantumAdapter

# Initialize adapter and optimizer
adapter = QuantumAdapter()
optimizer = BalancedQuantumOptimizer(adapter)

# Optimize circuit parameters (balancing fidelity vs. noise resilience)
result = optimizer.optimize_circuit(
    circuit_factory=my_circuit_func,
    param_space={'epsilon': [0.01, 0.05], 'idle_frac': [0.1, 0.2]}
)

print(f"Optimal Params: {result.optimal_params}")
print(f"Critical Stability (Sigma-C): {result.sigma_c_after:.4f}")
```

### Automated Reporting (v1.2.0) 🆕
```python
from sigma_c.reporting.latex import LatexGenerator

report = LatexGenerator()
report.generate_report(
    title="Criticality Analysis",
    sections=[{'title': 'Results', 'content': 'System is stable.'}],
    filename="analysis_report"
)
# Generates analysis_report.tex and compiles to PDF
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

