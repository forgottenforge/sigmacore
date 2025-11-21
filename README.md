[![PyPI version](https://badge.fury.io/py/sigma-c-framework.svg)](https://badge.fury.io/py/sigma-c-framework)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# Sigma-C Framework v1.0.0

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
- **High-Performance C++ Core** with Python bindings
- **Statistical Robustness** via bootstrap and permutation tests
- **Comprehensive Documentation** in English and German
- **Dual License**: AGPL-3.0 or Commercial

## 📚 Documentation

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) (5 minutes)
- **Full Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md) (English + German)
- **Release Guide**: See [RELEASE.md](RELEASE.md) (for contributors)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)

## 💡 Example

```python
from sigma_c import Universe

# Detect GPU performance critical point
gpu = Universe.gpu()
result = gpu.auto_tune(alpha_levels=[0.1, 0.5, 0.9])

print(f"Critical threshold: {result['sigma_c']:.3f}")
print(f"Stability score: {result['statistics']['kappa']:.2f}")
```

## 📄 License

Dual-licensed under AGPL-3.0 or Commercial License.

- **Open Source**: See [license_AGPL.txt](license_AGPL.txt)
- **Commercial**: Contact nfo@forgottenforge.xyz

For commercial licensing without AGPL-3.0 obligations, contact: **nfo@forgottenforge.xyz**

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📧 Contact

- **Email**: nfo@forgottenforge.xyz
- **GitHub**: [github.com/forgottenforge/sigmacore](https://github.com/forgottenforge/sigmacore)
- **Issues**: [github.com/forgottenforge/sigmacore/issues](https://github.com/forgottenforge/sigmacore/issues)

---

**Made with ❤️ by ForgottenForge**
