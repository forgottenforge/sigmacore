# Release v1.2.1: Universal Optimization & Rigor

## 🎉 Major Feature Release

This release introduces **Universal Optimization** and **Publication-Ready Reporting**, transforming Sigma-C from a detection tool into a full-cycle research assistant.

## ✨ New Features

### 1. Universal Optimization (Balanced Optimizers)
New `BalancedOptimizer` classes for Quantum, GPU, and Financial domains that automatically find the sweet spot between competing metrics:
- **Quantum**: Balances Fidelity vs. Noise Resilience
- **GPU**: Balances Throughput vs. Thermal Stability
- **Finance**: Balances Returns vs. Crash Risk

```python
from sigma_c.optimization.quantum import BalancedQuantumOptimizer
optimizer = BalancedQuantumOptimizer(adapter)
result = optimizer.optimize_circuit(...)
```

### 2. Publication-Ready Reporting
Automated generation of scientific reports and plots:
- **`LatexGenerator`**: Creates professional LaTeX reports with your analysis results.
- **`PublicationVisualizer`**: Generates Nature-style plots ready for papers.

```python
from sigma_c.reporting.latex import LatexGenerator
report = LatexGenerator()
report.generate_report(title="Criticality Analysis", ...)
```

### 3. Documentation Overhaul
- Updated `README.md` and `QUICKSTART.md` to reflect v1.2 capabilities.
- New "Full Power" demo: `examples_v1.2/demo_universal_rigor.py`.

## 🔧 Improvements & Fixes

- **Build System**: Fixed issues with `build_release.bat` and `MANIFEST.in` to ensure robust cross-platform builds.
- **Dependencies**: Added `pybind11` as a build requirement.
- **Versioning**: Bumped to v1.2.1 to align PyPI and GitHub releases.

## 📦 Installation

```bash
pip install sigma-c-framework==1.2.1
```

## 🔗 Links

- **PyPI**: https://pypi.org/project/sigma-c-framework/1.2.1/
- **Documentation**: https://github.com/forgottenforge/sigmacore/blob/main/DOCUMENTATION.md
- **Full Changelog**: https://github.com/forgottenforge/sigmacore/compare/v1.1.0...v1.2.1

---
**Made with ❤️ by ForgottenForge**
