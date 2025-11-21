# Changelog

All notable changes to the Sigma-C Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-21

### Added
- Initial production release
- 6 domain adapters:
  - Quantum: Noise resilience analysis for quantum circuits
  - GPU: Kernel optimization and auto-tuning
  - Financial: Market regime detection and crash prediction
  - Climate: Spatial scaling analysis for weather data
  - Seismic: Earthquake criticality detection
  - Magnetic: Phase transition analysis (Ising model)
- High-performance C++ core with Python bindings (pybind11)
- `Universe` orchestrator for unified API
- Comprehensive documentation (English + German)
- 6 production-ready demo scripts
- Dual licensing: AGPL-3.0 and Commercial
- Copyright headers in all source files

### Core Features
- Critical susceptibility computation with bootstrap validation
- Statistical tests: Jonckheere-Terpstra, Isotonic Regression, PAVA
- Graceful handling of optional dependencies (braket-sdk, cupy)
- Modular adapter architecture for easy extension

### Documentation
- README.md: Quick overview
- QUICKSTART.md: 5-minute developer guide
- DOCUMENTATION.md: Full reference (bilingual)
- RELEASE.md: GitHub and PyPI publication guide

### Dependencies
- Core: numpy, scipy, pandas, tqdm, matplotlib, seaborn
- Domain-specific: requests, yfinance
- Optional: amazon-braket-sdk (Quantum), cupy (GPU)

[1.0.0]: https://github.com/forgottenforge/sigmacore/releases/tag/v1.0.0
