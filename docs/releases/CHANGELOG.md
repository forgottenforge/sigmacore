# Changelog

All notable changes to the Sigma-C Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-21

### Added
- **Universal Diagnostics System** across all 6 domain adapters
  - `diagnose()`: Intelligent health checks and issue detection
  - `auto_search()`: Automated parameter optimization
  - `validate_techniques()`: Domain-specific validation
  - `explain()`: Human-readable result interpretation
- `DiagnosticsEngine` helper class with common utilities (259 lines)
- `demo_diagnostics.py`: Comprehensive diagnostics showcase
- `VERSION_HISTORY.md`: Tracking changes between v1.0.0 and v1.1.0

### Enhanced
- **Quantum Adapter**: Circuit validation, noise calibration, parameter optimization (~287 lines)
- **GPU Adapter**: Cache thrashing detection, bandwidth analysis, auto-tuning (~256 lines)
- **Financial Adapter**: Stationarity tests, volatility clustering, regime validation (~100 lines)
- **Climate Adapter**: Spatial coverage checks, grid resolution optimization (~80 lines)
- **Seismic Adapter**: Catalog completeness, magnitude range validation (~80 lines)
- **Magnetic Adapter**: Equilibration detection, lattice size validation (~80 lines)

### Documentation
- Updated DOCUMENTATION.md with complete diagnostics API reference
- Updated README.md with v1.1.0 features and examples
- Added diagnostics workflow examples

### Statistics
- Total new code: ~883 lines of diagnostics
- Files modified: 10
- New methods per adapter: 4

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
