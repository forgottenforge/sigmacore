# Sigma-C Framework - Version History

## Version 1.1.0 (In Development)

### New Features
- **Universal Diagnostics System**
  - `diagnose()` - Automated issue detection and recommendations
  - `auto_search()` - Parameter space exploration
  - `validate_techniques()` - Domain-specific requirement checks
  - `explain()` - Human-readable result explanations

### Domain Enhancements
- **Quantum:** Circuit validation, noise calibration, auto-optimization
- **GPU:** Cache analysis, bandwidth detection, kernel tuning
- **Financial:** Stationarity checks, regime validation, window optimization
- **Climate:** Spatial coverage analysis, correlation validation
- **Seismic:** Catalog completeness, binning optimization
- **Magnetic:** Equilibration detection, finite-size scaling checks

### API Changes
- Added diagnostics methods to all adapters
- New `DiagnosticsEngine` helper class
- Enhanced visualization utilities

---

## Version 1.0.0 (2025-11-21)

### Initial Release
- 6 production-ready domain adapters:
  - Quantum (Grover's Algorithm, Noise Analysis)
  - GPU (Auto-tuning, Cache Optimization)
  - Financial (Market Regime Detection)
  - Climate (Spatial Scaling)
  - Seismic (Earthquake Criticality)
  - Magnetic (Ising Model, Curie Temperature)
- High-performance C++ core with Python bindings
- Comprehensive documentation (English + German)
- Dual licensing (AGPL-3.0 / Commercial)
- Published on PyPI and GitHub

### Core Features
- Critical susceptibility computation
- Statistical validation (bootstrap, permutation tests)
- Unified `Universe` API
- Production-ready examples for all domains
