# Sigma-C Framework - Version History

## Version 3.0.0 (2026-03-26)

### New Features
- **Contraction Geometry**: D (contraction diameter) and gamma (contraction ratio) as first-class metrics
- **Four-Type Classification**: D (Divergent), O (Oscillatory), S (Stable), R (Resonant)
- **Extended Derivative Estimation**: Configurable derivative methods for susceptibility computation
- **Formal Validation**: Rigorous mathematical validation of criticality claims
- **Information Theory**: Shannon entropy and mutual information analysis

### New Domain Adapters
- **Number Theory**: `NumberTheoryAdapter` with 12-map verification and prime distribution analysis
- **Protein Stability**: `ProteinAdapter` with TTR/LYZ/GSN/SOD1/PRNP mutation datasets
- **Linguistics**: `LinguisticsAdapter` with English/German cross-language correlation

### New Modules
- `core/contraction.py` - Contraction geometry computations
- `core/classification.py` - Four-type system classification
- `core/derivatives.py` - Extended derivative estimation
- `core/validation.py` - Formal validation framework
- `beyond/information.py` - Information-theoretic analysis
- `RigorousNumberTheorySigmaC`, `RigorousProteinSigmaC` - Rigorous analysis classes
- `ProteinInterventionOptimizer` - Protein intervention optimization

### Testing & Demos
- 85+ new tests across 9 new test modules
- 7 new demo scripts
- 7 new documentation files

### Compatibility
- Fully backward compatible with v2.x
- 12 domain adapters total (was 9)
- Engine supports `derivative_method` parameter
- `Universe` has 3 new factory methods

---

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
