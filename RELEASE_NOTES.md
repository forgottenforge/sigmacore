# Sigma-C Release Notes

## v3.0.0 - "Contraction Geometry" (2026-03-26)

**Full details:** [docs/releases/RELEASE_NOTES_v3.0.0.md](docs/releases/RELEASE_NOTES_v3.0.0.md)

### Summary

Sigma-C v3.0.0 introduces **Contraction Geometry** (D and gamma as first-class metrics), **Four-Type Classification** (D/O/S/R), and three new domain adapters: Number Theory, Protein Stability, and Linguistics. The release adds extended derivative estimation, formal validation, and information theory modules. With 12 domain adapters, 85+ new tests, and 7 new demos, this is the largest feature release to date. Fully backward compatible with v2.x.

---

# Sigma-C v1.2.0 "Full Power" Release Notes

**Date:** November 21, 2025
**Version:** 1.2.0

## 🌟 Highlights

- **Universal Rigor**: The Sigma-C optimization methodology is now available for **Quantum**, **GPU**, and **Financial** domains.
- **Advanced GPU Optimization**: Includes support for **Tensor Cores**, **Async Streams**, and **Memory Coalescing**, validated by real-time hardware monitoring (`pynvml`).
- **Financial Stability**: New `BalancedFinancialOptimizer` for risk-adjusted portfolio optimization.
- **Machine Learning**: New `prediction` module for discovering critical features in data.
- **Publication Ready**: Automated LaTeX report generation and professional plotting tools.

## 🚀 New Features

### Optimization
- `BalancedGPUOptimizer`: Optimizes CUDA kernels for throughput and thermal stability.
- `BalancedFinancialOptimizer`: Optimizes trading strategies for Sharpe ratio and crash resilience.
- `BruteForceEngine`: Parallelized parameter sweeping.

### Physics
- `RigorousGPUSigmaC`: Validates GPU performance against Roofline Model and thermal limits.
- `RigorousFinancialSigmaC`: Checks market efficiency and arbitrage bounds.
- `RigorousQuantumSigmaC`: Enhanced QFI bounds checking.

### Prediction
- `MLDiscovery`: Random Forest-based feature importance for criticality drivers.
- `BlindPredictor`: Predicts system state without direct measurement.

### Reporting
- `LatexGenerator`: Generates .tex reports automatically.
- `PublicationVisualizer`: High-quality plots.

## 🛠️ Fixes & Improvements
- Fixed `QuantumAdapter` simulation mode for custom circuits.
- Improved `verify_all_modules.py` for comprehensive health checks.
- Updated `setup.py` with new dependencies (`pynvml`, `scikit-learn`).

## 📦 Installation

```bash
pip install .
```
