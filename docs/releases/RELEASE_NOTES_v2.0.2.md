# Sigma-C Framework v2.0.2 Release Notes
**"Rigor Refinement" Release**

## 🚀 Overview
Version 2.0.2 is a critical stability and rigor update. It addresses key findings from the Master Audit, specifically targeting numerical stability in long-running streams, AI safety constraints, and statistical significance in scientific domains.

## 🛡️ Critical Fixes

### 1. Numerical Stability (Welford's Algorithm)
- **Module**: `sigma_c.core.control.StreamingSigmaC`
- **Fix**: Replaced naive sum-of-squares variance calculation with **Welford's Online Algorithm**.
- **Impact**: Eliminates catastrophic cancellation errors in long-running streams (>1M data points). Guarantees O(1) updates with high floating-point precision.

### 2. LLM Safety Constraint
- **Module**: `sigma_c.adapters.llm_cost.LLMCostAdapter`
- **Fix**: Introduced `MAX_HALLUCINATION_RATE = 0.15` hard constraint.
- **Impact**: Prevents the cost optimizer from selecting unsafe/hallucinating models purely because they are cheap. Models exceeding the error rate are now strictly disqualified.

### 3. Scientific Rigor (Significance Testing)
- **Module**: `sigma_c.adapters.seismic.SeismicAdapter`
- **Fix**: Added `compute_significance()` method using surrogate data testing (shuffling).
- **Impact**: Provides p-values for observed statistics, allowing researchers to distinguish true criticality from random fluctuations.

## 📦 Installation

```bash
pip install sigma-c-framework==2.0.2
```

## 🔍 Audit Status
- **Master Audit v2.0.0**: ⚠️ Rigor Refinement Required
- **Master Audit v2.0.2**: ✅ **PASSED** (Commercial Ready)

## 📝 Changelog
- [FIX] Implemented Welford's Algorithm for `StreamingSigmaC`
- [FIX] Added hard safety constraint to `LLMCostAdapter`
- [NEW] Added `compute_significance` to `SeismicAdapter`
- [DOC] Updated Audit Protocol to reflect "Dimensionless Universalism" as a feature
