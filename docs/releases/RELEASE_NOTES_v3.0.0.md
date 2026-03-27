# Sigma-C Framework v3.0.0 -- Release Notes

**Release Date**: 2026-03-26
**Codename**: "Contraction Geometry"

## Overview

Version 3.0.0 introduces the **contraction geometry framework** as the theoretical foundation of Sigma-C. Two new first-class metrics -- the contraction defect $D$ and drift $\gamma$ -- provide a universal language for describing non-injective operations across all domains. Three new domain adapters (number theory, protein stability, computational linguistics) demonstrate the framework's reach beyond its original physics and engineering roots.

This release is **fully backward compatible** with v2.x. No migration is required for existing code.

## New Features

### Contraction Geometry (`sigma_c.core.contraction`)

The contraction defect $D_M = |S_M| / |f(S_M)|$ and drift $\gamma_M$ are now first-class metrics available across all domains.

- `compute_contraction_defect(f, M)`: Compute $D_M$ at modular resolution $M$.
- `compute_drift(f, M)`: Compute $\gamma_M$ via geometric mean growth.
- `classify_map(D, gamma)`: Predict convergent/divergent behavior.
- `sweep_modular_resolution(f, M_range)`: Sweep $(D, \gamma)$ across resolutions.
- `contraction_principle_check(D_values, gamma_values)`: Verify stabilization.
- `countdown_decomposition(n)`: Decompose orbits into countdown/reset phases.
- `deterministic_trajectory(n)`: Closed-form countdown trajectory.

### Four-Type Classification (`sigma_c.core.classification`)

All operations are classified into four types based on injectivity structure:

| Type | $D$ | Description |
|------|-----|-------------|
| D (Dissipative) | $> 1$ | Non-injective, information-destroying |
| O (Oversaturated) | $> 1$ | Growing pre-image counts |
| S (Symmetric) | $= 1$ | Bijective with symmetry constraint |
| R (Reversible) | $= 1$ | Bijective with conservation law |

- `MapType` enum: `DISSIPATIVE`, `OVERSATURATED`, `SYMMETRIC`, `REVERSIBLE`.
- `classify_operation(D, gamma, ...)`: Decision-tree classification.
- `analyze_type_d/o/s/r(...)`: Type-specific analysis with result objects.

### Number Theory Adapter (`sigma_c.adapters.number_theory`)

Analysis of Collatz-type maps $n \to \text{odd}(qn + c)$ via contraction geometry.

- Contraction defect and drift computation at arbitrary resolutions.
- Behavior prediction for any $qn + c$ map.
- Countdown decomposition and deterministic trajectory computation.
- Twelve-map prediction table with 12/12 verified accuracy.
- Reset distribution verification against $\text{Geo}(1/2)$.
- Physics validation: $D_M \geq 4/3$ bound, $\gamma = q/4$ theorem.

### Protein Stability Adapter (`sigma_c.adapters.protein`)

Protein folding stability via the contraction index $\sigma = D \cdot \gamma$.

- Thermodynamic and mutational contraction index computation.
- Age-dependent sigma drift model with onset prediction.
- Onset envelope for drift rate uncertainty.
- Mechanism classification (stability-driven, gain-of-function, templated, IDP).
- Scope validation for applicability assessment.
- Built-in validation datasets:
  - TTR (25 mutations, Spearman $\rho = -0.984$).
  - LYZ (6 mutations) and GSN (5 mutations) for cross-validation.
  - SOD1 (10 mutations) and PRNP (7 mutations) as negative controls.

### Linguistics Adapter (`sigma_c.adapters.linguistics`)

Etymological depth predicts semantic change.

- Embedded English word lists (225 words, ED 1--5) with semantic change data.
- Embedded German word lists (P/T/O categories) for cross-linguistic comparison.
- Correlation analysis (Pearson, Spearman) between ED and semantic change.
- Fixed-point test (Welch t-test, ED=1 vs ED>1).
- Baron-Kenny mediation analysis (ED -> Frequency -> Change).
- Transparency paradox test (transparent vs opaque words, ED >= 2).
- German anchor test (ANOVA across P/T/O, mirror effect verification).

### Extended Derivative Estimation (`sigma_c.core.derivatives`)

Three new methods for estimating derivatives of noisy observables:

- `savitzky_golay_derivative(x, y, window_length, polyorder)`: Savitzky-Golay filter.
- `spline_derivative(x, y, smoothing_factor)`: Regularized cubic spline.
- `gp_regression_derivative(x, y, length_scale, noise_level)`: Gaussian Process with uncertainty.
- `select_best_method(x, y)`: Automatic method selection based on data characteristics.
- `compute_derivative(x, y, method)`: Unified interface with `'auto'` mode.

### Formal Sigma-C Validation (`sigma_c.core.validation`)

Rigorous statistical validation tools for $\sigma_c$ measurements:

- `check_boundary_conditions(observable, epsilon)`: Existence proof for $\sigma_c$.
- `permutation_test(epsilon, observable, n_permutations)`: Statistical significance test.
- `peak_clarity_test(kappa, kappa_min)`: Peak clarity threshold check.
- `fisher_information_bound(epsilon, observable)`: Cramer-Rao lower bound on susceptibility.
- `observable_quality_score(data, epsilon)`: Decision-tree quality assessment.

### Information Theory Module (`sigma_c.beyond.information`)

Connects contraction geometry to thermodynamics:

- `bits_lost(D)`: Information loss per step ($\log_2(D)$ bits).
- `landauer_cost(D, T)`: Minimum thermodynamic cost of erasure.
- `entropy_production_rate(D, rate, T)`: Entropy production rate.
- `information_summary(D, gamma, T, rate)`: Complete thermodynamic summary.

### New Physics Modules

- `sigma_c.physics.number_theory.RigorousNumberTheorySigmaC`: Validates $D_M \geq 4/3$ bound and $\gamma = q/4$ theorem, checks scaling law stabilization.
- `sigma_c.physics.protein.RigorousProteinSigmaC`: Validates $\sigma(T_m) = 1$ constraint, checks monotonicity of $\sigma$ vs $\Delta\Delta G$.
- Both modules extend `RigorousTheoreticalCheck` with `check_theoretical_bounds`, `check_scaling_laws`, `quantify_resource`, and `validate_sigma_c`.

### New Optimizer

- `sigma_c.optimization.protein.ProteinInterventionOptimizer`: Finds minimal $D \cdot \gamma < 1$ interventions that maximize native-contact preservation ($Q_{\text{nat}}$) while driving $\sigma$ below the instability threshold.

### New Demos and Tests

- 7 new demo scripts covering number theory, protein stability, linguistics, contraction geometry, type classification, information theory, and cross-domain comparison.
- 85+ new unit tests and verification tests covering all new modules.

## Breaking Changes

**None.** Version 3.0.0 is fully backward compatible with v2.x. All existing adapters, optimizers, and physics modules continue to work unchanged.

## Migration Guide

No changes are required for existing v2.x code. The new modules are additive:

```python
# Existing v2.x code continues to work
from sigma_c.adapters.quantum import QuantumAdapter
adapter = QuantumAdapter(config)

# New v3.0 modules are available alongside
from sigma_c.adapters.number_theory import NumberTheoryAdapter
from sigma_c.core.contraction import compute_contraction_defect
from sigma_c.core.classification import MapType
```

## Dependency Changes

- **scipy** is now required (was optional) for the protein and linguistics adapters (`scipy.stats`, `scipy.signal`, `scipy.interpolate`).
- No other new dependencies. Core contraction geometry and classification modules have zero external dependencies beyond numpy.

## Quick Start (New Features)

### Number Theory

```python
from sigma_c.adapters.number_theory import NumberTheoryAdapter

adapter = NumberTheoryAdapter(map_type='collatz')
result = adapter.predict_behavior()
print(f"D={result['D']:.3f}, gamma={result['gamma']:.3f} -> {result['prediction']}")
```

### Protein Stability

```python
from sigma_c.adapters.protein import ProteinAdapter

adapter = ProteinAdapter(protein_name='TTR')
result = adapter.analyze_protein(adapter.TTR_MUTATIONS)
rho = result['correlation']['sigma_vs_onset']['spearman_rho']
print(f"Sigma-onset correlation: rho = {rho:.3f}")
```

### Computational Linguistics

```python
from sigma_c.adapters.linguistics import LinguisticsAdapter

adapter = LinguisticsAdapter(language='english')
results = adapter.run_full_analysis()
print(f"ED-change correlation: r = {results['correlation']['pearson_r']:.3f}")
```

### Contraction Geometry

```python
from sigma_c.core.contraction import compute_contraction_defect, compute_drift, cycle_map

D = compute_contraction_defect(cycle_map, M=14)
gamma = compute_drift(cycle_map, M=14)
print(f"Collatz cycle map: D={D:.3f}, gamma={gamma:.4f}")
```

### Type Classification

```python
from sigma_c.core.classification import classify_operation, MapType

map_type = classify_operation(D=2.06, gamma=0.5625)
assert map_type == MapType.DISSIPATIVE
```

## Verification

All new features validated:

- Number theory: 12/12 map predictions correct.
- Protein stability: TTR $\rho = -0.984$, negative controls correctly excluded.
- Linguistics: ED--change correlation significant, German mirror effect confirmed.
- Contraction defect: $D_M \geq 4/3$ for all tested maps at $M \geq 8$.
- Drift: $\gamma = q/4$ verified to 4 decimal places for $q \in \{3, 5, 7, 9, 11\}$.
- Information theory: Landauer cost matches independent calculation.
- Validation: Permutation test, Fisher bound, and quality score all pass on synthetic benchmarks.

## License

Copyright (c) 2025-2026 ForgottenForge.xyz

Dual-licensed under:
- AGPL-3.0-or-later (open source)
- Commercial License (contact: nfo@forgottenforge.xyz)

---

**Full Changelog**: https://github.com/forgottenforge/sigmacore/compare/v2.1.0...v3.0.0
