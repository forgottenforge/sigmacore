# Changelog

All notable changes to **sigma-c-framework** are documented here.
The library adheres to [Semantic Versioning](https://semver.org/).

Paper anchor: Wurm, M. C. (2026). *Operational scale selection: axioms,
spectral concentration, and a regime trichotomy.* Zenodo,
[doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818).

---

## [5.0.0] — 2026-06-13

**Major version increment.** Two independently justified items, each
validated against self-computed ground truth and combined into one
release for trace coherence:
(i) classifier reformulation with explicit, reportable preconditions
$P_0, P_1, P_2$ on the chi-profile, and an orthogonal spectral-type axis
(gapped / III-pp / III-anom); (ii) canonical AVS-QS empirical-anchor
pipeline exposed in `sigma_c_v4.adapters.avsqs`, with the downstream
drifted convention in `analyze_blind_qpu.compute_kappa` deprecated.

A companion document detailing the reformulations and their numerical
validation is in preparation; the code release stands on its own
validation against analytic ground truth (Markov chains, OU process,
irrational rotations, Pomeau--Manneville maps, doubling map) and against
the as-found AVS-QS publication-era pipeline.

### Reading the version-aware output

- $\sigma_c$ values are invariant under (i): the chi-profile argmax is
  unchanged; the reformulation only adds reportable preconditions and a
  per-precondition output layer.
- Regime labels in transitional bands now carry explicit
  $P_0/P_1/P_2$-flagged preconditions instead of unstated assumptions.
- The spectral-type axis is a new orthogonal output, not a replacement
  for the geometric trichotomy.
- A Branch-(ii) caveat now accompanies regime-I labels obtained from a
  single probe (inferential asymmetry: a contrasting second probe can
  falsify, agreement cannot corroborate).

### Breaking changes

- **Classifier output format.** `analyze().result.regime` now carries a
  preconditions dict with `P0, P1, P2` flags and measured `R, a`
  values. Downstream code that reads `regime` as a bare string will
  continue to work; code that calls `regime.preconditions` is new and
  optional.
- **Regime-I label semantics.** Branch-(ii) regime I (one visible mode)
  is now labeled `I (probe-conditional)` with a standing caveat
  attached. Code that consumed `I` as "the slowest system scale" should
  re-read as "the dominant scale visible to this probe": a single probe
  cannot rule out a slower mode it suppresses below the visibility
  threshold $\theta_\chi$.
- **Spectral-type axis.** New return field
  `analyze().result.spectral_type` ∈ {gapped, III-pp, III-anom,
  undetermined}. Existing code that does not consume this field is
  unaffected.

### Added — Reformulated classifier

- **$T_\ast$ dimensional anchor.** Identification $T_\ast = dt$ for
  continuous-time processes observed at sampling interval $dt$;
  $T_\ast = 1$ in step units for discrete-time. Verified on OU process
  across $\tau_{\rm steps} \in [1, 100]$ with relative bias $\leq 3\%$.
- **Explicit $P_0/P_1/P_2$ preconditions on the chi-profile.**
  Mode visibility ($\theta_\chi = 0.30$ of $\chi_{\max}$), scale
  separation ($R_{\min} = 3$ between rising-peak $\sigma$-values),
  amplitude balance ($a_{\max} = 3$ between rising-peak $\chi$-heights).
  Thresholds locked a priori. Each precondition is reported per
  analysis. Decision rule: regime II requires all three; regime I may
  arise from any of three distinct precondition failures, distinguished
  by which one violated.
- **Orthogonal spectral-type axis.** Locked thresholds: noise floor
  $3 / \sqrt{n}$, persistence threshold $0.05$ of $|\rho(0)|$,
  signal-region zero-crossings $\geq 3$ in first 30 lags. Twelve
  systems correctly classified (six in-sample, six out-of-sample
  including the doubling map and three OU configurations). Honest
  edge-band limitation: slow exponential and polynomial
  autocorrelations cannot be distinguished by persistence alone when
  $\tau_{\rm eff}$ is comparable to the persistence-check lag window.

### Added — Canonical AVS-QS pipeline

- `sigma_c_v4.adapters.avsqs.compute_susceptibility(gammas, observables)`
  exposes the as-found observable-curve gradient pipeline of
  `experiment_particles_v2.py` (lines 623-659, authored
  M. C. Wurm February 2026) that produced the empirical anchor dataset
  `particle_results_v2.json` cited in the foundation paper. Returns
  dict with `sigma_c`, `chi`, `chi_peak`, `scipy_prominence`,
  `kappa_med`, `kappa_z`, `kappa_prom`. The four conventions are locked
  as published:
    - $\sigma_c$ = $\gamma$ at peak with largest scipy prominence
      (prominence floor $10^{-4}$)
    - $\kappa_{\rm med}$ = $\chi(\sigma_c) / {\rm median}(\chi)$
    - $\kappa_z$ = $(\chi(\sigma_c) - {\rm mean}(\chi)) / {\rm std}(\chi)$
      (numpy default ddof=0)
    - $\kappa_{\rm prom}$ = scipy_prominence / mean$(\chi \mid \chi \leq Q_{75}(\chi))$
  The function reproduces all twelve published particle entries
  (anyon, phonon, skyrmion, exciton, photon, magnon, cooper pair,
  soliton, roton, plasmon, majorana, polaron) to two decimal places on
  all four quantities.

### Deprecated

- `analyze_blind_qpu.compute_kappa` (in `onto/particle_plots/`) now
  emits a `DeprecationWarning` pointing to
  `sigma_c_v4.adapters.avsqs.compute_susceptibility` as the canonical
  pipeline. The downstream function uses a different baseline
  normalization (`mean(chi)` over the full grid) and a different
  prominence floor ($10^{-3}$) introduced in a 16-circuit blind QPU
  experiment script post-publication; its kappa values do NOT reproduce
  the published AVS-QS table. This is a convention-drift issue, not a
  bug; the function continues to work but should be migrated.

### Reproducibility

- $\sigma_c$ values from prior releases reproduce exactly under v5.0
  (the chi-profile argmax is unchanged); regime labels in transitional
  bands may shift per-seed under the new explicit-precondition rules.
- AVS-QS-paper $\kappa$ values reproduce in full under
  `sigma_c_v4.adapters.avsqs.compute_susceptibility` against the
  archived dataset `particle_results_v2.json`.
- Prior versions remain installable via version pin; downstream code
  that depends on the prior classifier output format can pin to
  `sigma-c-framework==4.1.1` until migration.

---

## [4.1.1] — 2026-06-08

Three field-found improvements from the first end-to-end Phase-1 run of
an internal downstream Mode-1 application project. Each one extends the
set of failure modes the framework can detect itself, without changing
any analytic API contract from 4.0.

The three patches are **backward-compatible by default**: existing code
gets the new safety nets automatically (sub-grid σ_c, `probes_not_distinct`
precheck), and the one new opt-in kwarg
(`preprocessing_scale_equivariant`) defaults to `None` which preserves
v4.0 behaviour with a single informational note.

### Added — Field-finding 1: sub-grid peak localization

- `sigma_c_v4.core.susceptibility.quadratic_peak_in_log_sigma(sigma_grid,
  chi, peak_idx)` — three-point parabolic interpolation in log σ,
  returning `(sigma_c_sub, at_grid_boundary)`.
- `analyze()` now always returns a sub-grid-refined `sigma_c`. Regime I
  and regime II (vector σ_c) are both refined.
- `Result.sigma_c_at_grid_boundary: bool` — surfaces the (rare) case
  where the peak sits at the lowest or highest grid sample and no
  interpolation is defined. A boundary peak emits a note urging the
  user to widen the σ grid.

### Added — Field-finding 2: `probes_not_distinct` cause branch

- New cause value on `TwoProbeResult.cause`: `"probes_not_distinct"`.
- Triggered when both probes report σ_c values closer than the coarser
  scan's log-grid step. This is a **precondition** failure of the
  two-probe test (the user's observable construction collapsed two
  declared windows onto one probe), distinct from a system
  `"faithfulness_break"`.
- Prevents the trap of feeding the same observable through two different
  window labels and reading the bit-identical σ_c as "two distinct
  probes agreeing." The framework now says no.

### Added — Field-finding 3: `preprocessing_scale_equivariant` (A1 guard)

- New optional kwarg on `analyze()`:
  `preprocessing_scale_equivariant: Optional[bool] = None`.
- `Result.preprocessing_scale_equivariant: Optional[bool]` carries the
  declaration through.
- When the user declares `False` (the observable was built from a
  preprocessing chain that carries an absolute time-scale, e.g. a
  fixed-bandwidth filter, a rolling window with fixed lookback, a
  fixed-period detrending step), the Result is automatically
  downgraded:
    - `rho_star_source` becomes `"exploratory:absolute_scale_preprocessing(...)"`
    - `Result.falsifiable` returns `False` (paper Enforcement 2, with
       the A1-violation flag overriding the analytic-window source)
    - a note explains the downgrade and cites paper A1.
- When `None` (default), backward-compatible — same as v4.0.0, with one
  added note encouraging the user to declare explicitly.

### Changed

- `test_two_analytic_windows_pass_test` updated to use two **canonical**
  observables (`O_exp` from the exponential kernel, `O_gamma2` from the
  gamma_2 kernel) instead of feeding the same `O` to two window labels.
  The v4.0 version of this test silently passed under the
  one-probe-two-rho_star regime that v4.1 now correctly flags as a
  precondition failure.

### Tests

- 10 new tests in `sigma_c_v4/tests/test_field_findings_v41.py`
  pinning each of the three patches.
- 4 sub-grid localization tests, 2 `probes_not_distinct` tests,
  4 A1-guard tests.
- Total v4 test suite: **41 tests, all passing**.

### Notes on v4.1.0

v4.1.0 was published briefly and then yanked because a release note
referenced project-internal context that should not have been part of
the public package. v4.1.1 contains the same three patches and the
same test coverage; the underlying API contract is identical.

---

## [4.0.0] — 2026-06-04

Initial public release of the v4 disciplined-reader kernel + v3
multi-adapter applications stack. Foundation paper on Zenodo at
[doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818).

See the v4.0.0 GitHub release notes at
<https://github.com/forgottenforge/sigmacore/releases/tag/v4.0.0>
for the full feature set.
