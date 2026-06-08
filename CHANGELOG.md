# Changelog

All notable changes to **sigma-c-framework** are documented here.
The library adheres to [Semantic Versioning](https://semver.org/).

Paper anchor: Wurm, M. C. (2026). *Operational scale selection: axioms,
spectral concentration, and a regime trichotomy.* Zenodo,
[doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818).

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
