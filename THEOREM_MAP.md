# THEOREM_MAP — sigma-c-framework v4.0 ↔ foundation paper

This file is the **single source of truth** binding v4 docstrings, golden tests,
and API surfaces to the foundation paper. v4 code cites **labels**, never
numbers; this file holds the label → number mapping per paper revision.

When the paper is renumbered (camera-ready, arXiv revisions, post-review edits),
**update this file only** — the rest of v4 keeps its label citations and follows
the new numbers via this map.

## Active paper version

- **Paper title:** Operational scale selection: axioms, spectral concentration, and a regime trichotomy
- **Author:** M. C. Wurm (ForgottenForge)
- **Paper version:** JSP submission (2026-06-04), 57 pp.
- **Zenodo DOI:** `10.5281/zenodo.20548818`   <https://doi.org/10.5281/zenodo.20548818>
- **arXiv ID:** _(not assigned — math-ph endorsement open; Zenodo is the primary preprint home)_
- **JSP manuscript number:** JOSS-S-26-00346
- **Source paper file:** `paper.tex` (private repo; final version archived at the Zenodo DOI above)
- **Source paper commit / version-hash:** _(record before camera-ready)_

### Citation form for v4 docstrings and code comments

```
Wurm, M. C. (2026). Operational scale selection: axioms, spectral
concentration, and a regime trichotomy. Zenodo. doi:10.5281/zenodo.20548818
(JSP submission JOSS-S-26-00346, under review.)
```

## Mapping table

Stable LaTeX labels are listed verbatim from `paper.aux` of the submission
build. The third column points to the prescription item the label informs
(see [memory] `v4_implementation_prescription.md`, items A–I + Enforcement 1–8).

### Section 2 — Setup

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `def:Onice` | 2.1 | Admissible observable class | core domain type |
| `def:sigmac` | 2.2 | Susceptibility-peak functional | engine entry point |
| `def:Onice-window` | 2.5 | Windowed admissible class | item G |
| `prop:window-bridge` | 2.6 | Windowed bridge | item G (lift theorem) |
| `def:profile-dec` | 2.8 | Profile decomposition | core profile type |

### Section 3 — Axioms

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `thm:compat` | 3.7 | σ_c satisfies the five axioms, unconditional | item I (load-bearing direction) |
| `prop:axiomatic-char` | 3.8 | Conditional canonicity under (R†) | item I (out of scope) |

### Section 4 — Structural reduction

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `prop:structural-reduction` | 4.1 | σ_c = ρ_⋆ · τ | central interpretive output |
| `prop:stability` | 4.4 | Stability under C¹ perturbations | item F (γ_O indicator) |
| `rem:interpolation-rho` | (remark, see paper §4) | Interpolation as ρ_⋆ phenomenon | enforcement 8 (smoothing provenance) |
| `obs:probe-rho-star` | (observation, §4) | Probe-dependence is ρ_⋆ under faithfulness | first-stage faithfulness check copy |

### Section 5 — Spectral identification

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `def:transfer-op` | 5.2 | Positive-noise transfer operator | framework input contract |
| `prop:standard-frameworks` | 5.4 | Six standard frameworks satisfying 5.2 | item E (framework taxonomy) |
| `def:faithful` | 5.8 | Single-mode-faithful observable | faithfulness type |
| `prop:faith-sufficient` | 5.10 | Concrete sufficient conditions (F1/F2/F3) | item B |
| `thm:spectral-id-A` | 5.12 | Spectral identification, abstract perturbation | spectral solver |
| `thm:spectral-id-B` | 5.13 | Spectral identification, operator mechanism | spectral solver |
| `thm:spectral-id` | 5.14 | Spectral identification of τ (real positive case) | bridge prerequisite |

### Section 6 — Bridge

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `thm:cross-obs-concentration` | 6.1 | Cross-observable concentration | bridge output |
| `cor:two-probe` | 6.3 | Two-probe verification protocol | item C (entry point) |
| `def:operational-test-noncirc` | 6.4 | Non-circular operational single-mode test | item C (default) |
| `def:operational-test` | 6.6 | Fit-based operational test (legacy variant) | item C (`fit_based=True`) |

### Section 7 — Probe interpretation

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `prop:probe` | 7.1 | Spectral-gap probe in regime (I) | probe interpretation contract |

### Section 8 — Trichotomy

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `def:Onice-all` | 8.1 | Extended admissible class | trichotomy domain type |
| `thm:trichotomy-geometric` | 8.3 | Geometric trichotomy, unconditional | item A (geometric layer) |
| `def:thresholds` | 8.5 | (ε_c, δ_sep) thresholds | item H |
| `thm:trichotomy-spectral` | 8.6 | Spectral attribution under (ε_c, δ_sep) | item A (spectral layer) |
| `def:noise-floor-diagnostic` | 8.8 | Signal/noise diagnostic for regime III | item A (operational layer) + item H (η_O) |
| `thm:trichotomy` | 8.11 | Unified trichotomy reference | item A (umbrella result) |
| `thm:multimode` | 8.16 | Multi-mode diagnostic | regime II solver |

### Section 9 — Spectrally-flat diagnostic

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `thm:diagnostic` | 9.1 | Spectrally-flat diagnostic | regime III output |

### Section 11 — Time domain

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `prop:drift` | 11.1 | σ_c drift in regime (I) near gap-closure | time-domain extension |

### Section 12 — Transition zone

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `prop:transition-zone` | 12.1 | The transition zone | item F (γ_O low-confidence warning) |

### Appendix A — Conditional canonicity proof

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `lem:smooth-argmax-app` | A.1 | Smooth dependence of argmax | (proof infrastructure, not API) |
| `lem:richness-app` | A.2 | Richness of 𝒪_⋆ on local data | (proof infrastructure, not API) |
| `lem:k-zero-conditional` | A.5 | Boundary-killing of scaling degree | (proof infrastructure, not API) |
| `lem:aczel-A1A4` | A.6 | Aczél reduction (A1)+(A4), conditional on (R†) | (out of scope, item I) |
| `lem:aczel-A2` | A.7 | Aczél reduction (A2) | (out of scope, item I) |
| `lem:aczel-A5` | A.10 | Aczél reduction (A5)+(A4) uniqueness | (out of scope, item I) |
| `def:Onice-multi` | A.12 | Multi-mode admissible class | regime II domain type |
| `def:sigmac-multi` | A.13 | Multi-mode susceptibility-peak functional | regime II output type |
| `thm:multi-char` | A.15 | Multi-mode characterisation (sketched) | item A (II layer; honestly sketched) |

### Appendix B — Spectral identification proof

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `def:Phi-O` | B.1 | Canonical resolution-dependent observable | window construction |
| `lem:O-admissible` | B.3 | Admissibility | windowed-spectral solver |
| `lem:modal-sum-general` | B.4 | Modal-sum observables without window | bare-correlator path |
| `def:faithful-formal` | B.6 | Single-mode faithfulness, formal | item B (formal spec) |
| `lem:residual-perturbation` | B.7 | Log-perturbation of σ_c under residuals | bridge error bound |

### Appendix C — Worked examples

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `lem:multimode-perturb-stmt` | C.1 | Multi-mode argmax separation | item A (II local-max bound) |

### Appendix D — Genericity

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `thm:genericity` | D.1 | Genericity of λ_2-coupling | item B (F1 backbone) |
| `cor:generic-two-probe` | D.3 | Generic two-probe faithfulness | item C (genericity flag) |

### Appendix E — Complex λ_2

| Label | Number | Title (short) | v4 maps to |
|---|---|---|---|
| `thm:spectral-id-complex` | E.1 | Spectral identification, complex λ_2 | (companion / experimental) |
| `prop:reversibility` | E.2 | Real-positive sufficient condition | item E (framework selection) |

## Golden tests anchored to printed numbers

Per enforcement item 1: v4 reproduces the paper's printed numerical values as
contract tests. The anchors:

| Paper location | Numerical claim | Tolerance |
|---|---|---|
| Example C.6 (4-state Markov) | spec(P_ε) = {1, 0.6002, 0.3128, 0.1669} | rel 1e-3 |
| Example C.6 | r ≈ 0.521 | abs 1e-3 |
| Example C.6 | τ ≈ 1.959 (T_* = 1) | abs 1e-3 |
| Example C.6 | ρ_⋆ = 2 − √3 ≈ 0.2679 | exact analytic |
| Example C.6 | σ_c[O_A] ≈ 0.5576, σ_c[O_B] ≈ 0.5243 | abs 1e-3 |
| Example C.6 | cross-probe τ-agreement 6.3% | abs 0.5% |
| Example C.7 (1D Ising, βJ=0.5) | τ = −1/log tanh(βJ) ≈ 1.2954 | abs 1e-3 |
| Example C.7 | σ_c[C] = τ (ρ_⋆ = 1) | exact analytic |
| Example C.7 | σ_c[O_W] = (2 − √3)τ ≈ 0.3471 | abs 1e-3 |
| Example C.7 | grid error ≤ 0.08% (paper bound) | bound check |
| §4.2 window table | Gamma-2 ρ_⋆ = 2 − √3 | exact analytic |
| §4.2 window table | Gamma-3 ρ_⋆ = 3 − 2√2 | exact analytic |
| §4.2 window table | exp / log-Gaussian ρ_⋆ = 1 | exact analytic |
| §10 anchor table | E1 ferro σ_c = 8.0, κ = 1.79 | abs 0.01 (from dataset) |
| §10 anchor table | E2 ferro σ_c = 0.36, anti σ_c = 0.91 | abs 0.01 (from dataset) |

These are the contract tests. They are reproduced by
`supplementary/verify_examples.py` shipped with the foundation paper
(41/41 PASS at submission, archived at the Zenodo DOI above); v4's
test suite inherits this checking and adds the window-family and
anchor-table cases.

## Revision protocol

1. When the paper is renumbered (camera-ready, arXiv revision, post-acceptance edits): recompile `paper.tex` to get `paper.aux`, extract `\newlabel{...}` lines, update the `Number` columns above in place. Do not change the `Label` columns.
2. Append a new entry to the "Active paper version" block with the new version date and arXiv/journal ID. Keep one history line per revision.
3. v4 code is unchanged — it cites labels.
4. Golden tests are unchanged unless the paper actually changes a printed value (extremely rare for numerical anchors that have a verification script).

## Out-of-scope clarifications carried from the prescription

- No label of `prop:axiomatic-char` (3.8), `lem:aczel-*` (A.6, A.7, A.10) is referenced by v4 API code. They live in this map for completeness and to make the out-of-scope-list inspectable.
- The complex-λ_2 spectral identification (`thm:spectral-id-complex`, E.1) is a paper-companion result; v4 marks it as `experimental`, behind a separate feature flag.
- The `lem:multimode-perturb-stmt` "exactly k vs at least k" subtlety (Rem 8.15 of the paper, the Notion-3 bug 1 fix) is implemented as the conservative "at least k"; the tighter version requires a cross-term suppression check which v4 exposes as an opt-in audit, not a default verdict.
