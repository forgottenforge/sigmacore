# sigma_c v4

The disciplined-reader implementation of the σ_c foundation paper:

> Wurm, M. C. (2026). *Operational scale selection: axioms, spectral
> concentration, and a regime trichotomy.* Zenodo. doi:[10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818)
> (JSP submission JOSS-S-26-00346, under review.)

> **Every output either cites a theorem or admits it cannot --
> and both are visible from the outside.**

## Status

- 31/31 tests pass (window family + 1D Ising anchor + 4-state Markov C.6
  anchor + enforcement invariants).
- 13 modules in the kernel, all paper-theorem-backed.
- 5 hero examples + 1 audit gallery rendered.
- Paper labels bound via `../THEOREM_MAP.md` (camera-ready safe).
- **User-facing UI is delegated to FUSE** -- see *Out of scope* below.

## Install / use

After `pip install sigma-c-framework` (>= 4.0.0):

```python
from sigma_c_v4 import analyze, gamma_k, Framework

result = analyze(sigma, O, window=gamma_k(2),
                 framework=Framework.REVERSIBLE_MARKOV)
print(result.summary())
result.card("out.png")
```

To run in-place from a clone, install editable: `pip install -e .` from
the repository root.

## Layout

```
sigma_c_v4/
├── __init__.py           hero exports
├── api.py                analyze(), two_probe_test()
├── framework.py          six standard frameworks (paper Prop 5.4)
├── windows/              5 canonical windows with analytical rho_star
├── core/
│   ├── susceptibility.py chi_O = |sigma * dO/dsigma|, smoothing provenance
│   ├── trichotomy.py     3-layer classifier (geom / spec / op)
│   ├── stability.py      gamma_O strict-SOC indicator (Prop 4.4)
│   └── faithfulness.py   F1/F2/F3 checkers with explicit C_R + KL helper
├── result.py             Result, Trichotomy, TwoProbeResult
├── card.py               FUSE-inspired light theme renderer
├── theorem_map.py        cite() helper (reads THEOREM_MAP.md)
├── adapters/             Layer 1: pure ports, no claims
│   ├── magnetic.py       qmag NISQ anchor adapter
│   ├── conventions.py    paper-convention layer (savgol + |dO/dsigma|)
│   └── gallery.py        7-panel composite figure
├── examples/             5 hero examples + rendered cards
└── tests/                31 golden tests
```

## Tests

```
python -m pytest sigma_c_v4/tests/ -q
```

Targets golden values from the paper's `verify_examples.py` (41/41 PASS
at submission) plus enforcement invariants from the v4 prescription.

## Examples

```
PYTHONPATH=. python sigma_c_v4/examples/01_coffee_cooling.py
PYTHONPATH=. python sigma_c_v4/examples/02_zipf_wealth.py
PYTHONPATH=. python sigma_c_v4/examples/03_two_windows.py
PYTHONPATH=. python sigma_c_v4/examples/04_bimodal_relaxation.py
PYTHONPATH=. python sigma_c_v4/examples/05_ising_chain.py
```

Each writes a PNG card next to the script. See
`examples/README.md` for the regime-coverage table.

## qmag NISQ anchor audit

```
python -c "
from sigma_c_v4.adapters import audit_all_experiments, render_audit_gallery
audits = audit_all_experiments(tolerance=0.15)
render_audit_gallery(audits, save_to='sigma_c_v4/adapters/gallery_qmag_audit.png')
"
```

Outcome: v4-pure reading matches paper headline for 1/7 probes (E1
ferromagnetic, exact match). Paper-convention reproduction
(`qmag2026:savgol5_dOdsigma`) matches 6/7. The 1/7 drift is honestly
documented, not hidden. See `adapters/gallery_qmag_audit.png` for the
composite figure.

## What v4 does

- The discipline kernel: turn raw (sigma, O) data into a typed Result
  with regime classification, rho_star provenance, gamma_O stability,
  theorem citations.
- F1/F2/F3 faithfulness checkers with explicit C_R constants (paper
  Prop 5.10) -- no fits.
- Non-circular two-probe test (paper Def 6.4) as default; fit-based
  legacy is opt-in and flagged exploratory.
- Layer-1 adapter for the qmag NISQ anchor with paper-convention overlay.

## What v4 does NOT do (out of scope, intentional)

- **User-facing UI / web app / SaaS surface.** Delegated to a separate
  product layer (the project's existing Card/UI framework). v4 ships as
  a Python library; the UI layer picks it up as one analysis mode among
  several. Keeping the UI out of v4 is intentional discipline: the v4
  surface must be small enough that every output maps to a paper theorem.
- **Unconditional Aczél canonicity** (would discharge R†; flagged as a
  specific open functional-equation problem in the paper's §11.3).
- **Same-mode physical-system verification at r > 0** beyond the 4-state
  Markov anchor (TFIM at h ≠ 0 or AF Heisenberg at finite T are the
  named candidates).
- **Full numerical-robustness analysis** (noise models, sampling, error
  propagation); paper Rem 4.12 delegates this to a companion paper.
  v4 implements the propositional Prop 4.4 bound, not a full numerical
  analysis.
- **Silent detrending** of power-law backgrounds (Cantor IFS, Casimir-
  style RG subtractions). Paper §C.3 marks these as heuristic edges;
  no default detrending, ever.

## Roadmap

1. **Now:** kernel + adapters + golden tests + gallery + memory.
2. **After Zenodo DOI:** update `../THEOREM_MAP.md` `Active paper version`
   block with the DOI; v4 docstring citations then have a permanent
   external anchor.
3. **UI layer integration:** thin wrapper + card variant + web page in
   the project's separate UI product. ~2 days of work.
4. **Pip package:** v4 ships as part of `sigma-c-framework >= 4.0.0`.
   See the repository root `pyproject.toml`.

See:
- `../THEOREM_MAP.md` -- label -> paper-number binding
- `memory:v4-implementation-prescription` -- the discipline rules
- `memory:paper-foundation-sigmac` -- foundation paper claim register
