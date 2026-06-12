# Sigma-C Framework v5.0.0

**Susceptibility-based scale selection across physical, computational, and
data-driven systems.**

[![License: AGPL-3.0-or-later OR Commercial](https://img.shields.io/badge/License-AGPL_v3_%7C_Commercial-blue.svg)](#license)
[![Version](https://img.shields.io/badge/version-5.0.0-green.svg)](https://pypi.org/project/sigma-c-framework/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.20548818-blue.svg)](https://doi.org/10.5281/zenodo.20548818)
[![Status](https://img.shields.io/badge/status-stable-success.svg)]()

Sigma-C Framework now ships in **two interoperable layers**:

- **v4 disciplined-reader kernel** (`sigma_c_v4/`): the implementation of
  the foundation paper *Operational scale selection: axioms, spectral
  concentration, and a regime trichotomy* (Zenodo DOI
  [10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818),
  JSP submission JOSS-S-26-00346, under review). Every output either
  cites a theorem from the paper or admits it cannot.
- **v1–v3 multi-adapter applications stack** (`sigma_c/`): the existing
  domain adapters (Quantum, Finance, Climate, Seismic, Magnetic, GPU,
  ML, Number Theory, Protein, Edge, LLM cost), public API since v1.0,
  unchanged in v4 except for additive integration with the v4 kernel.

The two layers consume the same susceptibility-peak idea. The kernel is
strict and paper-anchored; the adapters are domain-specific and
operationally framed. Choose whichever fits the question.

## What is sigma_c?

For any system observed at variable resolution `sigma`, the
**susceptibility-peak scale-selection functional** is

```
sigma_c[O] := argmax_sigma |dO / d log sigma|
```

In words: where does the observable `O` respond most sharply to a
multiplicative change of resolution? The location of the peak,
`sigma_c`, is the characteristic scale; the sharpness of the peak,
`kappa`, quantifies how pronounced the transition is.

Under explicit named hypotheses (positive-noise transfer operator with
discrete spectral gap, single-mode faithful observable), `sigma_c`
factors as `sigma_c = rho_star * tau` with `tau = -T_* / log
|lambda_2 / lambda_1|` the system-intrinsic relaxation scale and
`rho_star` a probe-shape constant. The foundation paper formalises
this and proves three load-bearing results: a compatibility theorem,
a cross-observable concentration theorem, and a regime trichotomy
(single-mode / multi-mode / spectrally-flat).

## Install

```bash
pip install sigma-c-framework
```

Optional extras for domain integrations:

```bash
pip install sigma-c-framework[quantum]   # Qiskit, Cirq, PennyLane
pip install sigma-c-framework[gpu]       # CuPy
pip install sigma-c-framework[finance]   # QuantLib
pip install sigma-c-framework[all]       # everything
```

## Quick start

### v4 disciplined kernel

```python
import numpy as np
from sigma_c_v4 import analyze, gamma_k, Framework

# Bare exponential decay -> regime I, sigma_c = relaxation time
sigma = np.geomspace(0.1, 100, 400)
O = np.exp(-sigma / 5.0)
result = analyze(sigma, O, window="bare")

print(result.summary())
# sigma_c       : 5.003
# tau           : 5.003
# rho_star      : 1   [analytic:bare, analytic]
# regime        : geom=I_geom
# Theorem backing:
#   * paper Def 2.2 (def:sigmac)
#   * paper Prop 4.1 (prop:structural-reduction)
#   * paper Thm 8.3 (thm:trichotomy-geometric)
#   * paper Thm 6.1 (thm:cross-obs-concentration)

result.card("out.png")  # FUSE-style light-theme card
```

Power-law data (no characteristic scale) returns `sigma_c = None`
honestly, not NaN:

```python
result = analyze(sigma, sigma**0.6, window="bare")
assert result.sigma_c is None        # regime III, bottom value
assert result.regime.geometric == "III_geom"
```

### v3 multi-adapter applications

```python
import numpy as np
from sigma_c import Universe

temperatures = np.linspace(1.5, 3.5, 50)
magnetization = np.where(
    temperatures < 2.269,
    np.abs(2.269 - temperatures)**0.125,
    0.01 * np.random.randn(50),
)
mag = Universe.magnetic()
result = mag.compute_susceptibility(temperatures, magnetization)
print(f"Detected Tc: {result['sigma_c']:.3f}, kappa = {result['kappa']:.1f}")
```

## Examples

| Path | What it covers |
|------|----------------|
| [`sigma_c_v4/examples/01_coffee_cooling.py`](sigma_c_v4/examples/01_coffee_cooling.py) | Regime I single-mode, sigma_c = relaxation time |
| [`sigma_c_v4/examples/02_zipf_wealth.py`](sigma_c_v4/examples/02_zipf_wealth.py) | Regime III, `sigma_c = bottom` as positive output |
| [`sigma_c_v4/examples/03_two_windows.py`](sigma_c_v4/examples/03_two_windows.py) | rho_star separation principle, two windows on one system |
| [`sigma_c_v4/examples/04_bimodal_relaxation.py`](sigma_c_v4/examples/04_bimodal_relaxation.py) | Regime II multi-mode, vector-valued sigma_c |
| [`sigma_c_v4/examples/05_ising_chain.py`](sigma_c_v4/examples/05_ising_chain.py) | 1D Ising chain anchor, textbook correlation length |
| [`sigma_c_v4/adapters/gallery_qmag_audit.png`](sigma_c_v4/adapters/gallery_qmag_audit.png) | 7-panel audit gallery of the NISQ quantum-magnetism dataset |
| [`examples/v4/demo_quantum.py`](examples/v4/demo_quantum.py) | Quantum noise threshold (v3 stack) |
| [`examples/v4/demo_finance.py`](examples/v4/demo_finance.py) | GARCH volatility (v3 stack) |
| [`examples/v4/demo_climate.py`](examples/v4/demo_climate.py) | Atmospheric mesoscale boundary (v3 stack) |

## Documentation

- [`sigma_c_v4/README.md`](sigma_c_v4/README.md) — v4 kernel scope, modules,
  what is and is not in scope.
- [`THEOREM_MAP.md`](THEOREM_MAP.md) — label → paper-number binding,
  camera-ready safe; v4 docstrings cite labels and resolve numbers here.
- [`docs/01_getting_started.md`](docs/01_getting_started.md) — quick-start
  for the v3 adapters.
- [`docs/02_core_concepts.md`](docs/02_core_concepts.md) — chi, sigma_c,
  kappa, observable discovery.
- [`docs/04_contraction_geometry.md`](docs/04_contraction_geometry.md) —
  v3 contraction geometry (D, gamma).
- [`docs/05_type_classification.md`](docs/05_type_classification.md) — v3
  four-type classification (D / O / S / R), separate from the v4 trichotomy.
- [`docs/06_api_reference.md`](docs/06_api_reference.md) — v3 API.
- [`docs/EXTENDING_DOMAINS.md`](docs/EXTENDING_DOMAINS.md) — write your
  own domain adapter.

## Version history

| Version | What changed |
|---------|--------------|
| **4.0.0** | v4 disciplined-reader kernel (`sigma_c_v4/`): foundation paper implementation, three-layer regime trichotomy (geometric / spectral / operational), F1/F2/F3 faithfulness checkers with explicit C_R constants, non-circular two-probe test, paper-theorem-backed Result type, THEOREM_MAP.md, Wurm 2026 NISQ-anchor adapter with paper-convention overlay, 7-panel audit gallery. v1–v3 API preserved. |
| 3.1.1   | README header bump. |
| 3.1.0   | Linguistics adapter removed (tested and failed; documented). |
| 3.0.0   | Contraction Geometry: D and gamma as first-class metrics; four-type classification (D / O / S / R); Number Theory and Protein Stability adapters. |
| 2.1.0   | Full feature implementation across all adapters. |
| 2.0.3   | Production hardening and the AVS Quantum Science 8, 013804 (2026) reference. |
| 2.0.x   | Rigor refinement, repository reorganisation, bugfix releases. |
| 2.0.0   | 22 integrations, IonQ support, complete documentation. |
| 1.2.3   | Universal optimisation. |
| 1.2.1   | Release notes added. |
| 1.1.0   | Multi-domain extensions. |
| 1.0.0   | Initial release: sigma_c, kappa, eight domain adapters. |

## Foundation paper

> Wurm, M. C. (2026). *Operational scale selection: axioms, spectral
> concentration, and a regime trichotomy.* Zenodo.
> doi:[10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818)
> Manuscript JOSS-S-26-00346 under review at Journal of Statistical
> Physics (Springer).

For the applied validation on real quantum hardware:

> Wurm, M. C. (2026). *Operational scale detection in quantum magnetism.*
> AVS Quantum Science **8** (1), 013804.
> doi:[10.1116/5.0312410](https://doi.org/10.1116/5.0312410)

## License

Sigma-C Framework is offered under a **dual license**:

1. **Open Source / Non-Commercial:** [GNU Affero General Public
   License v3.0 or later](license_AGPL.txt). Derivative works must
   also be licensed under AGPL-3.0-or-later.
2. **Commercial:** for proprietary use without AGPL obligations, see
   [`license_COMMERCIAL.txt`](license_COMMERCIAL.txt) or contact
   [nfo@forgottenforge.xyz](mailto:nfo@forgottenforge.xyz).

See [`LICENSE.txt`](LICENSE.txt) for the dual-license notice.

## Contributing

Issue reports and pull requests are welcome. Before opening a PR
please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Security disclosures go
through [`SECURITY.md`](SECURITY.md).

## Acknowledgements

Sigma-C Framework is developed at **ForgottenForge** by M. C. Wurm.
