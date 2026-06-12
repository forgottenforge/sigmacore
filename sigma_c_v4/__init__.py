"""
sigma_c_v4 — The disciplined reader for operational scale selection.

Built directly on the foundation paper
  Wurm, "Operational scale selection: axioms, spectral concentration,
   and a regime trichotomy" (JSP submission JOSS-S-26-00346, 2026)

The hero call:
    from sigma_c_v4 import analyze
    result = analyze(sigma, O_values, window="gamma2", framework="reversible_markov")
    print(result.summary())
    result.card("out.png")

Every output either cites a theorem (via THEOREM_MAP) or admits it cannot.
Both are visible from the outside.
"""
from sigma_c_v4.api import analyze, two_probe_test
from sigma_c_v4.result import Result, Regime, Trichotomy
from sigma_c_v4.framework import Framework
from sigma_c_v4.windows import (
    bare,
    gamma_k,
    exponential,
    log_gaussian,
    Window,
    WINDOW_REGISTRY,
)
from sigma_c_v4.core.faithfulness import (
    check_F1,
    check_F2,
    check_F3,
    kl_modal_coefficients,
    FaithfulnessCheck,
)
from sigma_c_v4.theorem_map import cite

__version__ = "5.0.0"
__paper_version__ = "JSP-submission-2026-06-04 (companion addendum: in preparation)"
__paper_doi__ = "10.5281/zenodo.20548818"
__paper_url__ = "https://doi.org/10.5281/zenodo.20548818"

__all__ = [
    "analyze",
    "two_probe_test",
    "Result",
    "Regime",
    "Trichotomy",
    "Framework",
    "Window",
    "WINDOW_REGISTRY",
    "bare",
    "gamma_k",
    "exponential",
    "log_gaussian",
    "check_F1",
    "check_F2",
    "check_F3",
    "kl_modal_coefficients",
    "FaithfulnessCheck",
    "cite",
    "__version__",
    "__paper_version__",
    "__paper_doi__",
    "__paper_url__",
]
