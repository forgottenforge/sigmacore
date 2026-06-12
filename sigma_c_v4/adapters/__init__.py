"""
sigma_c_v4 adapters (Layer 1) -- pure ports, no claims.

Each adapter takes domain-specific raw data and translates it to the
v4 input contract: (sigma_grid, O_values, window choice, framework
declaration). It then calls the sigma_c_v4 hero `analyze()` and returns
the standard Result.

Layer 1 rule: adapters may USE Results, never PRAGE Results. No own
claims. If a fact about a domain is not derivable from the foundation
paper, it does not belong here.

See the v4 prescription (memory:v4-implementation-prescription) for
the discipline.
"""
from sigma_c_v4.adapters.magnetic import (
    load_qmag_nisq_dataset,
    audit_all_experiments,
    audit_experiment,
    format_audit_report,
    ExperimentAudit,
)
from sigma_c_v4.adapters.conventions import (
    QmagNISQConvention,
    ConventionResult,
    apply_convention,
)
from sigma_c_v4.adapters.gallery import render_audit_gallery
from sigma_c_v4.adapters.avsqs import compute_susceptibility as avsqs_compute_susceptibility

__all__ = [
    "load_qmag_nisq_dataset",
    "audit_all_experiments",
    "audit_experiment",
    "format_audit_report",
    "ExperimentAudit",
    "QmagNISQConvention",
    "ConventionResult",
    "apply_convention",
    "render_audit_gallery",
    "avsqs_compute_susceptibility",
]
