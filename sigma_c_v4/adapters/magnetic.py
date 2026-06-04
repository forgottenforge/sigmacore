"""
v4-Layer-1 adapter: NISQ quantum-magnetism anchor dataset.

The dataset is the empirical anchor of the foundation paper (AVS
Quantum Science 8, 013804, 2026). Six NISQ experiments E1..E6,
each with measured correlation profiles (distances, correlations,
chi). This adapter pipes each experiment through sigma_c_v4.analyze()
and reports the trichotomy verdict per experiment.

The adapter prages no claims of its own. The paper claims (paper
Section 10 anchor table):
  E1 ferromagnetic       regime I    sigma_c = 8.0,  kappa = 1.79
  E2 antiferromagnetic   regime I x 2 (two probes resolving two
                                       different modes)
  E3 entanglement        regime I    sigma_c = 0.67, kappa = 8.58
  E4 domains             regime I    sigma_c = 5.0,  kappa = 1.41
  E5 phase transition    regime I    sigma_c = 1.82, kappa = 2.41
  E6 decoherence         regime I    sigma_c = 0.68, kappa = 1.65

What v4 reports is exactly what analyze() outputs from the raw chi
profile -- not the paper-printed value. Audit: do the v4 verdicts
match the paper's classification?

Cite (THEOREM_MAP):
  def:sigmac, prop:structural-reduction, thm:trichotomy-geometric,
  thm:trichotomy, def:operational-test-noncirc
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np

from sigma_c_v4 import analyze, bare
from sigma_c_v4.result import Result
from sigma_c_v4.adapters.conventions import (
    QmagNISQConvention,
    ConventionResult,
    apply_convention,
)


# The NISQ quantum-magnetism anchor dataset is published separately and
# is NOT bundled with the framework. Obtain it from the foundation paper's
# Zenodo record (doi:10.5281/zenodo.20548818) or from AVS Quantum Science
# 8, 013804 (2026), then pass its path via the `path=` argument.
#
# Environment variable override: SIGMA_C_QMAG_DATASET may be set to the
# absolute path of the JSON file.
DEFAULT_DATASET_ENV_VAR = "SIGMA_C_QMAG_DATASET"


def load_qmag_nisq_dataset(
    path: Optional[Path] = None,
) -> Dict[str, dict]:
    """
    Load the NISQ quantum-magnetism anchor dataset (paper empirical anchor).

    The dataset is published separately at the foundation paper's Zenodo
    record (doi:10.5281/zenodo.20548818) and is NOT bundled with this
    package. Resolution order:

      1. The explicit `path=` argument, if given.
      2. The `SIGMA_C_QMAG_DATASET` environment variable, if set.
      3. Raise FileNotFoundError with a clear message.

    Returns the `experiments` dict keyed by experiment ID (E1_ferromagnetic,
    E2_antiferromagnetic, ...). The adapter does not transform or
    validate the data -- pure port.
    """
    import os

    if path is None:
        env_path = os.environ.get(DEFAULT_DATASET_ENV_VAR)
        if env_path is None:
            raise FileNotFoundError(
                "NISQ quantum-magnetism anchor dataset path not supplied. "
                f"Pass path=... or set the {DEFAULT_DATASET_ENV_VAR} "
                "environment variable. The dataset is published at "
                "https://doi.org/10.5281/zenodo.20548818 (foundation paper) "
                "and is not bundled with this package."
            )
        path = env_path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"NISQ quantum-magnetism anchor dataset not found at {p}."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["experiments"]


# ---------------------------------------------------------------------------
# Per-experiment v4 audit
# ---------------------------------------------------------------------------

@dataclass
class ExperimentAudit:
    """The v4 verdict on a single NISQ quantum-magnetism anchor experiment."""
    experiment_id: str
    """E1_ferromagnetic, E2_antiferromagnetic, ..."""

    paper_sigma_c: Optional[float]
    """The sigma_c value the paper reports for this experiment."""

    v4_result: Result
    """The full v4 Result -- a typed object with regime, gamma_O,
    citations, and rho_star_source."""

    matches_paper: bool
    """True iff the v4 verdict's sigma_c is within tolerance of the
    paper's reported value."""

    notes: List[str]
    """Human-readable notes from the audit (e.g. why it didn't match)."""

    convention_result: Optional[ConventionResult] = None
    """Paper-convention reproduction (savgol smoothing + |dO/dsigma|
    + highest-prominence-peak). Filled when the v4-pure reading disagrees
    with the paper -- documents the convention call honestly rather than
    silently disagreeing."""


# Per-experiment axis mapping derived from inspecting the dataset.
# Each tuple is
#   (x_field, y_field, chi_field, paper_sigma_c_field, x_axis_label)
# chi_field is the precomputed paper-stored chi profile when present
# (paper's domain-specific normalization); use it verbatim per the v4
# discipline (do not silently re-derive). chi_field=None means
# compute chi from O via v4 default formula.
_AXIS_MAP: Dict[str, List[Tuple[str, str, Optional[str], str, str]]] = {
    "E1_ferromagnetic": [
        ("distances", "correlations", "chi", "sigma_c", "lattice distance r"),
    ],
    "E2_antiferromagnetic": [
        ("times", "magnetization_ferro", None, "sigma_c_ferro", "time t (ferro probe)"),
        ("times", "staggered_mag_anti", None, "sigma_c_anti", "time t (anti probe)"),
    ],
    "E3_entanglement_timescales": [
        ("noise_levels", "entanglement_witness", "chi", "sigma_c", "noise level"),
    ],
    "E4_domains": [
        ("domain_sizes", "domain_energies", None, "sigma_c", "domain size"),
    ],
    "E5_phase_transition": [
        ("fields", "zz_correlations", None, "sigma_c_field", "field h"),
    ],
    "E6_decoherence": [
        ("damping_rates", "witnesses", None, "sigma_c", "damping rate"),
    ],
}


def audit_experiment(
    experiment_id: str,
    record: dict,
    *,
    tolerance: float = 0.15,
) -> List[ExperimentAudit]:
    """
    Audit a single experiment record through v4.

    Returns a LIST of ExperimentAudit because some experiments contain
    multiple probes (e.g. E2 antiferromagnetic has both ferro and anti
    probes that the paper treats as two separate regime-(I) measurements).

    Each probe is translated to the v4 input contract:
      - x_field    -> sigma grid (resolution axis)
      - y_field    -> O(sigma)   (the measured observable)
      - bare window (no smoothing applied to data; rho_star = 1 analytic)
      - framework=None (NISQ hardware: dominant-scale-probe domain)
    """
    probes = _AXIS_MAP.get(experiment_id)
    if probes is None:
        return [ExperimentAudit(
            experiment_id=experiment_id,
            paper_sigma_c=record.get("sigma_c"),
            v4_result=None,
            matches_paper=False,
            notes=[f"no axis map registered for {experiment_id}"],
        )]

    results: List[ExperimentAudit] = []
    for x_field, y_field, chi_field, paper_sigma_c_field, x_label in probes:
        notes: List[str] = []
        if x_field not in record or y_field not in record:
            results.append(ExperimentAudit(
                experiment_id=experiment_id,
                paper_sigma_c=record.get(paper_sigma_c_field),
                v4_result=None,
                matches_paper=False,
                notes=[f"record missing '{x_field}' or '{y_field}'"],
            ))
            continue

        x_raw = np.asarray(record[x_field], dtype=float)
        y_raw = np.asarray(record[y_field], dtype=float)
        paper_sigma_c = record.get(paper_sigma_c_field)

        # Lift to positive domain
        mask = x_raw > 0
        sigma = x_raw[mask]
        O = y_raw[mask]

        # Use the paper-stored chi if available (domain-specific normalization);
        # otherwise let v4 derive it from O.
        chi_arr = None
        if chi_field is not None and chi_field in record:
            chi_raw = np.asarray(record[chi_field], dtype=float)
            if chi_raw.shape == x_raw.shape:
                chi_arr = chi_raw[mask]
                notes.append(f"using paper-stored '{chi_field}' (domain "
                             "normalization preserved per v4 honest-data rule)")

        if len(sigma) < 5:
            results.append(ExperimentAudit(
                experiment_id=f"{experiment_id} ({y_field})",
                paper_sigma_c=paper_sigma_c,
                v4_result=None,
                matches_paper=False,
                notes=[f"only {len(sigma)} positive-{x_field} samples; insufficient"],
            ))
            continue

        # Higher prominence ratio than v4 default: NISQ profiles carry
        # finite-shot noise (~5-10% of peak amplitude) that would otherwise
        # register as spurious peaks. 0.30 = 30% of global max -- consistent
        # with the paper's de-facto peak-finding convention.
        result = analyze(
            sigma, O,
            chi=chi_arr,
            window=bare(),
            framework=None,
            min_prominence_ratio=0.30,
            label=f"{experiment_id} ({y_field})",
        )
        result.x_name = x_label
        result.y_name = y_field.replace("_", " ")

        matches = False
        if (paper_sigma_c is not None
                and result.sigma_c is not None
                and not isinstance(result.sigma_c, list)):
            rel_err = abs(result.sigma_c - paper_sigma_c) / max(abs(paper_sigma_c), 1e-12)
            matches = rel_err <= tolerance
            notes.append(f"|v4 - paper| / paper = {rel_err:.1%} "
                         f"(tolerance {tolerance:.0%})")
        elif isinstance(result.sigma_c, list):
            notes.append(f"v4 verdict: regime II ({len(result.sigma_c)} peaks); "
                         "paper anchor lists a single regime-(I) sigma_c")
        elif result.sigma_c is None:
            notes.append("v4 verdict: regime III (no interior peak)")

        # Always run the paper convention alongside the v4-pure reading.
        # This makes the v4 disagreement explicit and reproducible.
        try:
            conv_result = apply_convention(
                experiment_id=f"{experiment_id} ({y_field})",
                sigma=sigma, O=O,
                paper_sigma_c=paper_sigma_c,
                v4_result=result,
            )
            if conv_result.matches_paper:
                notes.append(
                    f"paper-convention reproduction: sigma_c = "
                    f"{conv_result.sigma_c_convention:.4g} matches paper "
                    f"({conv_result.convention_id})"
                )
            else:
                notes.append(
                    f"paper-convention reproduction: sigma_c = "
                    f"{conv_result.sigma_c_convention} "
                    f"(paper {paper_sigma_c}); convention call documented"
                )
        except ImportError:
            conv_result = None
            notes.append("scipy missing -- paper-convention reproduction skipped")

        results.append(ExperimentAudit(
            experiment_id=f"{experiment_id} ({y_field})",
            paper_sigma_c=paper_sigma_c,
            v4_result=result,
            matches_paper=matches,
            notes=notes,
            convention_result=conv_result,
        ))
    return results


def audit_all_experiments(
    *,
    path: Optional[Path] = None,
    tolerance: float = 0.15,
) -> Dict[str, ExperimentAudit]:
    """
    Audit all six NISQ quantum-magnetism anchor experiments through v4.

    Returns a dict {audit_label: ExperimentAudit}. E2 (antiferromagnetic)
    expands into two entries because it has two probes.
    """
    experiments = load_qmag_nisq_dataset(path=path)
    out: Dict[str, ExperimentAudit] = {}
    for exp_id, record in experiments.items():
        for audit in audit_experiment(exp_id, record, tolerance=tolerance):
            out[audit.experiment_id] = audit
    return out


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_audit_report(
    audits: Dict[str, ExperimentAudit],
) -> str:
    """One-page text report of the six-experiment audit."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("v4 audit of NISQ quantum-magnetism anchor dataset (six experiments)")
    lines.append("=" * 70)
    header = (f"{'Probe':<46} {'paper':>9} {'v4-pure':>9} {'conv':>9} "
              f"{'regime':>9}")
    lines.append(header)
    lines.append("-" * 90)
    for exp_id, a in audits.items():
        if a.v4_result is None:
            lines.append(f"{exp_id:<46} {'-':>9} {'-':>9} {'-':>9} {'no v4':>9}")
            continue
        sc_paper = (f"{a.paper_sigma_c:.3g}"
                    if a.paper_sigma_c is not None else "-")
        sc_v4 = ("bottom" if a.v4_result.sigma_c is None
                 else f"{a.v4_result.sigma_c:.3g}"
                 if not isinstance(a.v4_result.sigma_c, list)
                 else f"{len(a.v4_result.sigma_c)}-vec")
        if a.convention_result is not None:
            sc_conv = (f"{a.convention_result.sigma_c_convention:.3g}"
                       if a.convention_result.sigma_c_convention is not None
                       else "-")
            if a.convention_result.matches_paper:
                sc_conv = sc_conv + "*"
        else:
            sc_conv = "-"
        regime = a.v4_result.regime.geometric.replace("_geom", "")
        lines.append(
            f"{exp_id:<46} {sc_paper:>9} {sc_v4:>9} {sc_conv:>9} {regime:>9}"
        )
    lines.append("-" * 90)
    n_v4 = sum(1 for a in audits.values() if a.matches_paper)
    n_conv = sum(
        1 for a in audits.values()
        if a.convention_result is not None
        and a.convention_result.matches_paper
    )
    lines.append(
        f"Summary: v4-pure matches paper in {n_v4}/{len(audits)}; "
        f"paper convention reproduces {n_conv}/{len(audits)}."
    )
    lines.append("Legend: conv* = paper-convention reproduction matches paper.")
    lines.append("")
    lines.append("Per-experiment notes:")
    for exp_id, a in audits.items():
        lines.append(f"  {exp_id}:")
        for note in a.notes:
            lines.append(f"    - {note}")
    lines.append("=" * 70)
    return "\n".join(lines)
