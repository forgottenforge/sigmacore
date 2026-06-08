"""
Result object — every output carries provenance, regime layers, and
gamma_O stability indicator.

Enforces:
- (Enforcement 2) rho_star_source as a typed field.
- (Enforcement 4) ⊥ as None, never NaN, never default exception.
- (Enforcement 5) two-probe failure with cause branch — see TwoProbeResult.
- (Enforcement 6) detrended flag, off by default.
- (Enforcement 8) smoothing parameters logged.

Cite: paper Def 2.2 (def:sigmac) for the partial-functional convention.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from sigma_c_v4.framework import Framework


# ---------------------------------------------------------------------------
# Regime layers — paper §8 (three-layer trichotomy)
# ---------------------------------------------------------------------------

GeometricRegime = str  # "I_geom" | "II_geom" | "III_geom"
SpectralRegime = str   # "I_spec" | "II_spec" | "III_spec" | None


@dataclass(frozen=True)
class Trichotomy:
    """
    The three-layer trichotomy verdict — paper §8 (cite:
    thm:trichotomy-geometric, thm:trichotomy-spectral,
    def:noise-floor-diagnostic).

    Geometric layer is always present and is unconditional. Spectral
    attribution is None when no spectral data was provided. Operational
    floor is bool.
    """
    geometric: GeometricRegime
    """One of 'I_geom' (unique peak), 'II_geom' (>=2 peaks), 'III_geom' (none).
    Cite: thm:trichotomy-geometric. Unconditional, parameter-free."""

    spectral: Optional[SpectralRegime] = None
    """Spectral attribution under (epsilon_c, delta_sep).
    None when no spectral data was supplied.
    Cite: thm:trichotomy-spectral."""

    operational_floor_triggered: bool = False
    """True iff the candidate peak amplitude is below eta_O * ||O||.
    Cite: def:noise-floor-diagnostic. Measurement-side diagnostic."""

    epsilon_c: float = 0.25  # paper default, Rem after def:thresholds
    delta_sep: float = 0.10
    eta_O: float = 0.0       # 0 = pure-mathematics limit

    def as_dict(self) -> dict:
        return {
            "geometric": self.geometric,
            "spectral": self.spectral,
            "operational_floor_triggered": self.operational_floor_triggered,
            "thresholds": {
                "epsilon_c": self.epsilon_c,
                "delta_sep": self.delta_sep,
                "eta_O": self.eta_O,
            },
        }


# Convenience alias for users who don't want to import Trichotomy directly.
Regime = Trichotomy


# ---------------------------------------------------------------------------
# Main Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """
    The disciplined-reader output. Cite: paper Def 2.2 (def:sigmac).

    sigma_c is None when chi_O has no interior maximum (paper Def 8.1's bottom
    value) — never NaN, never a thrown exception (Enforcement 4).

    For regime II, sigma_c is a list of peak locations.
    """
    # --- σ_c value(s) ---
    sigma_c: Optional[Union[float, List[float]]]
    """Scalar (regime I), list (regime II), or None (regime III). Cite:
    def:sigmac and def:Onice-all."""

    # --- The derived intrinsic scale, when available ---
    tau: Optional[float]
    """The system-intrinsic scale tau = -T_*/log|lambda_2/lambda_1|.
    None when sigma_c is None or no spectral framework was supplied.
    Cite: thm:spectral-id."""

    # --- Profile constant provenance ---
    rho_star: Optional[float]
    """The profile constant. None when sigma_c is None."""

    rho_star_source: str
    """Provenance of rho_star. Either "analytic:<window_name>" for
    the five canonical windows (Enforcement 2), or "fitted" for the
    legacy fit-based path (cite: def:operational-test)."""

    # --- Trichotomy verdict ---
    regime: Trichotomy
    """The three-layer regime classification. Cite: thm:trichotomy."""

    # --- Stability indicator ---
    gamma_O: Optional[float]
    """Strict-SOC constant gamma_O = -d^2/dsigma^2 chi_O^2 at sigma_c.
    Low gamma_O ⟹ flat peak ⟹ noisy/regime-transition-zone reading.
    None when sigma_c is None. Cite: prop:stability."""

    # --- Reproduction recipe (Enforcement 8) ---
    smoothing: dict = field(default_factory=dict)
    """Smoothing parameters used in computing chi_O from samples.
    Keys: kernel, bandwidth, interpolant_order. Empty when input was
    analytical."""

    detrended: bool = False
    """Whether power-law detrending was applied (Enforcement 6).
    Off by default — never silently true. If true, the result is marked
    heuristic per paper §C.3."""

    framework: Optional[Framework] = None
    """The declared transfer-operator framework. None when domain is
    continuous-spectrum or unknown (then tau interpretation is
    dominant-scale-probe, not literal spectral gap; Principle 2)."""

    # --- v4.1 field-found patches ---

    sigma_c_at_grid_boundary: bool = False
    """True iff the chi peak sits at the lowest or highest grid sample,
    so sub-grid quadratic refinement was not possible. Downstream
    cross-tests should treat boundary peaks as soft (the true peak may
    lie outside the scan range entirely). v4.1: field-found in
    a downstream-application Phase 1."""

    preprocessing_scale_equivariant: Optional[bool] = None
    """User declaration about the preprocessing operators applied to
    the observable BEFORE analyze() was called.

    - True  : every preprocessing step (filter, smoothing) is scale
              equivariant; rho_star analytic is applicable, A1 holds.
    - False : at least one preprocessing step carries an absolute
              time-scale (e.g. fixed-bandwidth Butterworth bandpass);
              the framework's analytic rho_star is NOT guaranteed valid
              -- rho_star_source becomes "exploratory:..." and
              .falsifiable returns False.
    - None  : not declared (backward-compatible default). The result
              is the same as v4.0.0, with a single note appended urging
              the user to declare. v4.1: A1 guard introduced after the
              a downstream-application Phase 1 finding."""

    # --- Falsifiability ---
    @property
    def falsifiable(self) -> bool:
        """
        True iff tau was obtained from an analytic rho_star (paper's
        non-circular reading) AND the user has not declared the
        preprocessing as carrying an absolute scale.

        v4.1: a False preprocessing_scale_equivariant declaration
        downgrades the result to exploratory regardless of rho_star
        source -- the user has explicitly told the framework that A1
        is not satisfied at the observable construction layer.
        """
        if self.preprocessing_scale_equivariant is False:
            return False
        return self.rho_star_source.startswith("analytic:")

    # --- Diagnostics ---
    notes: List[str] = field(default_factory=list)
    """Human-readable diagnostic notes (low gamma_O warnings, threshold-flip
    warnings, etc.). Always present, may be empty."""

    # --- Citation: which theorem this output is anchored on ---
    citations: List[str] = field(default_factory=list)
    """List of paper theorem labels (THEOREM_MAP entries) that backed this
    result. The "every output cites a theorem" half of the perfekt-Definition."""

    # --- Visualization data ---
    _profile_sigma: Optional[Any] = field(default=None, repr=False)
    _profile_chi: Optional[Any] = field(default=None, repr=False)
    title: str = ""
    x_name: str = "sigma (resolution)"
    y_name: str = "chi_O(sigma)"

    def card(self, save_to: Optional[str] = None):
        """
        Render a visualization card (PNG if save_to is set; matplotlib Figure
        returned regardless).
        """
        if self._profile_sigma is None or self._profile_chi is None:
            raise RuntimeError(
                "card() requires the chi profile. Use analyze() which "
                "retains it automatically."
            )
        from sigma_c_v4.card import render
        return render(
            self,
            self._profile_sigma,
            self._profile_chi,
            title=self.title,
            x_name=self.x_name,
            y_name=self.y_name,
            save_to=save_to,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def summary(self, ascii_only: bool = True) -> str:
        """Human-readable single-page summary -- what a Card shows in text form.

        ASCII by default for portability (Windows cp1252 console renders
        Unicode as ?). Set ascii_only=False to get the prettier Unicode form
        for UTF-8 terminals.
        """
        from sigma_c_v4.theorem_map import cite

        bot = "_|_" if ascii_only else "⊥"  # ⊥
        bullet = "*" if ascii_only else "•"  # •
        warn = "[!]" if ascii_only else "⚠"  # ⚠
        dash = "--" if ascii_only else "—"   # —

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("sigma_c v4 -- disciplined-reader output")
        lines.append("=" * 60)

        # sigma_c
        if self.sigma_c is None:
            lines.append(f"sigma_c       : {bot} (no interior maximum -- regime III)")
        elif isinstance(self.sigma_c, list):
            vals = ", ".join(f"{v:.4g}" for v in self.sigma_c)
            lines.append(f"sigma_c       : [{vals}]  (regime II, vector-valued)")
        else:
            lines.append(f"sigma_c       : {self.sigma_c:.4g}")

        # tau
        if self.tau is not None:
            lines.append(f"tau           : {self.tau:.4g}")
        else:
            lines.append(f"tau           : {dash}")

        # rho_star + provenance
        if self.rho_star is not None:
            kind = "analytic" if self.falsifiable else "FITTED"
            lines.append(
                f"rho_star      : {self.rho_star:.4g}  "
                f"[{self.rho_star_source}, {kind}]"
            )
        else:
            lines.append(f"rho_star      : {dash}")

        # regime
        rg = self.regime
        spec_str = rg.spectral or dash
        floor_str = " (FLOOR triggered)" if rg.operational_floor_triggered else ""
        lines.append(
            f"regime        : geom={rg.geometric}, spec={spec_str}{floor_str}"
        )

        # stability
        if self.gamma_O is not None:
            gamma_warn = "  (LOW -- transition-zone)" if self.gamma_O < 0.1 else ""
            lines.append(f"gamma_O (SOC) : {self.gamma_O:.4g}{gamma_warn}")
        else:
            lines.append(f"gamma_O (SOC) : {dash}")

        # framework
        if self.framework is not None:
            lines.append(
                f"framework     : {self.framework.value} "
                f"({self.framework.reading_kind})"
            )
        else:
            lines.append("framework     : --  (dominant-scale-probe reading only)")

        # falsifiability flag (Enforcement 2) -- only meaningful when rho_star exists
        if self.rho_star is not None and not self.falsifiable:
            lines.append("                " + warn + " rho_star FITTED -- exploratory, not falsifiable")

        # detrending flag (Enforcement 6)
        if self.detrended:
            lines.append("                " + warn + " DETRENDED -- heuristic, see paper §C.3")

        # notes
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  {bullet} {note}")

        # citations (the "every output cites a theorem" half)
        if self.citations:
            lines.append("")
            lines.append("Theorem backing:")
            for label in self.citations:
                lines.append(f"  {bullet} {cite(label)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        rg = self.regime.geometric
        if self.sigma_c is None:
            return f"Result(sigma_c=None, regime={rg})"
        if isinstance(self.sigma_c, list):
            return f"Result(sigma_c={self.sigma_c}, regime={rg})"
        return (
            f"Result(sigma_c={self.sigma_c:.4g}, tau={self.tau!s:.6}, "
            f"regime={rg}, source={self.rho_star_source})"
        )


# ---------------------------------------------------------------------------
# Two-probe test result (Enforcement 5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoProbeResult:
    """
    Outcome of the non-circular two-probe test, paper Def 6.4
    (def:operational-test-noncirc).

    A failure carries the *cause* — Enforcement 5 — not just a False.
    """
    passed: bool
    delta: float
    """Measured |log(sigma_c1/rho_star_1) - log(sigma_c2/rho_star_2)|."""
    delta_threshold: float
    """The pre-declared tolerance the user supplied."""

    cause: Optional[str] = None
    """When passed=False, one of:
    - "regime_ii"           : probes resolved different moduli, multi-mode system
    - "faithfulness_break"  : at least one probe is not single-mode faithful
    - "system_disagreement" : probes disagree on what counts as 'same system'
    - "probes_not_distinct" : the two reported sigma_c values are closer than the
                              coarser scan's grid spacing -- a PRECONDITION
                              failure of the two-probe test, not a system
                              property. v4.1 field-found in a downstream-application Phase 1:
                              the user's observable construction collapsed two
                              declared windows onto one probe. Fix the
                              observable, not the system.
    - "indeterminate"       : test result inconclusive from data alone
    When passed=True, cause is None.
    """

    tau_1: Optional[float] = None
    tau_2: Optional[float] = None

    rho_star_source_1: str = ""
    rho_star_source_2: str = ""

    @property
    def both_analytic(self) -> bool:
        return (
            self.rho_star_source_1.startswith("analytic:")
            and self.rho_star_source_2.startswith("analytic:")
        )

    def summary(self) -> str:
        from sigma_c_v4.theorem_map import cite

        head = "PASSED" if self.passed else f"FAILED ({self.cause})"
        cite_str = cite(
            "def:operational-test-noncirc" if self.both_analytic
            else "def:operational-test"
        )
        lines = [
            "Two-probe test:",
            f"  status     : {head}",
            f"  delta      : {self.delta:.4g} (tolerance {self.delta_threshold:.4g})",
            f"  tau_1      : {self.tau_1!r}",
            f"  tau_2      : {self.tau_2!r}",
            f"  provenance : {self.rho_star_source_1} | {self.rho_star_source_2}",
            f"  backed by  : {cite_str}",
        ]
        return "\n".join(lines)
