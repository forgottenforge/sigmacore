"""
Framework taxonomy — the six standard operator settings of paper Prop 5.4.

Cite: prop:standard-frameworks (THEOREM_MAP).

The declaration determines which kind of spectral identification the bridge
can deliver. None → restrict to dominant-scale-probe reading (Principle 2 of
the v4 prescription).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sigma_c_v4.theorem_map import cite


class Framework(str, Enum):
    """The six paper-verified transfer-operator settings (Prop 5.4)."""

    REVERSIBLE_MARKOV = "reversible_markov"
    DOEBLIN = "doeblin"
    COMPACT_HILBERT = "compact_hilbert"
    GNS_LINDBLAD = "gns_lindblad"
    TRANSFER_MATRIX_1D = "transfer_matrix_1d"
    # Prop 5.4 case (6): functional-space-dependent. Shipped experimental.
    ANISOTROPIC_BANACH = "anisotropic_banach"

    @property
    def is_experimental(self) -> bool:
        """Anisotropic-Banach requires user-supplied spectral hypotheses."""
        return self is Framework.ANISOTROPIC_BANACH

    @property
    def reading_kind(self) -> str:
        """
        Whether sigma_c/rho_star is a literal spectral-gap reading
        (cite: prop:probe, Caveat 2 of §7.2) or the operational
        dominant-scale reading.
        """
        return "spectral_gap" if not self.is_experimental else "spectral_gap_experimental"

    def cite_paper(self) -> str:
        return cite("prop:standard-frameworks", note=f"case for {self.value}")


def reading_kind(framework: Optional[Framework]) -> str:
    """
    Resolve the reading-kind for the user-declared framework.

    framework=None → continuous-spectrum / unknown-operator domains.
    Principle 2 of the prescription: dominant-scale-probe only.
    """
    if framework is None:
        return "dominant_scale_probe"
    return framework.reading_kind


@dataclass(frozen=True)
class FrameworkHypotheses:
    """
    Spectral hypotheses required when framework=ANISOTROPIC_BANACH.

    The user must supply these explicitly; the framework refuses to quote tau
    without them (Enforcement item 7).
    """
    anisotropic_norm: str
    asserted_leading_resonance: complex

    def __post_init__(self):
        if not self.anisotropic_norm:
            raise ValueError(
                "anisotropic_banach framework requires a stated anisotropic "
                "norm; see prop:standard-frameworks case (6)."
            )
