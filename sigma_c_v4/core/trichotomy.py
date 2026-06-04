"""
Three-layer trichotomy classifier — paper §8.

Geometric layer (thm:trichotomy-geometric, unconditional, threshold-free):
  classify by count of strict interior maxima of chi_O.

Spectral attribution (thm:trichotomy-spectral, under epsilon_c, delta_sep):
  attaches each geometric regime to the transfer-operator spectral
  configuration when spectrum is supplied.

Operational diagnostic (def:noise-floor-diagnostic, under eta_O):
  measurement-side check whether candidate peak amplitude is above the
  signal/noise floor.

The three layers are reported as a single Trichotomy dict (Enforcement 1,
Item A of the v4 prescription).
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np

from sigma_c_v4.core.susceptibility import (
    find_interior_maxima,
    has_strictly_monotone_chi,
)
from sigma_c_v4.result import Trichotomy


def geometric_trichotomy(
    chi: np.ndarray,
    *,
    min_prominence_ratio: float = 0.10,
) -> Tuple[str, list]:
    """
    Return (geometric_regime, peak_indices).

    geometric_regime ∈ {"I_geom", "II_geom", "III_geom"}.

    Cite: thm:trichotomy-geometric (unconditional, parameter-free).
    """
    peaks = find_interior_maxima(chi, chi, min_prominence_ratio=min_prominence_ratio)
    # The above call wants (sigma, chi) but only uses chi; we re-use shape.
    peaks = find_interior_maxima(
        np.arange(len(chi)), chi, min_prominence_ratio=min_prominence_ratio
    )
    if len(peaks) == 0:
        return "III_geom", []
    if len(peaks) == 1:
        return "I_geom", peaks
    return "II_geom", peaks


def operational_floor_check(
    chi: np.ndarray,
    O: np.ndarray,
    eta_O: float,
) -> bool:
    """
    True iff the candidate peak amplitude is below eta_O * ||O||_inf.

    Cite: def:noise-floor-diagnostic. Measurement-side diagnostic.
    """
    if eta_O <= 0:
        return False
    O_range = float(np.max(O) - np.min(O))
    if O_range == 0:
        return True
    candidate_peak = float(np.max(chi))
    return candidate_peak < eta_O * O_range


def spectral_attribution(
    spectrum: Optional[Sequence[complex]],
    *,
    epsilon_c: float,
    delta_sep: float,
) -> Optional[str]:
    """
    Spectral attribution under (epsilon_c, delta_sep).

    spectrum: full spectrum {lambda_1, lambda_2, ...} in decreasing |lambda|,
    OR None when no spectral data is available.

    Returns "I_spec" | "II_spec" | "III_spec" | None.

    Cite: thm:trichotomy-spectral.
    """
    if spectrum is None:
        return None
    abs_spec = sorted((abs(complex(x)) for x in spectrum), reverse=True)
    if len(abs_spec) < 2:
        return "III_spec"  # no non-trivial spectrum
    lam1 = abs_spec[0]
    nontrivial = abs_spec[1:]
    # No-gap case: leading nontrivial reaches lambda_1
    if nontrivial[0] >= lam1 * (1 - 1e-12):
        return "III_spec"
    # Single dominant: gap to remaining
    if len(nontrivial) == 1:
        return "I_spec"
    lam2 = nontrivial[0]
    lam3 = nontrivial[1]
    r = lam3 / lam2
    if r >= 1:
        return "III_spec"
    # Are lam2 and lam3 delta_sep-separated?
    sep = (lam2 - lam3) / min(lam2, lam3)
    if sep < delta_sep:
        # modulus-degenerate sub-case — paper rem:modulus-degenerate;
        # we collapse to I_spec following the convention used in §8.
        return "I_spec"
    # Two distinct moduli, both possibly resolved — depends on modal coupling
    # if we had it. With spectrum only, we report II_spec.
    return "II_spec"


def classify(
    sigma: np.ndarray,
    chi: np.ndarray,
    O: np.ndarray,
    *,
    spectrum: Optional[Sequence[complex]] = None,
    epsilon_c: float = 0.25,
    delta_sep: float = 0.10,
    eta_O: float = 0.0,
    min_prominence_ratio: float = 0.10,
) -> Tuple[Trichotomy, list]:
    """
    Full three-layer classification.

    Returns (Trichotomy, peak_indices).
    """
    geom, peaks = geometric_trichotomy(
        chi, min_prominence_ratio=min_prominence_ratio
    )

    # Special case: monotone chi → regime III.c (power-law)
    if geom == "III_geom" and has_strictly_monotone_chi(chi):
        geom = "III_geom"  # already

    spec = spectral_attribution(
        spectrum, epsilon_c=epsilon_c, delta_sep=delta_sep
    )
    floor = operational_floor_check(chi, O, eta_O)

    tri = Trichotomy(
        geometric=geom,
        spectral=spec,
        operational_floor_triggered=floor,
        epsilon_c=epsilon_c,
        delta_sep=delta_sep,
        eta_O=eta_O,
    )
    return tri, peaks
