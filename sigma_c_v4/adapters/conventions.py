"""
Paper-specific convention layer for the NISQ quantum-magnetism anchor (AVS Quantum Science 8, 013804, 2026)
dataset.

The paper's reported sigma_c values use a domain convention that differs
from v4's bare reading: Savitzky-Golay smoothing then a different
chi normalization. This module documents that convention explicitly
and produces a Result with `rho_star_source = "paper_convention:qmag2026"`,
preserving the v4 discipline: a paper-convention result is honestly
flagged as such, not silently passed off as the disciplined-reader output.

Source of the convention (verbatim, derived from the AVS Quantum Science
8, 013804 (2026) reproducibility pipeline):

    def compute_susceptibility(sigmas, observables, smoothing_window=5):
        # Savitzky-Golay smoothing, then chi(sigma) = |dO/dsigma|
        # find_peaks(chi, prominence=0.001), pick highest-prominence
        ...

The convention does NOT use chi(sigma) = |sigma * dO/dsigma| (the paper
Def 2.2 formula). v4 implements the paper Def 2.2 formula; the magnetic
paper used |dO/dsigma| as a domain shortcut. Both are valid measurements
of "susceptibility" with different conventions; v4 lets you pick which.

This file is the bridge between the two. Use only when you want to
reproduce the paper's headline sigma_c values exactly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter, find_peaks, peak_prominences
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# The convention object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QmagNISQConvention:
    """
    The NISQ quantum-magnetism anchor paper convention for computing sigma_c.

    Differences from v4 default:
      1. Apply Savitzky-Golay smoothing to O(sigma) first (window=5, poly=3)
      2. chi(sigma) := |dO/dsigma|     (NOT |sigma * dO/dsigma|)
      3. find_peaks with prominence threshold 0.001
      4. Pick the HIGHEST-PROMINENCE peak (not the global max)

    Used to reproduce the dataset's reported sigma_c values for E2-E6
    where the v4-pure reading sees more structure than the paper headline.
    """
    smoothing_window: int = 5
    polyorder: int = 3
    prominence_threshold: float = 0.001
    convention_id: str = "qmag2026:savgol5_dOdsigma"

    def apply(
        self,
        sigma: np.ndarray,
        O: np.ndarray,
    ) -> Tuple[Optional[float], np.ndarray, dict]:
        """
        Apply the convention. Returns (sigma_c, chi_array, diagnostics).

        sigma_c is None when no peaks pass the prominence threshold.
        """
        if not _HAVE_SCIPY:
            raise ImportError(
                "QmagNISQConvention requires scipy. "
                "Install with: pip install scipy"
            )
        sigma = np.asarray(sigma, dtype=float)
        O = np.asarray(O, dtype=float)

        # 1) Savitzky-Golay smoothing of O
        if len(O) < self.smoothing_window:
            O_smooth = O.copy()
        else:
            poly = min(self.polyorder, self.smoothing_window - 1)
            O_smooth = savgol_filter(
                O, self.smoothing_window, poly, mode="nearest"
            )

        # 2) chi = |dO/dsigma|   (paper convention, NOT |sigma * dO/dsigma|)
        chi = np.abs(np.gradient(O_smooth, sigma))

        # 3) find_peaks with prominence threshold
        peaks, properties = find_peaks(chi, prominence=self.prominence_threshold)

        diagnostics = {
            "convention": self.convention_id,
            "smoothing_window": self.smoothing_window,
            "polyorder": self.polyorder,
            "prominence_threshold": self.prominence_threshold,
            "n_peaks_found": len(peaks),
            "chi_max": float(np.max(chi)) if len(chi) else 0.0,
        }

        if len(peaks) == 0:
            # Fallback: global max
            if len(chi) == 0:
                return None, chi, diagnostics
            peak_idx = int(np.argmax(chi))
            sigma_c = float(sigma[peak_idx])
            diagnostics["selected_peak_idx"] = peak_idx
            diagnostics["selection_rule"] = "fallback_global_max"
        else:
            # 4) Highest-prominence peak wins
            prominences = peak_prominences(chi, peaks)[0]
            best = int(peaks[np.argmax(prominences)])
            sigma_c = float(sigma[best])
            diagnostics["selected_peak_idx"] = best
            diagnostics["selection_rule"] = "highest_prominence"
            diagnostics["prominences"] = prominences.tolist()
            diagnostics["peak_indices"] = peaks.tolist()

        return sigma_c, chi, diagnostics


# ---------------------------------------------------------------------------
# Result objects produced by the convention layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConventionResult:
    """
    The result of applying a paper-specific convention to a probe.

    Distinct type from v4's Result -- to make sure these are NEVER
    accidentally treated as the disciplined-reader output. Falsifiability
    is a CONVENTION CALL, not a theorem.
    """
    experiment_id: str
    sigma_c_convention: Optional[float]
    sigma_c_paper: Optional[float]
    sigma_c_v4_disciplined: Optional[object]  # Result object or None
    convention_id: str
    diagnostics: dict

    @property
    def matches_paper(self) -> bool:
        if self.sigma_c_convention is None or self.sigma_c_paper is None:
            return False
        rel = abs(self.sigma_c_convention - self.sigma_c_paper) / max(
            abs(self.sigma_c_paper), 1e-12
        )
        return rel <= 0.05  # 5% tolerance for convention reproduction

    def summary(self) -> str:
        sigma_str = ("None"
                     if self.sigma_c_convention is None
                     else f"{self.sigma_c_convention:.4g}")
        paper_str = ("-" if self.sigma_c_paper is None
                     else f"{self.sigma_c_paper:.4g}")
        match = "OK" if self.matches_paper else "MISMATCH"
        lines = [
            f"Convention result for {self.experiment_id}:",
            f"  sigma_c (convention)  : {sigma_str}",
            f"  sigma_c (paper)       : {paper_str}",
            f"  convention            : {self.convention_id}",
            f"  match                 : {match}",
            f"  rho_star_source       : paper_convention:{self.convention_id}",
            f"  *** NOT falsifiable as a sigma_c reading; this is a "
            f"convention call ***",
        ]
        for k, v in self.diagnostics.items():
            if isinstance(v, list) and len(v) > 5:
                continue
            lines.append(f"    {k:<22}: {v}")
        return "\n".join(lines)


def apply_convention(
    experiment_id: str,
    sigma: np.ndarray,
    O: np.ndarray,
    paper_sigma_c: Optional[float] = None,
    convention: Optional[QmagNISQConvention] = None,
    v4_result: Optional[object] = None,
) -> ConventionResult:
    """
    Apply a paper convention to a probe and produce a ConventionResult.

    Cite (in user-facing docs): this is NOT the v4 disciplined output --
    it is the paper-convention reproduction, honest about being a
    convention call. See paper Rem 4.12 and the v4 prescription
    Enforcement item 2 for why this is a SEPARATE typed object.
    """
    conv = convention if convention is not None else QmagNISQConvention()
    sigma_c, chi, diag = conv.apply(sigma, O)
    return ConventionResult(
        experiment_id=experiment_id,
        sigma_c_convention=sigma_c,
        sigma_c_paper=paper_sigma_c,
        sigma_c_v4_disciplined=v4_result,
        convention_id=conv.convention_id,
        diagnostics=diag,
    )
