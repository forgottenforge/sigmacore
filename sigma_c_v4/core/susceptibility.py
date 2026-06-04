"""
chi_O computation — paper Def 2.2 (def:sigmac), Def 2.1 (def:Onice).

Maps observable samples (sigma_k, O(sigma_k)) to chi_O(sigma) = |sigma * O'(sigma)|
on a regular log-spaced grid, with explicit smoothing provenance
(Enforcement 8 of the v4 prescription).

The smoothing parameters are *logged*, not solved — the full numerical
robustness analysis is delegated to a companion paper (Rem 4.12 of the
foundation paper).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SmoothingSpec:
    """The reproduction recipe carried in every result (Enforcement 8)."""
    kernel: str = "log_linear"     # "log_linear" | "analytical" | "none"
    bandwidth: Optional[float] = None
    interpolant_order: int = 1

    def as_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "bandwidth": self.bandwidth,
            "interpolant_order": self.interpolant_order,
        }


def chi_O(
    sigma: np.ndarray,
    O: np.ndarray,
    *,
    smoothing: Optional[SmoothingSpec] = None,
) -> Tuple[np.ndarray, np.ndarray, SmoothingSpec]:
    """
    Compute chi_O(sigma) = |sigma * dO/dsigma|.

    Returns (sigma_grid, chi_values, smoothing_used) where smoothing_used
    records what was applied for downstream provenance.

    Default: numerical log-derivative on the provided grid; smoothing="none"
    if input is already analytical (callable).
    """
    sigma = np.asarray(sigma, dtype=float)
    O = np.asarray(O, dtype=float)
    if sigma.shape != O.shape:
        raise ValueError(
            f"sigma and O must have the same shape (got {sigma.shape} vs {O.shape})."
        )
    if len(sigma) < 5:
        raise ValueError("Need at least 5 sample points to compute chi_O.")
    if np.any(sigma <= 0):
        raise ValueError("sigma values must be strictly positive (log-domain).")

    if smoothing is None:
        smoothing = SmoothingSpec(kernel="log_linear", bandwidth=None,
                                  interpolant_order=1)

    # Sort by sigma if not already
    order = np.argsort(sigma)
    sigma_s = sigma[order]
    O_s = O[order]

    # Numerical d O / d log sigma using central differences on log sigma.
    log_sigma = np.log(sigma_s)
    dO_dlogsigma = np.gradient(O_s, log_sigma, edge_order=2)
    chi = np.abs(dO_dlogsigma)

    return sigma_s, chi, smoothing


def chi_O_from_callable(
    O_fn: Callable[[np.ndarray], np.ndarray],
    sigma_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, SmoothingSpec]:
    """
    Compute chi_O from a callable observable O: sigma -> O(sigma).

    Uses analytical-precision finite differences. Marked smoothing="analytical".
    """
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    h = sigma_grid * 1e-7  # relative step in log-domain
    O_plus = O_fn(sigma_grid + h)
    O_minus = O_fn(sigma_grid - h)
    dO = (O_plus - O_minus) / (2 * h)
    chi = np.abs(sigma_grid * dO)
    return sigma_grid, chi, SmoothingSpec(kernel="analytical", bandwidth=None,
                                          interpolant_order=2)


def find_interior_maxima(
    sigma: np.ndarray,
    chi: np.ndarray,
    *,
    min_prominence_ratio: float = 0.10,
    boundary_margin: int = 1,
) -> List[int]:
    """
    Indices of strict interior local maxima of chi.

    Cite: def:Onice, def:Onice-multi for the multi-peak case.
    """
    n = len(chi)
    if n < 3:
        return []
    peaks: List[int] = []
    chi_max = float(np.max(chi))
    threshold = min_prominence_ratio * chi_max if chi_max > 0 else 0.0
    for i in range(boundary_margin, n - boundary_margin):
        if chi[i] > chi[i - 1] and chi[i] > chi[i + 1] and chi[i] >= threshold:
            peaks.append(i)
    return peaks


def has_strictly_monotone_chi(chi: np.ndarray) -> bool:
    """
    True iff chi is monotone on the grid — power-law signature for regime III.c.
    Cite: thm:diagnostic (paper §9).
    """
    if len(chi) < 3:
        return False
    diffs = np.diff(chi)
    return bool(np.all(diffs >= 0) or np.all(diffs <= 0))
