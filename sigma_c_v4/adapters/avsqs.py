"""
sigma_c_v4.adapters.avsqs -- Canonical AVS QS empirical-anchor pipeline.

This module exposes the publication-era observable-curve gradient
pipeline used in Wurm 2026, AVS Quantum Sci. 8, 013804 ("Operational
scale detection in quantum magnetism"), reproducing the published
12-particle table (sigma_c, kappa_med, kappa_z, kappa_prom) to two
decimal places under the as-found conventions.

The function is taken AS FOUND from lines 623-659 of the archived
script experiment_particles_v2.py (authored M. C. Wurm, February 2026),
which produced the dataset particle_results_v2.json cited as the
empirical anchor of the foundation paper. See the JSP addendum
(d:/code/onto/new/paper_foundation_sigmac/addendum_v1/addendum.tex),
Section 5 (Version trace), for the full provenance argument and the
12-particle reproduction table.

Conventions (locked, AS PUBLISHED):
    obs_smooth = savgol_filter(obs, 5, polyorder=3, mode='nearest')
    chi(gamma) = |gradient(obs_smooth, gamma)|
    peaks = scipy.signal.find_peaks(chi, prominence=1e-4)
    sigma_c = gamma at peak with largest scipy prominence
    kappa_med  = chi[peak] / median(chi)
    kappa_z    = (chi[peak] - mean(chi)) / std(chi, ddof=0)
    kappa_prom = scipy_prominence / mean(chi | chi <= percentile(chi, 75))

A separate downstream script analyze_blind_qpu.compute_kappa (used in a
later 16-circuit blind QPU experiment, post-AVS-QS publication) uses a
different baseline normalization (mean(chi) over the full grid) and a
different prominence floor (1e-3). The downstream script is now
deprecated; the canonical pipeline lives in this module.
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter


# Locked thresholds, AS PUBLISHED (Wurm 2026 AVS QS 8, 013804).
SAVGOL_WINDOW: int = 5
SAVGOL_POLYORDER: int = 3
SAVGOL_BOUNDARY_MODE: str = 'nearest'
PROMINENCE_FLOOR: float = 1e-4


def compute_susceptibility(
    gammas: np.ndarray,
    observables: np.ndarray,
    smoothing_window: int = SAVGOL_WINDOW,
) -> Dict[str, float]:
    """
    Canonical AVS QS observable-curve gradient pipeline.

    Parameters
    ----------
    gammas : 1D array of decoherence couplings (or other control parameter)
    observables : 1D array of expectation values <O>(gamma), same length as gammas
    smoothing_window : Savitzky-Golay window length, default 5 (AS PUBLISHED)

    Returns
    -------
    dict with keys
      'sigma_c'    : gamma at the principal chi-peak
      'chi'        : full chi(gamma) array
      'chi_peak'   : chi value at sigma_c
      'kappa_med'  : chi[peak] / median(chi)
      'kappa_z'    : (chi[peak] - mean(chi)) / std(chi, ddof=0)
      'kappa_prom' : scipy_prominence / mean(chi | chi <= Q75(chi))

    The four numerical conventions are AS PUBLISHED and should not be
    altered without breaking reproduction of the AVS QS paper's table.
    For sigmacore framework analysis use the higher-level sigma_c_v4.analyze()
    instead; this module is the empirical-anchor compatibility layer.
    """
    obs = np.asarray(observables, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    if len(obs) >= smoothing_window >= 3:
        polyorder = min(SAVGOL_POLYORDER, smoothing_window - 1)
        obs_smooth = savgol_filter(obs, smoothing_window, polyorder,
                                     mode=SAVGOL_BOUNDARY_MODE)
    else:
        obs_smooth = obs.copy()

    chi = np.abs(np.gradient(obs_smooth, gammas))

    peaks, _ = find_peaks(chi, prominence=PROMINENCE_FLOOR)
    if len(peaks) > 0:
        proms = peak_prominences(chi, peaks)[0]
        best_idx = int(np.argmax(proms))
        best_peak = int(peaks[best_idx])
        sigma_c = float(gammas[best_peak])
        peak_val = float(chi[best_peak])
        best_prom = float(np.max(proms))
    else:
        best_peak = int(np.argmax(chi))
        sigma_c = float(gammas[best_peak])
        peak_val = float(chi[best_peak])
        best_prom = float(peak_val - np.min(chi))

    median_chi = float(np.median(chi))
    mean_chi = float(np.mean(chi))
    std_chi = float(np.std(chi))  # ddof=0 (numpy default)
    q75 = np.percentile(chi, 75)
    baseline_mean = float(np.mean(chi[chi <= q75]))

    safe = lambda x: max(x, 1e-15)
    return {
        'sigma_c': sigma_c,
        'chi': chi,
        'chi_peak': peak_val,
        'scipy_prominence': best_prom,
        'kappa_med': peak_val / safe(median_chi),
        'kappa_z': (peak_val - mean_chi) / safe(std_chi),
        'kappa_prom': best_prom / safe(baseline_mean),
    }


__all__ = ['compute_susceptibility',
            'SAVGOL_WINDOW', 'SAVGOL_POLYORDER', 'SAVGOL_BOUNDARY_MODE',
            'PROMINENCE_FLOOR']
