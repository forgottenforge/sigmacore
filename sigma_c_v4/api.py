"""
Hero API — the disciplined-reader entry point.

    >>> from sigma_c_v4 import analyze
    >>> result = analyze(sigma, O_values)
    >>> print(result.summary())
    >>> result.card("out.png")

Cite (in docstrings throughout): paper labels via THEOREM_MAP.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np

from sigma_c_v4.core.susceptibility import (
    chi_O,
    chi_O_from_callable,
    find_interior_maxima,
    SmoothingSpec,
)
from sigma_c_v4.core.trichotomy import classify
from sigma_c_v4.core.stability import compute_gamma_O
from sigma_c_v4.framework import Framework
from sigma_c_v4.result import Result, TwoProbeResult
from sigma_c_v4.windows import Window, bare, resolve


def analyze(
    sigma: Union[np.ndarray, Sequence[float]],
    O: Union[np.ndarray, Sequence[float], Callable[[np.ndarray], np.ndarray], None] = None,
    *,
    chi: Optional[Union[np.ndarray, Sequence[float]]] = None,
    window: Union[str, Window, None] = "bare",
    framework: Optional[Framework] = None,
    spectrum: Optional[Sequence[complex]] = None,
    T_star: float = 1.0,
    epsilon_c: float = 0.25,
    delta_sep: float = 0.10,
    eta_O: float = 0.0,
    min_prominence_ratio: float = 0.10,
    label: str = "",
) -> Result:
    """
    The hero call. Cite: def:sigmac, thm:trichotomy, prop:structural-reduction.

    Parameters
    ----------
    sigma : array
        Resolution scale samples, strictly positive.
    O : array or callable
        Observable values O(sigma) — either an array matching sigma or
        a callable.
    window : str or Window
        One of "bare", "gamma2", "gamma3", "exponential", "log_gaussian",
        or a Window instance. Cite: paper §4.2 window-family table.
        The window's rho_star is *analytical*, not fitted (Enforcement 2).
    framework : Framework or None
        The transfer-operator setting per paper Prop 5.4. None means
        continuous-spectrum / unknown-operator — dominant-scale-probe
        reading only (Principle 2).
    spectrum : sequence of complex or None
        Optional full spectrum {lambda_1, ...} for spectral attribution
        (cite: thm:trichotomy-spectral).
    T_star : float
        Per-step time-scale; sets units for tau. Default 1.0.
    epsilon_c, delta_sep, eta_O : float
        Trichotomy thresholds. Defaults paper-calibrated (Rem after
        def:thresholds).
    label : str
        Human-readable label for cards / reports.

    Returns
    -------
    Result
        See `sigma_c_v4.result.Result`. sigma_c may be None (regime III),
        scalar (regime I), or list (regime II) — never NaN (Enforcement 4).
    """
    win = resolve(window)
    sigma_arr = np.asarray(sigma, dtype=float)

    if chi is not None:
        # Caller supplied a precomputed chi profile (domain-specific
        # normalization, smoothed measurement, etc.). Honest data path:
        # use the chi values verbatim, do not re-derive them.
        chi_arr = np.asarray(chi, dtype=float)
        order = np.argsort(sigma_arr)
        sigma_grid = sigma_arr[order]
        chi = chi_arr[order]
        if O is not None and not callable(O):
            O_vals = np.asarray(O, dtype=float)[order]
        elif callable(O):
            O_vals = O(sigma_grid)
        else:
            # No O supplied; chi-only mode. We still need *some* O for the
            # signal/noise diagnostic; use chi itself as a proxy amplitude.
            O_vals = chi
        from sigma_c_v4.core.susceptibility import SmoothingSpec
        smoothing = SmoothingSpec(kernel="precomputed",
                                  bandwidth=None,
                                  interpolant_order=0)
    elif callable(O):
        sigma_grid, chi, smoothing = chi_O_from_callable(O, sigma_arr)
        O_vals = O(sigma_grid)
    else:
        if O is None:
            raise ValueError("analyze() needs either O (observable) or chi.")
        O_vals = np.asarray(O, dtype=float)
        sigma_grid, chi, smoothing = chi_O(sigma_arr, O_vals)

    citations: List[str] = ["def:sigmac", "prop:structural-reduction"]
    notes: List[str] = []

    # Classify (three-layer)
    regime, peak_indices = classify(
        sigma_grid, chi, O_vals,
        spectrum=spectrum,
        epsilon_c=epsilon_c, delta_sep=delta_sep, eta_O=eta_O,
        min_prominence_ratio=min_prominence_ratio,
    )
    citations.append("thm:trichotomy-geometric")
    if spectrum is not None:
        citations.append("thm:trichotomy-spectral")
    if eta_O > 0:
        citations.append("def:noise-floor-diagnostic")
    # Framework declaration justifies the spectral-identification citations
    # independent of the geometric regime (declaring a transfer operator IS
    # the spectral hypothesis chain).
    if framework is not None:
        if "prop:standard-frameworks" not in citations:
            citations.append("prop:standard-frameworks")
        if not framework.is_experimental and "thm:spectral-id" not in citations:
            citations.append("thm:spectral-id")

    # No interior peak → regime III, sigma_c = None (Enforcement 4)
    if len(peak_indices) == 0:
        citations.append("thm:diagnostic")
        return Result(
            sigma_c=None, tau=None,
            rho_star=None, rho_star_source="--",
            regime=regime,
            gamma_O=None,
            smoothing=smoothing.as_dict(),
            framework=framework,
            notes=notes + ["No interior peak -- regime III (sigma_c = bottom)."],
            citations=citations,
            _profile_sigma=sigma_grid,
            _profile_chi=chi,
            title=label,
        )

    # Stability check on each peak; warn if low gamma_O
    gamma_vals: List[Optional[float]] = [
        compute_gamma_O(sigma_grid, chi, idx) for idx in peak_indices
    ]
    citations.append("prop:stability")

    # Regime II (multi-mode, vector sigma_c)
    if len(peak_indices) >= 2:
        sigma_c_vec = [float(sigma_grid[idx]) for idx in peak_indices]
        gamma_min = min((g for g in gamma_vals if g is not None), default=None)
        notes.append(
            f"Multi-mode (regime II): {len(sigma_c_vec)} resolved peaks. "
            "sigma_c is vector-valued."
        )
        citations.append("thm:multimode")
        return Result(
            sigma_c=sigma_c_vec, tau=None,
            rho_star=win.rho_star, rho_star_source=win.rho_star_source,
            regime=regime,
            gamma_O=gamma_min,
            smoothing=smoothing.as_dict(),
            framework=framework,
            notes=notes,
            citations=citations,
            _profile_sigma=sigma_grid,
            _profile_chi=chi,
            title=label,
        )

    # Regime I (single mode)
    peak_idx = peak_indices[0]
    sigma_c_val = float(sigma_grid[peak_idx])
    tau_val = sigma_c_val / win.rho_star if win.rho_star > 0 else None
    gamma_val = gamma_vals[0]

    citations.append("thm:cross-obs-concentration")
    if framework is not None and not framework.is_experimental:
        citations.append("prop:standard-frameworks")
        citations.append("thm:spectral-id")
    if framework is not None and framework.is_experimental:
        notes.append(
            "framework=anisotropic_banach is experimental; spectrum is "
            "functional-space-dependent (paper Prop 5.4 case 6)."
        )

    if gamma_val is not None and gamma_val < 0.1:
        notes.append(
            f"Low gamma_O = {gamma_val:.3g} -- peak is flat, reading is in "
            f"the regime-transition zone (paper Prop 12.1)."
        )
        citations.append("prop:transition-zone")

    if regime.operational_floor_triggered:
        notes.append(
            f"Signal/noise floor eta_O={eta_O:.3g} was exceeded; the geometric "
            "verdict is below the operational discrimination threshold "
            "(paper Def 8.8). Operationally classify as regime III."
        )

    return Result(
        sigma_c=sigma_c_val, tau=tau_val,
        rho_star=win.rho_star, rho_star_source=win.rho_star_source,
        regime=regime,
        gamma_O=gamma_val,
        smoothing=smoothing.as_dict(),
        framework=framework,
        notes=notes,
        citations=citations,
        _profile_sigma=sigma_grid,
        _profile_chi=chi,
        title=label,
    )


# ---------------------------------------------------------------------------
# Two-probe test — Enforcement 5 cause-branching
# ---------------------------------------------------------------------------

def two_probe_test(
    result_1: Result,
    result_2: Result,
    *,
    delta_threshold: float = 0.20,
    label_1: str = "probe 1",
    label_2: str = "probe 2",
) -> TwoProbeResult:
    """
    Non-circular operational two-probe test — paper Def 6.4
    (def:operational-test-noncirc).

    Both Result objects must have rho_star_source starting with "analytic:"
    for the test to be falsifiable; otherwise it is a fit-based legacy test
    (def:operational-test) and is marked exploratory.

    Failure carries the cause (Enforcement 5).
    """
    # Coerce to floats / handle None
    if result_1.sigma_c is None or result_2.sigma_c is None:
        # One is in regime III — at least one probe doesn't see a scale.
        s1 = result_1.sigma_c if result_1.sigma_c is not None else float("nan")
        s2 = result_2.sigma_c if result_2.sigma_c is not None else float("nan")
        return TwoProbeResult(
            passed=False,
            delta=float("nan"),
            delta_threshold=delta_threshold,
            cause="indeterminate",
            tau_1=result_1.tau, tau_2=result_2.tau,
            rho_star_source_1=result_1.rho_star_source,
            rho_star_source_2=result_2.rho_star_source,
        )

    if isinstance(result_1.sigma_c, list) or isinstance(result_2.sigma_c, list):
        return TwoProbeResult(
            passed=False, delta=float("nan"),
            delta_threshold=delta_threshold,
            cause="regime_ii",
            tau_1=result_1.tau, tau_2=result_2.tau,
            rho_star_source_1=result_1.rho_star_source,
            rho_star_source_2=result_2.rho_star_source,
        )

    if (result_1.rho_star is None or result_2.rho_star is None
            or result_1.rho_star <= 0 or result_2.rho_star <= 0):
        return TwoProbeResult(
            passed=False, delta=float("nan"),
            delta_threshold=delta_threshold,
            cause="indeterminate",
            tau_1=result_1.tau, tau_2=result_2.tau,
            rho_star_source_1=result_1.rho_star_source,
            rho_star_source_2=result_2.rho_star_source,
        )

    log_tau_1 = math.log(result_1.sigma_c) - math.log(result_1.rho_star)
    log_tau_2 = math.log(result_2.sigma_c) - math.log(result_2.rho_star)
    delta = abs(log_tau_1 - log_tau_2)

    passed = delta <= delta_threshold
    cause: Optional[str] = None
    if not passed:
        # Failure cause inference (Enforcement 5)
        # If both probes produced regime-I geometric verdicts but disagree
        # on tau by a large amount, it's most likely a faithfulness break or
        # a system-disagreement. With only sigma_c info we cannot perfectly
        # distinguish (a) regime-II coverage from (b) faithfulness, so we
        # mark it as "faithfulness_break" by default and let the user
        # supply more data to refine.
        if delta > 2.0:
            cause = "system_disagreement"
        else:
            cause = "faithfulness_break"

    return TwoProbeResult(
        passed=passed,
        delta=delta,
        delta_threshold=delta_threshold,
        cause=cause,
        tau_1=math.exp(log_tau_1),
        tau_2=math.exp(log_tau_2),
        rho_star_source_1=result_1.rho_star_source,
        rho_star_source_2=result_2.rho_star_source,
    )
