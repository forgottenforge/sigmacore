#!/usr/bin/env python3
"""
Rigorous Number Theory Sigma_c
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Validates sigma_c values for number-theoretic systems against fractal dimension
bounds (D_M >= 4/3) and modular resolution scaling laws.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Any, Dict, List
import numpy as np
from sigma_c.physics.rigorous import RigorousTheoreticalCheck


class RigorousNumberTheorySigmaC(RigorousTheoreticalCheck):
    """
    Checks if measured sigma_c respects number-theoretic bounds.

    The key quantities are:
    - D_M : fractal (Hausdorff) dimension of the modular distribution at
      resolution M.  Theory requires D_M >= 4/3.
    - gamma : dissipation rate, defined as gamma = q / 4 where q is the
      modular parameter (e.g. prime or prime power).
    - M : modular resolution, which serves as the fundamental resource.
    """

    def check_theoretical_bounds(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Verify that the fractal dimension D_M satisfies the theoretical
        lower bound D_M >= 4/3 and compute the dissipation rate gamma = q/4.

        Parameters
        ----------
        data : dict
            Must contain:
            - ``D_M`` (float): measured fractal dimension.
            - ``q`` (int or float): modular parameter.
            May optionally contain:
            - ``upper_bound`` (float): domain-specific upper bound for D_M
              (defaults to 2.0, the embedding dimension for planar sets).

        Returns
        -------
        dict
            Bound-checking results including ``lower_bound``, ``upper_bound``,
            ``gamma``, and pass/fail flags.
        """
        D_M = data.get("D_M", 0.0)
        q = data.get("q", 4)

        lower_bound = 4.0 / 3.0  # theoretical minimum for D_M
        upper_bound = data.get("upper_bound", 2.0)
        gamma = q / 4.0

        passes_lower = D_M >= lower_bound
        passes_upper = D_M <= upper_bound

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "D_M": D_M,
            "gamma": gamma,
            "passes_lower": passes_lower,
            "passes_upper": passes_upper,
            "metric": "D_M",
            "theory": "Fractal Dimension Bound (D_M >= 4/3)",
        }

    def check_scaling_laws(
        self, data: Any, param_range: List[float], **kwargs
    ) -> Dict[str, Any]:
        """
        Verify that D_M stabilises as the modular resolution M increases and
        fit a power-law D_M(M) = a * M^b to quantify the scaling exponent.

        For well-behaved number-theoretic distributions the exponent should
        converge toward 0 (i.e. D_M becomes independent of M), indicating
        stabilisation.

        Parameters
        ----------
        data : dict or Any
            If a dict, may contain ``D_M_values`` (list of floats) measured at
            each resolution in *param_range*.  If not provided, a synthetic
            model D_M = 4/3 + 0.1 / sqrt(M) is used.
        param_range : list of float
            Modular resolution values M at which D_M was (or will be) measured.

        Returns
        -------
        dict
            Scaling-law fit results including the exponent, stabilisation flag,
            and fit quality.
        """
        if not param_range or len(param_range) < 2:
            return {"status": "insufficient_data"}

        M = np.array(param_range, dtype=float)

        # Obtain D_M values ---------------------------------------------------
        D_M_values = (
            data.get("D_M_values", None) if isinstance(data, dict) else None
        )

        if D_M_values is not None and len(D_M_values) == len(M):
            D_M = np.array(D_M_values, dtype=float)
        else:
            # Synthetic stabilisation model: D_M -> 4/3 as M -> inf
            D_M = 4.0 / 3.0 + 0.1 / np.sqrt(M)

        # Power-law fit: log(D_M) = b * log(M) + log(a) ----------------------
        try:
            coeffs = np.polyfit(np.log(M), np.log(D_M + 1e-12), 1)
            exponent = coeffs[0]

            # Stabilisation criterion: |exponent| < threshold
            stabilisation_threshold = kwargs.get("stabilisation_threshold", 0.05)
            is_stabilised = abs(exponent) < stabilisation_threshold

            return {
                "status": "completed",
                "exponent": float(exponent),
                "is_stabilised": is_stabilised,
                "stabilisation_threshold": stabilisation_threshold,
                "fit_success": True,
                "theory": "D_M Stabilisation with Modular Resolution",
            }
        except Exception:
            return {
                "status": "fit_failed",
                "theory": "D_M Stabilisation with Modular Resolution",
            }

    def quantify_resource(self, data: Any) -> float:
        """
        Return the modular resolution M as the fundamental resource.

        Parameters
        ----------
        data : dict
            Must contain ``M`` (int or float): the modular resolution.

        Returns
        -------
        float
            The modular resolution M (>= 1).
        """
        M = data.get("M", 1) if isinstance(data, dict) else 1
        return float(max(M, 1))

    def validate_sigma_c(
        self, sigma_c_value: float, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check consistency between D_M, gamma, and the predicted sigma_c.

        In the number-theoretic domain sigma_c is meaningful when:
        1. D_M respects its theoretical bounds (>= 4/3).
        2. The product D * gamma (where D is the effective dimensionality and
           gamma = q/4) is consistent with the reported sigma_c value.

        Parameters
        ----------
        sigma_c_value : float
            The measured or computed sigma_c.
        context : dict
            Must contain ``D_M``, ``q``; may contain ``tolerance``.

        Returns
        -------
        dict
            Validation results with ``is_valid``, ``reason``, and details.
        """
        bounds = self.check_theoretical_bounds(context)

        is_valid = True
        reasons: List[str] = []

        # 1. Check D_M bounds -------------------------------------------------
        if not bounds.get("passes_lower", False):
            is_valid = False
            reasons.append(
                f"D_M={bounds['D_M']:.4f} below lower bound {bounds['lower_bound']:.4f}"
            )
        if not bounds.get("passes_upper", True):
            is_valid = False
            reasons.append(
                f"D_M={bounds['D_M']:.4f} above upper bound {bounds['upper_bound']:.4f}"
            )

        # 2. Consistency: sigma_c should be close to D * gamma ----------------
        D_M = bounds.get("D_M", 0.0)
        gamma = bounds.get("gamma", 0.0)
        predicted = D_M * gamma
        tolerance = context.get("tolerance", 0.15)

        if abs(sigma_c_value - predicted) > tolerance:
            is_valid = False
            reasons.append(
                f"sigma_c={sigma_c_value:.4f} inconsistent with "
                f"D_M*gamma={predicted:.4f} (tol={tolerance})"
            )

        reason = "; ".join(reasons) if reasons else "Within bounds and consistent"

        return {
            "is_valid": is_valid,
            "sigma_c": sigma_c_value,
            "predicted_sigma_c": predicted,
            "reason": reason,
            "bounds": bounds,
        }
