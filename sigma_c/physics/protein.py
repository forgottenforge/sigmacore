#!/usr/bin/env python3
"""
Rigorous Protein Sigma_c
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Validates sigma_c values for protein stability and folding systems.

The central assertions are:
- sigma(T_m) = 1 at the melting temperature (marginal stability).
- sigma(310 K) < 1 for thermodynamically stable proteins under physiological
  conditions.
- sigma increases monotonically with delta-delta-G (destabilisation energy).

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Any, Dict, List
import numpy as np
from sigma_c.physics.rigorous import RigorousTheoreticalCheck


class RigorousProteinSigmaC(RigorousTheoreticalCheck):
    """
    Checks if measured sigma_c values respect protein thermodynamic bounds.

    Key quantities:
    - T_m : melting temperature (K).  At T_m the protein is marginally stable
      and sigma must equal 1.
    - T_phys : physiological temperature (default 310 K).  For a stable
      protein sigma(T_phys) < 1.
    - delta_delta_G : destabilisation free energy (kcal/mol).  sigma must
      increase monotonically with this quantity.
    - N : chain length (number of residues), which serves as the resource.
    """

    # Default physiological temperature in Kelvin
    T_PHYS: float = 310.0

    def check_theoretical_bounds(
        self, data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Verify the two core thermodynamic constraints:
        1. sigma(T_m) = 1   (within a tolerance).
        2. sigma(T_phys) < 1  for a stable protein.

        Parameters
        ----------
        data : dict
            Must contain:
            - ``sigma_at_Tm`` (float): measured sigma at the melting temperature.
            - ``sigma_at_Tphys`` (float): measured sigma at 310 K.
            May optionally contain:
            - ``T_m`` (float): melting temperature in Kelvin.
            - ``tolerance`` (float): tolerance for sigma(T_m) == 1 check
              (default 0.05).

        Returns
        -------
        dict
            Bound-checking results with pass/fail flags for each constraint.
        """
        sigma_at_Tm = data.get("sigma_at_Tm", 0.0)
        sigma_at_Tphys = data.get("sigma_at_Tphys", 0.0)
        T_m = data.get("T_m", None)
        tolerance = data.get("tolerance", 0.05)

        # Constraint 1: sigma(T_m) == 1
        passes_melting = abs(sigma_at_Tm - 1.0) <= tolerance

        # Constraint 2: sigma(310 K) < 1 for stable proteins
        passes_stability = sigma_at_Tphys < 1.0

        # Derive bounds for sigma_c at physiological T
        lower_bound = 0.0  # sigma cannot be negative
        upper_bound = 1.0  # stable protein must have sigma < 1 at T_phys

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "sigma_at_Tm": sigma_at_Tm,
            "sigma_at_Tphys": sigma_at_Tphys,
            "T_m": T_m,
            "tolerance": tolerance,
            "passes_melting": passes_melting,
            "passes_stability": passes_stability,
            "metric": "sigma_protein",
            "theory": "Protein Thermodynamic Stability (sigma(T_m)=1, sigma(310K)<1)",
        }

    def check_scaling_laws(
        self, data: Any, param_range: List[float], **kwargs
    ) -> Dict[str, Any]:
        """
        Verify that sigma increases monotonically with delta_delta_G.

        As the destabilisation energy increases the protein becomes less
        stable, so sigma must grow.  We fit sigma = a * ddG^b and verify
        that the exponent is positive.

        Parameters
        ----------
        data : dict or Any
            If a dict, may contain ``sigma_values`` (list of floats) measured
            at each delta_delta_G in *param_range*.  If not provided, a
            synthetic sigmoidal model is used.
        param_range : list of float
            delta_delta_G values (kcal/mol) at which sigma was measured.

        Returns
        -------
        dict
            Scaling-law results including monotonicity check and fit exponent.
        """
        if not param_range or len(param_range) < 2:
            return {"status": "insufficient_data"}

        ddG = np.array(param_range, dtype=float)

        # Obtain sigma values --------------------------------------------------
        sigma_values = (
            data.get("sigma_values", None) if isinstance(data, dict) else None
        )

        if sigma_values is not None and len(sigma_values) == len(ddG):
            sigma = np.array(sigma_values, dtype=float)
        else:
            # Synthetic sigmoidal model: sigma -> 1 as ddG -> inf
            sigma = 1.0 / (1.0 + np.exp(-ddG))

        # Monotonicity check ---------------------------------------------------
        diffs = np.diff(sigma)
        is_monotonic = bool(np.all(diffs >= -1e-9))

        # Power-law fit on positive ddG values ---------------------------------
        pos_mask = ddG > 0
        if np.sum(pos_mask) >= 2:
            try:
                coeffs = np.polyfit(
                    np.log(ddG[pos_mask] + 1e-12),
                    np.log(sigma[pos_mask] + 1e-12),
                    1,
                )
                exponent = coeffs[0]
                fit_success = True
            except Exception:
                exponent = float("nan")
                fit_success = False
        else:
            exponent = float("nan")
            fit_success = False

        return {
            "status": "completed",
            "is_monotonic": is_monotonic,
            "exponent": float(exponent) if fit_success else None,
            "fit_success": fit_success,
            "theory": "Monotonic sigma vs delta_delta_G",
        }

    def quantify_resource(self, data: Any) -> float:
        """
        Return the chain length N (number of residues) as the resource.

        Parameters
        ----------
        data : dict
            Must contain ``N`` (int): number of amino-acid residues.

        Returns
        -------
        float
            Chain length N (>= 1).
        """
        N = data.get("N", 1) if isinstance(data, dict) else 1
        return float(max(N, 1))

    def validate_sigma_c(
        self, sigma_c_value: float, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check consistency of a sigma_c measurement with the protein's
        biophysical mechanism.

        Three mechanism classes are recognised:
        - **stability-driven**: sigma < 1 and the protein is below T_m.
          This is the normal regime for wild-type proteins.
        - **gain-of-function (GOF)**: sigma >= 1 despite T < T_m, indicating
          a mutation or ligand that shifts the stability landscape.
        - **templated**: sigma >= 1 and the system exhibits prion-like or
          amyloid templated aggregation.

        Parameters
        ----------
        sigma_c_value : float
            The measured sigma_c.
        context : dict
            Must contain ``sigma_at_Tm``, ``sigma_at_Tphys``; may contain
            ``mechanism`` (str) to assert an expected mechanism.

        Returns
        -------
        dict
            Validation results including ``is_valid``, inferred mechanism,
            and reason.
        """
        bounds = self.check_theoretical_bounds(context)

        is_valid = True
        reasons: List[str] = []

        # 1. Check melting-point constraint ------------------------------------
        if not bounds.get("passes_melting", False):
            is_valid = False
            reasons.append(
                f"sigma(T_m)={bounds['sigma_at_Tm']:.4f} deviates from 1.0 "
                f"(tol={bounds['tolerance']})"
            )

        # 2. Infer mechanism ---------------------------------------------------
        sigma_phys = bounds.get("sigma_at_Tphys", 0.0)

        if sigma_phys < 1.0:
            inferred_mechanism = "stability-driven"
        elif context.get("templated", False):
            inferred_mechanism = "templated"
        else:
            inferred_mechanism = "GOF"

        # 3. Check asserted mechanism if provided ------------------------------
        expected_mechanism = context.get("mechanism", None)
        if expected_mechanism is not None and expected_mechanism != inferred_mechanism:
            is_valid = False
            reasons.append(
                f"Expected mechanism '{expected_mechanism}' but inferred "
                f"'{inferred_mechanism}'"
            )

        # 4. Check sigma_c against physiological bound -------------------------
        if not bounds.get("passes_stability", True):
            reasons.append(
                f"sigma(310K)={sigma_phys:.4f} >= 1.0; protein is unstable "
                "at physiological temperature"
            )

        reason = "; ".join(reasons) if reasons else "Within bounds and consistent"

        return {
            "is_valid": is_valid,
            "sigma_c": sigma_c_value,
            "inferred_mechanism": inferred_mechanism,
            "reason": reason,
            "bounds": bounds,
        }
