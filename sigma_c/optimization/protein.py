#!/usr/bin/env python3
"""
Protein Intervention Optimizer
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Finds minimal D*gamma < 1 interventions that preserve native contacts while
driving the protein toward the sigma = 1 stability threshold.

The two optimisation axes are:
- D-axis: number of conformationally restricted residues.
- gamma-axis: energetic stabilisation boost (kcal/mol).

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Any, Dict, List, Optional
import logging
import numpy as np

from sigma_c.optimization.universal import UniversalOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class ProteinInterventionOptimizer(UniversalOptimizer):
    """
    Optimiser for protein stabilisation interventions.

    Balances the fraction of preserved native contacts (Q_nat) against the
    distance from the sigma = 1 instability threshold.  The goal is to find
    the minimal combination of conformational restriction (D) and energetic
    stabilisation (gamma) such that D * gamma < 1 while maximising Q_nat.

    Parameters
    ----------
    performance_weight : float
        Weight for the native-contact performance term (default 0.7).
    stability_weight : float
        Weight for the sigma-distance stability term (default 0.3).
    sigma_target : float
        Target sigma value representing the stability boundary (default 1.0).
    """

    def __init__(
        self,
        performance_weight: float = 0.7,
        stability_weight: float = 0.3,
        sigma_target: float = 1.0,
    ):
        super().__init__(
            performance_weight=performance_weight,
            stability_weight=stability_weight,
        )
        self.sigma_target = sigma_target

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _evaluate_performance(
        self, system: Any, params: Dict[str, Any]
    ) -> float:
        """
        Measure the fraction of native contacts preserved (Q_nat).

        The *system* is expected to be a dict (or object) with a method or key
        ``compute_Q_nat(params) -> float`` returning a value in [0, 1].

        If unavailable, a simple heuristic model is used:
            Q_nat = 1 - 0.02 * D
        where D is the number of conformationally restricted residues.  More
        restricted residues slightly reduce the native-contact fraction.

        Returns
        -------
        float
            Native-contact fraction Q_nat in [0, 1].  Higher is better.
        """
        if isinstance(system, dict) and callable(system.get("compute_Q_nat")):
            return float(system["compute_Q_nat"](params))

        # Heuristic fallback
        D = params.get("D", 0)
        Q_nat = max(0.0, 1.0 - 0.02 * D)
        return Q_nat

    def _evaluate_stability(
        self, system: Any, params: Dict[str, Any]
    ) -> float:
        """
        Measure the distance from the sigma = 1 instability threshold.

        The stability score is defined as:
            stability = 1 - |sigma - sigma_target|
        so a protein sitting exactly at the target scores 1.0.

        The *system* may provide ``compute_sigma(params) -> float``.
        Otherwise a heuristic model is used:
            sigma = sigma_base - gamma * 0.1
        where gamma is the energetic stabilisation boost.

        Returns
        -------
        float
            Stability score in [0, 1].  Higher means closer to the target.
        """
        if isinstance(system, dict) and callable(system.get("compute_sigma")):
            sigma = float(system["compute_sigma"](params))
        else:
            sigma_base = (
                system.get("sigma_base", 0.8) if isinstance(system, dict) else 0.8
            )
            gamma = params.get("gamma", 0.0)
            sigma = sigma_base - gamma * 0.1

        distance = abs(sigma - self.sigma_target)
        return max(0.0, 1.0 - distance)

    def _apply_params(
        self, system: Any, params: Dict[str, Any]
    ) -> Any:
        """
        Apply intervention parameters to the protein system.

        Returns a shallow copy of *system* (if dict) with the intervention
        parameters merged in, so the original system is not mutated.
        """
        if isinstance(system, dict):
            updated = dict(system)
            updated["intervention"] = dict(params)
            return updated

        # For non-dict systems, try to set attributes
        try:
            import copy

            updated = copy.copy(system)
            for key, value in params.items():
                setattr(updated, key, value)
            return updated
        except Exception:
            return system

    # ------------------------------------------------------------------
    # Domain-specific optimisation
    # ------------------------------------------------------------------

    def optimize_intervention(
        self,
        system: Any,
        D_values: List[int],
        gamma_values: List[float],
        strategy: str = "brute_force",
        callbacks: Optional[List[Any]] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Find the minimal D * gamma < 1 combination that maximises the
        composite score (native contacts + stability).

        Parameters
        ----------
        system : Any
            Protein system representation (dict or object).
        D_values : list of int
            Candidate values for D (number of conformationally restricted
            residues).
        gamma_values : list of float
            Candidate values for gamma (energetic stabilisation boost,
            kcal/mol).
        strategy : str
            Optimisation strategy forwarded to ``optimize()``
            (default ``'brute_force'``).
        callbacks : list, optional
            Callback objects forwarded to ``optimize()``.

        Returns
        -------
        OptimizationResult
            The best intervention satisfying D * gamma < 1, or the overall
            best if no combination satisfies the constraint.
        """
        # Filter parameter space to D*gamma < 1 combinations ------------------
        feasible_D: List[int] = []
        feasible_gamma: List[float] = []

        for d in D_values:
            for g in gamma_values:
                if d * g < 1.0:
                    if d not in feasible_D:
                        feasible_D.append(d)
                    if g not in feasible_gamma:
                        feasible_gamma.append(g)

        # Fall back to full space if nothing is feasible
        if not feasible_D or not feasible_gamma:
            logger.warning(
                "No D*gamma < 1 combinations found; using full parameter space."
            )
            feasible_D = list(D_values)
            feasible_gamma = list(gamma_values)

        param_space: Dict[str, List[Any]] = {
            "D": sorted(set(feasible_D)),
            "gamma": sorted(set(feasible_gamma)),
        }

        # Delegate to the base-class optimize method ---------------------------
        result = self.optimize(
            system,
            param_space,
            strategy=strategy,
            callbacks=callbacks,
            **kwargs,
        )

        # Post-filter: if the best result violates D*gamma < 1, scan history
        # for the best *feasible* result.
        best_D = result.optimal_params.get("D", 0)
        best_gamma = result.optimal_params.get("gamma", 0.0)

        if best_D * best_gamma >= 1.0:
            logger.info(
                "Best overall result violates D*gamma < 1; scanning for "
                "feasible alternative."
            )
            feasible_best_score = -float("inf")
            feasible_best_entry = None

            for entry in self.history:
                d = entry["params"].get("D", 0)
                g = entry["params"].get("gamma", 0.0)
                if d * g < 1.0 and entry["score"] > feasible_best_score:
                    feasible_best_score = entry["score"]
                    feasible_best_entry = entry

            if feasible_best_entry is not None:
                result = OptimizationResult(
                    optimal_params=feasible_best_entry["params"],
                    score=feasible_best_entry["score"],
                    history=self.history,
                    sigma_c_before=result.sigma_c_before,
                    sigma_c_after=feasible_best_entry["stability"],
                    performance_metric_name="Q_nat",
                    performance_before=result.performance_before,
                    performance_after=feasible_best_entry["performance"],
                    strategy_used=result.strategy_used,
                )

        return result
