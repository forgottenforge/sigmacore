#!/usr/bin/env python3
"""
Type Classification Module
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Implements the four-type classification of non-injective operations:
- Type D (Dissipative): D > 1, gamma classifies convergence/divergence
- Type O (Oversaturated): Growing pre-image count
- Type S (Symmetric): D = 1, bijective constraint (symmetry)
- Type R (Reversible): D = 1, bijective preservation (conservation)

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from enum import Enum
from typing import Dict, Any, Optional
import numpy as np


class MapType(Enum):
    """Classification of operations by injectivity structure."""
    DISSIPATIVE = "D"       # D > 1, non-injective contraction
    OVERSATURATED = "O"     # Growing pre-image (redundancy)
    SYMMETRIC = "S"         # D = 1, bijective with symmetry constraint
    REVERSIBLE = "R"        # D = 1, bijective preservation


class TypeDResult:
    """Result of Type D (Dissipative) analysis."""

    def __init__(self, D: float, gamma: float, has_cycles: bool = False,
                 prediction: str = '', details: str = ''):
        self.D = D
        self.gamma = gamma
        self.has_cycles = has_cycles
        self.prediction = prediction
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'D',
            'D': self.D,
            'gamma': self.gamma,
            'has_cycles': self.has_cycles,
            'prediction': self.prediction,
            'details': self.details,
        }


class TypeOResult:
    """Result of Type O (Oversaturated) analysis."""

    def __init__(self, oversaturation_ratio: float, min_count: int,
                 reference_prediction: float, details: str = ''):
        self.oversaturation_ratio = oversaturation_ratio
        self.min_count = min_count
        self.reference_prediction = reference_prediction
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'O',
            'oversaturation_ratio': self.oversaturation_ratio,
            'min_count': self.min_count,
            'reference_prediction': self.reference_prediction,
            'details': self.details,
        }


class TypeSResult:
    """Result of Type S (Symmetric) analysis."""

    def __init__(self, symmetry_deviation: float, constraint_tightness: float,
                 details: str = ''):
        self.symmetry_deviation = symmetry_deviation
        self.constraint_tightness = constraint_tightness
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'S',
            'symmetry_deviation': self.symmetry_deviation,
            'constraint_tightness': self.constraint_tightness,
            'details': self.details,
        }


class TypeRResult:
    """Result of Type R (Reversible) analysis."""

    def __init__(self, is_bijective: bool, orbits_preserved: bool,
                 conserved_quantity: Optional[str] = None, details: str = ''):
        self.is_bijective = is_bijective
        self.orbits_preserved = orbits_preserved
        self.conserved_quantity = conserved_quantity
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'R',
            'is_bijective': self.is_bijective,
            'orbits_preserved': self.orbits_preserved,
            'conserved_quantity': self.conserved_quantity,
            'details': self.details,
        }


def classify_operation(D: float, gamma: Optional[float] = None,
                       has_cycles: bool = False,
                       is_bijective: Optional[bool] = None,
                       has_symmetry: bool = False,
                       has_growing_preimage: bool = False) -> MapType:
    """
    Classify an operation into one of the four types.

    Args:
        D: Contraction defect (|domain|/|image|)
        gamma: Drift (average multiplicative growth), if applicable
        has_cycles: Whether non-trivial cycles exist
        is_bijective: Override: True forces D=1 interpretation
        has_symmetry: Whether the map has known symmetry constraints
        has_growing_preimage: Whether pre-image counts grow with scale

    Returns:
        MapType enum value
    """
    if is_bijective or D == 1.0:
        if has_symmetry:
            return MapType.SYMMETRIC
        return MapType.REVERSIBLE

    if D > 1.0:
        if has_growing_preimage:
            return MapType.OVERSATURATED
        return MapType.DISSIPATIVE

    # D < 1 shouldn't happen for well-defined maps, but handle gracefully
    return MapType.REVERSIBLE


def analyze_type_d(D: float, gamma: float,
                   has_cycles: bool = False) -> TypeDResult:
    """
    Full Type D analysis with prediction.

    Args:
        D: Contraction defect
        gamma: Drift
        has_cycles: Whether cycles are known

    Returns:
        TypeDResult with prediction
    """
    if gamma < 1.0:
        if has_cycles:
            prediction = 'convergent_to_cycles'
            details = (f'D={D:.4f}>1, gamma={gamma:.4f}<1 with known cycles. '
                       f'Trajectories converge to cycles.')
        else:
            prediction = 'convergent'
            details = (f'D={D:.4f}>1, gamma={gamma:.4f}<1, no cycles known. '
                       f'Trajectories converge to fixed point.')
    elif gamma > 1.0:
        prediction = 'divergent'
        details = (f'D={D:.4f}>1, gamma={gamma:.4f}>1. '
                   f'Trajectories diverge (values escape to infinity).')
    else:
        prediction = 'critical'
        details = f'D={D:.4f}>1, gamma={gamma:.4f}~1. At critical threshold.'

    return TypeDResult(D, gamma, has_cycles, prediction, details)


def analyze_type_o(counts: np.ndarray, predictions: np.ndarray,
                   n_range: Optional[np.ndarray] = None) -> TypeOResult:
    """
    Type O (Oversaturated) analysis.
    Computes oversaturation ratio O_M = min(count/prediction).

    Args:
        counts: Observed representation counts (e.g., Goldbach r(n))
        predictions: Theoretical predictions (e.g., Hardy-Littlewood)
        n_range: Optional range of n values

    Returns:
        TypeOResult
    """
    valid = predictions > 0
    ratios = np.zeros_like(counts, dtype=float)
    ratios[valid] = counts[valid] / predictions[valid]

    O_M = float(np.min(ratios[valid])) if np.any(valid) else 0.0
    min_count = int(np.min(counts))
    mean_prediction = float(np.mean(predictions[valid])) if np.any(valid) else 0.0

    details = (f'Oversaturation ratio O_M={O_M:.4f}. '
               f'Minimum count={min_count}. '
               f'Mean prediction={mean_prediction:.2f}.')

    return TypeOResult(O_M, min_count, mean_prediction, details)


def analyze_type_s(deviations: np.ndarray,
                   reference_variance: float = 0.178) -> TypeSResult:
    """
    Type S (Symmetric) analysis.
    Measures symmetry deviation and constraint tightness.

    Args:
        deviations: Array of deviations from symmetry axis
        reference_variance: Reference variance (e.g., GUE = 0.178)

    Returns:
        TypeSResult
    """
    S_M = float(np.max(np.abs(deviations)))
    observed_var = float(np.var(deviations))
    C_M = 1.0 - abs(observed_var - reference_variance)
    C_M = max(0.0, min(1.0, C_M))

    details = (f'Max symmetry deviation S_M={S_M:.6f}. '
               f'Constraint tightness C_M={C_M:.4f} '
               f'(vs reference variance {reference_variance:.3f}).')

    return TypeSResult(S_M, C_M, details)


def analyze_type_r(f_values: np.ndarray, g_orbits: np.ndarray) -> TypeRResult:
    """
    Type R (Reversible) analysis.
    Checks if f preserves g-orbits (Noether-type conservation).

    Args:
        f_values: Image of f applied to domain
        g_orbits: Orbit labels under symmetry operation g

    Returns:
        TypeRResult
    """
    domain_size = len(f_values)
    image_size = len(set(f_values))
    is_bijective = (domain_size == image_size)

    # Check orbit preservation
    orbits_preserved = True
    if is_bijective and len(g_orbits) == domain_size:
        for i in range(domain_size):
            if g_orbits[i] != g_orbits[f_values[i] % domain_size]:
                orbits_preserved = False
                break

    conserved = 'orbit_size' if orbits_preserved else None
    details = (f'Bijective: {is_bijective}. '
               f'Orbits preserved: {orbits_preserved}.')

    return TypeRResult(is_bijective, orbits_preserved, conserved, details)
