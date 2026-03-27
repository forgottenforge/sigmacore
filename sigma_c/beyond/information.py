#!/usr/bin/env python3
"""
Information-Theoretic Module
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Connects contraction geometry to information theory:
- Information loss from non-injectivity
- Landauer principle connection
- Entropy production rates

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Dict, Any

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
LN2 = np.log(2)


def bits_lost(D: float) -> float:
    """
    Information lost per application of a non-injective map.

    For a map with contraction defect D, each application
    erases log2(D) bits on average.

    Args:
        D: Contraction defect (|domain|/|image|), must be >= 1

    Returns:
        Information loss in bits per step
    """
    if D < 1.0:
        raise ValueError(f"D must be >= 1.0, got {D}")
    return np.log2(D)


def landauer_cost(D: float, T: float = 300.0) -> float:
    """
    Minimum thermodynamic cost of information erasure (Landauer principle).

    E_min = k_B * T * ln(2) * log2(D)

    Args:
        D: Contraction defect
        T: Temperature in Kelvin (default: 300K = room temperature)

    Returns:
        Minimum energy cost in Joules per step
    """
    return K_B * T * LN2 * bits_lost(D)


def entropy_production_rate(D: float, steps_per_second: float,
                            T: float = 300.0) -> float:
    """
    Entropy production rate from iterated non-injective operations.

    dS/dt = k_B * ln(2) * log2(D) * rate

    Args:
        D: Contraction defect
        steps_per_second: Number of map applications per second
        T: Temperature in Kelvin

    Returns:
        Entropy production rate in J/(K*s)
    """
    return K_B * LN2 * bits_lost(D) * steps_per_second


def information_summary(D: float, gamma: float,
                        T: float = 300.0,
                        steps_per_second: float = 1.0) -> Dict[str, Any]:
    """
    Complete information-theoretic summary of a contraction system.

    Args:
        D: Contraction defect
        gamma: Drift
        T: Temperature (K)
        steps_per_second: Application rate

    Returns:
        Dict with all information-theoretic quantities
    """
    bl = bits_lost(D)
    lc = landauer_cost(D, T)
    epr = entropy_production_rate(D, steps_per_second, T)

    net_contraction = D > 1 and gamma < 1
    sigma_product = D * gamma

    return {
        'bits_lost_per_step': bl,
        'landauer_cost_J': lc,
        'landauer_cost_eV': lc / 1.602176634e-19,
        'entropy_production_rate': epr,
        'D': D,
        'gamma': gamma,
        'sigma_product': sigma_product,
        'net_contraction': net_contraction,
        'interpretation': (
            f'Each step erases {bl:.3f} bits. '
            f'Minimum energy cost: {lc:.2e} J at {T}K. '
            f'{"Net contraction (convergent)." if net_contraction else "No net contraction."}'
        ),
    }
