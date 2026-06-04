"""sigma_c_v4 core -- chi_O, trichotomy, stability, faithfulness."""
from sigma_c_v4.core.susceptibility import chi_O, find_interior_maxima
from sigma_c_v4.core.trichotomy import classify, geometric_trichotomy
from sigma_c_v4.core.stability import compute_gamma_O
from sigma_c_v4.core.faithfulness import (
    check_F1,
    check_F2,
    check_F3,
    kl_modal_coefficients,
    FaithfulnessCheck,
)

__all__ = [
    "chi_O",
    "find_interior_maxima",
    "classify",
    "geometric_trichotomy",
    "compute_gamma_O",
    "check_F1",
    "check_F2",
    "check_F3",
    "kl_modal_coefficients",
    "FaithfulnessCheck",
]
