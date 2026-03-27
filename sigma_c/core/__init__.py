"""
Sigma-C Core Module
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from .base import SigmaCAdapter
from .orchestrator import Universe
from .discovery import ObservableDiscovery, MultiScaleAnalysis
from .control import StreamingSigmaC, AdaptiveController
from .classification import MapType
from .contraction import (
    compute_contraction_defect, compute_drift, classify_map,
    v2, odd_part, embedding_depth, single_step_map, cycle_map,
)

__all__ = [
    "Universe",
    "SigmaCAdapter",
    "ObservableDiscovery",
    "MultiScaleAnalysis",
    "StreamingSigmaC",
    "AdaptiveController",
    "MapType",
    "compute_contraction_defect",
    "compute_drift",
    "classify_map",
    "v2",
    "odd_part",
    "embedding_depth",
    "single_step_map",
    "cycle_map",
]
