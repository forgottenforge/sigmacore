#!/usr/bin/env python3
"""
Sigma-C Base Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Base classes for all domain-specific adapters.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union
from .engine import Engine

class SigmaCAdapter(ABC):
    """
    Abstract base class for all domain adapters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine = Engine()
    
    @abstractmethod
    def get_observable(self, data: Any, **kwargs) -> float:
        """
        Calculate the domain-specific observable from raw data.
        """
        pass
    
    def compute_susceptibility(self, 
                               epsilon: np.ndarray, 
                               observable: np.ndarray,
                               kernel_sigma: float = 0.6) -> Dict[str, Any]:
        """
        Compute susceptibility using the C++ core engine.
        """
        return self.engine.compute_susceptibility(epsilon, observable, kernel_sigma)

class SigmaCExperiment(ABC):
    """
    Base class for standardized experiments.
    """
    def __init__(self, adapter: SigmaCAdapter):
        self.adapter = adapter
        self.results = {}
        
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        pass
