#!/usr/bin/env python3
"""
Sigma-C Core Orchestrator
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

The unified entry point for the Sigma-C framework.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from ..adapters.factory import AdapterFactory
from ..core.base import SigmaCAdapter

class Universe:
    """
    The unified entry point for the Sigma-C framework.
    """
    
    @staticmethod
    def quantum(device: str = 'simulator', **kwargs) -> SigmaCAdapter:
        """Create a Quantum adapter."""
        return AdapterFactory.create('quantum', device=device, **kwargs)
        
    @staticmethod
    def gpu(**kwargs) -> SigmaCAdapter:
        """Create a GPU adapter."""
        return AdapterFactory.create('gpu', **kwargs)
        
    @staticmethod
    def finance(**kwargs) -> SigmaCAdapter:
        """Create a Financial adapter."""
        return AdapterFactory.create('financial', **kwargs)
        
    @staticmethod
    def climate(**kwargs) -> SigmaCAdapter:
        """Create a Climate adapter."""
        return AdapterFactory.create('climate', **kwargs)
        
    @staticmethod
    def seismic(**kwargs) -> SigmaCAdapter:
        """Create a Seismic adapter."""
        return AdapterFactory.create('seismic', **kwargs)

    @staticmethod
    def magnetic(**kwargs) -> SigmaCAdapter:
        """Create a Magnetic adapter."""
        return AdapterFactory.create('magnetic', **kwargs)
        
    @staticmethod
    def analyze(data: Any, domain: str = 'auto', **kwargs) -> Dict[str, Any]:
        """
        Universal analysis entry point.
        """
        if domain == 'auto':
            domain = Universe._detect_domain(data)
            
        adapter = AdapterFactory.create(domain, **kwargs)
        
        # Generic analysis flow
        if domain == 'quantum':
            # Expects circuit or counts
            return adapter.get_observable(data, **kwargs)
        elif domain == 'gpu':
            # Expects performance metrics
            return adapter.get_observable(data, **kwargs)
        elif domain == 'financial':
            # Expects price series
            return adapter.get_observable(data, **kwargs)
        elif domain == 'climate':
            # Expects dataframe with lat/lon/value
            return adapter.get_observable(data, **kwargs)
        elif domain == 'seismic':
            # Expects catalog dataframe
            stress = adapter.compute_stress_proxy(data)
            return adapter.get_observable(stress, **kwargs)
            
        return {}

    @staticmethod
    def _detect_domain(data: Any) -> str:
        """
        Auto-detect domain based on data structure.
        """
        if isinstance(data, dict):
            # Counts dict -> Quantum
            if all(isinstance(k, str) and all(c in '01' for c in k) for k in data.keys()):
                return 'quantum'
                
        if isinstance(data, pd.DataFrame):
            cols = set(data.columns.str.lower())
            if {'lat', 'lon', 'value'}.issubset(cols):
                return 'climate'
            if {'latitude', 'longitude', 'mag', 'time'}.issubset(cols):
                return 'seismic'
            if {'close', 'volume'}.issubset(cols) or 'return' in cols:
                return 'financial'
                
        if isinstance(data, (np.ndarray, list)):
            # Array data is ambiguous, default to GPU/Generic
            return 'gpu'
            
        return 'gpu' # Default fallback
