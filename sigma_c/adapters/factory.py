#!/usr/bin/env python3
"""
Sigma-C Adapter Factory
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Factory for creating and registering domain adapters.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from typing import Dict, Any
from ..core.base import SigmaCAdapter

class AdapterFactory:
    """
    Factory to create domain-specific adapters.
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, domain: str, adapter_cls):
        cls._registry[domain] = adapter_cls
        
    @classmethod
    def create(cls, domain: str, **kwargs) -> SigmaCAdapter:
        if domain not in cls._registry:
            # Lazy import to avoid circular dependencies and unused imports
            if domain == 'quantum':
                from .quantum import QuantumAdapter
                cls.register('quantum', QuantumAdapter)
            elif domain == 'gpu':
                from .gpu import GPUAdapter
                cls.register('gpu', GPUAdapter)
            elif domain == 'financial':
                from .financial import FinancialAdapter
                cls.register('financial', FinancialAdapter)
            elif domain == 'climate':
                from .climate import ClimateAdapter
                cls.register('climate', ClimateAdapter)
            elif domain == 'seismic':
                from .seismic import SeismicAdapter
                cls.register('seismic', SeismicAdapter)
            elif domain == 'magnetic':
                from .magnetic import MagneticAdapter
                cls.register('magnetic', MagneticAdapter)
            else:
                raise ValueError(f"Unknown domain: {domain}")
                
        return cls._registry[domain](config=kwargs)
