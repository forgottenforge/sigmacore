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

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, domain: str, adapter_cls):
        cls._registry[domain] = adapter_cls

    @classmethod
    def create(cls, domain: str, **kwargs) -> SigmaCAdapter:
        if domain not in cls._registry:
            _lazy_imports = {
                'quantum': ('.quantum', 'QuantumAdapter'),
                'gpu': ('.gpu', 'GPUAdapter'),
                'financial': ('.financial', 'FinancialAdapter'),
                'ml': ('.ml', 'MLAdapter'),
                'climate': ('.climate', 'ClimateAdapter'),
                'seismic': ('.seismic', 'SeismicAdapter'),
                'magnetic': ('.magnetic', 'MagneticAdapter'),
                'edge': ('.edge', 'EdgeAdapter'),
                'llm_cost': ('.llm_cost', 'LLMCostAdapter'),
            }
            if domain not in _lazy_imports:
                raise ValueError(f"Unknown domain: {domain}. Available: {list(_lazy_imports.keys())}")

            import importlib
            module_path, class_name = _lazy_imports[domain]
            module = importlib.import_module(module_path, package='sigma_c.adapters')
            adapter_cls = getattr(module, class_name)
            cls.register(domain, adapter_cls)

        return cls._registry[domain](config=kwargs)
