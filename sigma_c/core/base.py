#!/usr/bin/env python3
"""
Sigma-C Base Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Base classes for all domain-specific adapters.

For commercial licensing without AGPL-3.0 obligations, contact:
nfo@forgottenforge.xyz

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from .engine import Engine

class SigmaCAdapter(ABC):
    """
    Abstract base class for all domain adapters.
    
    Version 1.0.0: Basic susceptibility computation
    Version 1.1.0: Added universal diagnostics system
    Version 2.1.0: Full Feature Implementation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine = Engine()
        self._version = "2.1.0"
    
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
    
    # ========== v1.1.0: Universal Diagnostics System ==========
    
    def diagnose(self, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Universal diagnostics entry point (v1.1.0).
        
        Analyzes data and provides:
        - Issue detection
        - Recommendations
        - Auto-fix suggestions
        
        Returns:
            {
                'status': 'ok' | 'warning' | 'error',
                'issues': List[str],
                'recommendations': List[str],
                'auto_fix': Optional[Callable],
                'details': Dict[str, Any]
            }
        """
        # Call domain-specific diagnostics
        # Pass data as positional only if provided, to avoid conflicts
        # with domain-specific parameter names (e.g., price_data, catalog)
        if data is not None:
            domain_diag = self._domain_specific_diagnose(data, **kwargs)
        else:
            domain_diag = self._domain_specific_diagnose(**kwargs)
        
        return {
            'status': domain_diag.get('status', 'ok'),
            'issues': domain_diag.get('issues', []),
            'recommendations': domain_diag.get('recommendations', []),
            'auto_fix': domain_diag.get('auto_fix'),
            'details': domain_diag.get('details', {})
        }
    
    def auto_search(self, data: Any = None, param_ranges: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Systematically search parameter space (v1.1.0).
        
        Args:
            data: Input data
            param_ranges: Dict of parameter ranges to search
            **kwargs: Domain-specific options
        
        Returns:
            {
                'best_params': Dict[str, Any],
                'all_results': List[Dict],
                'convergence_data': Dict,
                'recommendation': str
            }
        """
        # Domain-specific implementation
        if data is not None:
            return self._domain_specific_auto_search(data, param_ranges, **kwargs)
        else:
            return self._domain_specific_auto_search(param_ranges=param_ranges, **kwargs)
    
    def validate_techniques(self, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Check domain-specific technique requirements (v1.1.0).
        
        Returns:
            {
                'all_passed': bool,
                'checks': Dict[str, bool],
                'failed_checks': List[str],
                'details': Dict[str, Any]
            }
        """
        if data is not None:
            checks = self._domain_specific_validate(data, **kwargs)
        else:
            checks = self._domain_specific_validate(**kwargs)
        failed = [k for k, v in checks.items() if not v]
        
        return {
            'all_passed': len(failed) == 0,
            'checks': checks,
            'failed_checks': failed,
            'details': {}
        }
    
    def explain(self, result: Dict[str, Any], **kwargs) -> str:
        """
        Generate human-readable explanation of results (v1.1.0).
        
        Args:
            result: Result dictionary from compute_susceptibility or other methods
        
        Returns:
            Markdown-formatted explanation string
        """
        return self._domain_specific_explain(result, **kwargs)
    
    # ========== Domain-Specific Hooks (Override in subclasses) ==========
    
    def _domain_specific_diagnose(self, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Override in subclass for domain-specific diagnostics."""
        return {
            'status': 'ok',
            'issues': [],
            'recommendations': [],
            'auto_fix': None,
            'details': {}
        }

    def _domain_specific_auto_search(self, data: Any = None, param_ranges: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Override in subclass for domain-specific parameter search."""
        return {
            'best_params': {},
            'all_results': [],
            'convergence_data': {},
            'recommendation': 'No auto-search implemented for this domain'
        }
    
    def _domain_specific_validate(self, data: Any = None, **kwargs) -> Dict[str, bool]:
        """Override in subclass for domain-specific validation."""
        return {'basic_validation': True}
    
    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """Override in subclass for domain-specific explanations."""
        sigma_c = result.get('sigma_c', 'N/A')
        kappa = result.get('kappa', 'N/A')
        
        explanation = f"""
# Sigma-C Analysis Results

**Critical Scale (sigma_c):** {sigma_c}
**Criticality Score (kappa):** {kappa}

## Interpretation
- sigma_c indicates the scale where the system transitions
- Higher kappa values suggest stronger critical behavior
- kappa > 10 typically indicates significant criticality

For domain-specific interpretation, see documentation.
"""
        return explanation.strip()

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
