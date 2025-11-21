"""
Sigma-C PennyLane Plugin
========================
Copyright (c) 2025 ForgottenForge.xyz

PennyLane device plugin for criticality-aware quantum computing.
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np

try:
    import pennylane as qml
    from pennylane.devices import DefaultQubit
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False
    DefaultQubit = object

from ..core.engine import Engine


class SigmaCDevice(DefaultQubit if _HAS_PENNYLANE else object):
    """
    PennyLane device with built-in criticality tracking.
    
    Usage:
        import pennylane as qml
        from sigma_c.plugins.pennylane import SigmaCDevice
        
        dev = qml.device('sigma_c.simulator', wires=4, critical_point=0.1)
        
        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            return qml.expval(qml.PauliZ(0))
    """
    
    name = "Sigma-C Simulator"
    short_name = "sigma_c.simulator"
    pennylane_requires = ">=0.30.0"
    version = "2.0.0"
    author = "ForgottenForge"
    
    def __init__(self, wires, *, shots=None, critical_point=0.1, **kwargs):
        if not _HAS_PENNYLANE:
            raise ImportError("PennyLane not installed. Run: pip install pennylane")
        
        super().__init__(wires, shots=shots, **kwargs)
        self.critical_point = critical_point
        self.sigma_c_history = []
        self.engine = Engine()
    
    def apply(self, operations, **kwargs):
        """
        Apply operations with criticality tracking.
        """
        # Track operation complexity
        self._track_complexity(operations)
        
        # Call parent apply
        return super().apply(operations, **kwargs)
    
    def _track_complexity(self, operations):
        """
        Track circuit complexity as proxy for criticality.
        """
        # Count gates and parameters
        n_gates = len(operations)
        n_params = sum(len(op.parameters) for op in operations if hasattr(op, 'parameters'))
        
        # Simple criticality metric: parameter/gate ratio
        if n_gates > 0:
            sigma_c = n_params / (n_gates + 1e-9)
            self.sigma_c_history.append(sigma_c)
    
    def get_criticality_report(self) -> Dict[str, float]:
        """
        Get criticality statistics.
        
        Returns:
            Dictionary with sigma_c metrics
        """
        if not self.sigma_c_history:
            return {'mean_sigma_c': 0.0, 'samples': 0}
        
        history = np.array(self.sigma_c_history)
        return {
            'mean_sigma_c': float(np.mean(history)),
            'std_sigma_c': float(np.std(history)),
            'samples': len(history)
        }


# Register device with PennyLane
if _HAS_PENNYLANE:
    try:
        qml.device._device_classes['sigma_c.simulator'] = SigmaCDevice
    except:
        pass  # Graceful degradation if registration fails
