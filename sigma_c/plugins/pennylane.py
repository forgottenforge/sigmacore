"""
Sigma-C PennyLane Plugin
========================
Copyright (c) 2025 ForgottenForge.xyz

PennyLane device plugin for criticality-aware quantum computing.
Tracks circuit complexity as a proxy for noise sensitivity during
variational quantum algorithm execution.

Requires: pip install pennylane

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Dict, Any
import numpy as np

try:
    import pennylane as qml
    from pennylane.devices import DefaultQubit
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False
    qml = None
    DefaultQubit = None

try:
    from ..core.engine import Engine
    _HAS_ENGINE = True
except Exception:
    _HAS_ENGINE = False


class SigmaCDevice:
    """
    PennyLane device wrapper with built-in criticality tracking.

    Wraps DefaultQubit and monitors circuit executions to estimate
    how close the circuit's parameter regime is to a noise-induced
    phase transition.

    Usage (with PennyLane installed):
        import pennylane as qml
        from sigma_c.plugins.pennylane import SigmaCDevice

        dev = SigmaCDevice(wires=4)

        @qml.qnode(dev.device)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = circuit([0.5])
        dev.record_execution(n_gates=2, n_params=1)
        print(dev.get_criticality_report())

    Usage (standalone, without PennyLane):
        dev = SigmaCDevice(wires=2)
        dev.record_execution(n_gates=10, n_params=4)
        dev.record_execution(n_gates=20, n_params=8)
        print(dev.get_criticality_report())
    """

    name = "Sigma-C Simulator"
    short_name = "sigma_c.simulator"
    version = "2.1.0"

    def __init__(self, wires: int = 2, shots: int = None, **kwargs):
        self.wires = wires
        self.shots = shots
        self.sigma_c_history = []
        self._engine = Engine() if _HAS_ENGINE else None

        # Create underlying PennyLane device if available
        self.device = None
        if _HAS_PENNYLANE:
            self.device = qml.device('default.qubit', wires=wires, shots=shots)

    def record_execution(self, n_gates: int, n_params: int):
        """
        Record a circuit execution for criticality tracking.

        Args:
            n_gates: Number of gates in the circuit.
            n_params: Number of variational parameters.
        """
        if n_gates > 0:
            sigma_c = float(np.clip(n_params / n_gates, 0, 1))
            self.sigma_c_history.append(sigma_c)

    def get_criticality_report(self) -> Dict[str, Any]:
        """
        Get criticality statistics across recorded executions.

        Returns:
            Dictionary with sigma_c statistics.
        """
        if not self.sigma_c_history:
            return {'mean_sigma_c': 0.0, 'std_sigma_c': 0.0, 'samples': 0}

        history = np.array(self.sigma_c_history)
        return {
            'mean_sigma_c': float(np.mean(history)),
            'std_sigma_c': float(np.std(history)),
            'min_sigma_c': float(np.min(history)),
            'max_sigma_c': float(np.max(history)),
            'samples': len(history),
        }

    def analyze_noise_threshold(self, n_gates: int, n_qubits: int = None) -> Dict[str, float]:
        """
        Estimate the critical noise threshold for a circuit.

        Args:
            n_gates: Total number of gates.
            n_qubits: Number of qubits (defaults to self.wires).

        Returns:
            Dictionary with estimated sigma_c and interpretation.
        """
        if n_qubits is None:
            n_qubits = self.wires

        if n_gates > 0:
            sigma_c = 1.0 - 0.5 ** (1.0 / n_gates)
        else:
            sigma_c = 1.0

        return {
            'sigma_c': float(sigma_c),
            'n_gates': n_gates,
            'n_qubits': n_qubits,
            'interpretation': (
                f'Gate error must stay below {sigma_c:.2%} for this '
                f'{n_gates}-gate circuit to maintain >50% success rate.'
            ),
        }


# Register device with PennyLane if available.
if _HAS_PENNYLANE:
    try:
        if hasattr(qml, 'plugin_devices'):
            qml.plugin_devices['sigma_c.simulator'] = SigmaCDevice
        elif hasattr(qml, '_device_registry'):
            qml._device_registry['sigma_c.simulator'] = SigmaCDevice
    except Exception:
        pass
