"""
Sigma-C Qiskit Connector
========================
Copyright (c) 2025 ForgottenForge.xyz

Integrates Sigma-C criticality analysis with IBM Qiskit.
Analyzes quantum circuits for noise sensitivity by sweeping
depolarizing error rates and finding the critical threshold.

Requires: pip install qiskit

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from qiskit import QuantumCircuit
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False
    QuantumCircuit = None

from ..core.engine import Engine


class QiskitSigmaC:
    """
    Qiskit integration for automatic criticality analysis.

    Sweeps noise levels on a given circuit and uses susceptibility
    analysis to find the critical noise threshold (sigma_c).

    Usage:
        from sigma_c.connectors.qiskit import QiskitSigmaC

        analyzer = QiskitSigmaC()
        result = analyzer.analyze_circuit(circuit, noise_levels=np.linspace(0, 0.2, 20))
        print(f"Critical noise: {result['sigma_c']:.4f}")
    """

    def __init__(self):
        self._engine = Engine()

    def analyze_circuit(self, circuit, noise_levels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze a Qiskit circuit's noise sensitivity.

        Simulates the circuit at each noise level and measures how the
        success probability decays. The critical point sigma_c is where
        the susceptibility (derivative) peaks.

        Args:
            circuit: Qiskit QuantumCircuit instance.
            noise_levels: Array of depolarizing error rates to sweep.

        Returns:
            Dictionary with sigma_c, kappa, chi_max, and per-level data.
        """
        if not _HAS_QISKIT:
            raise ImportError("Qiskit not installed. Run: pip install qiskit")

        if noise_levels is None:
            noise_levels = np.linspace(0.0, 0.25, 20)

        n_qubits = circuit.num_qubits
        depth = circuit.depth()
        n_gates = sum(circuit.count_ops().values())

        # Simulate success probability at each noise level
        observables = np.zeros(len(noise_levels))
        for i, eps in enumerate(noise_levels):
            observables[i] = self._simulate_noisy_circuit(n_qubits, depth, n_gates, eps)

        # Compute susceptibility via engine
        result = self._engine.compute_susceptibility(noise_levels, observables)

        return {
            'sigma_c': float(result['sigma_c']),
            'kappa': float(result['kappa']),
            'chi_max': float(result['chi_max']),
            'epsilon': noise_levels.tolist(),
            'observable': observables.tolist(),
            'n_qubits': n_qubits,
            'depth': depth,
            'n_gates': n_gates,
            'framework': 'qiskit',
        }

    def optimize_for_noise(self, circuit, target_noise: float = 0.01) -> Dict[str, Any]:
        """
        Analyze whether a circuit can tolerate the target noise level.

        Args:
            circuit: Qiskit QuantumCircuit instance.
            target_noise: Expected hardware noise level.

        Returns:
            Dictionary with analysis and recommendation.
        """
        if not _HAS_QISKIT:
            raise ImportError("Qiskit not installed. Run: pip install qiskit")

        analysis = self.analyze_circuit(circuit)
        sigma_c = analysis['sigma_c']

        margin = sigma_c - target_noise
        if margin > 0.05:
            status = 'safe'
            recommendation = 'Circuit tolerates the target noise level.'
        elif margin > 0:
            status = 'marginal'
            recommendation = 'Circuit is near the noise threshold. Consider reducing depth.'
        else:
            status = 'critical'
            recommendation = (
                f'Target noise {target_noise:.1%} exceeds critical threshold '
                f'{sigma_c:.1%}. Circuit will not function reliably.'
            )

        return {
            'status': status,
            'sigma_c': sigma_c,
            'target_noise': target_noise,
            'margin': float(margin),
            'recommendation': recommendation,
            **analysis,
        }

    @staticmethod
    def _simulate_noisy_circuit(n_qubits: int, depth: int, n_gates: int,
                                epsilon: float) -> float:
        """
        Estimate success probability under depolarizing noise.

        Uses the analytical approximation:
            P_success ~ (1 - epsilon)^n_gates * P_ideal

        where P_ideal = 1/2^n_qubits for a random circuit with structure.
        """
        survival = (1.0 - epsilon) ** n_gates
        p_random = 1.0 / (2 ** n_qubits)
        p_success = p_random + (1.0 - p_random) * survival
        return float(p_success)

    def add_metadata(self, circuit) -> None:
        """
        Attach sigma_c analysis as circuit metadata.

        Args:
            circuit: Qiskit QuantumCircuit (modified in place).
        """
        analysis = self.analyze_circuit(circuit)
        if not hasattr(circuit, 'metadata') or circuit.metadata is None:
            circuit.metadata = {}
        circuit.metadata['sigma_c'] = {
            'sigma_c': analysis['sigma_c'],
            'kappa': analysis['kappa'],
            'n_qubits': analysis['n_qubits'],
            'depth': analysis['depth'],
        }
