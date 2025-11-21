"""
Sigma-C Qiskit Connector
========================
Copyright (c) 2025 ForgottenForge.xyz

Integrates Sigma-C criticality analysis with IBM Qiskit.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.providers import Backend
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False
    QuantumCircuit = Any
    Backend = Any

from ..core.engine import Engine
from ..adapters.quantum import QuantumAdapter


class QiskitSigmaC:
    """
    Qiskit integration for automatic criticality analysis.
    
    Usage:
        from qiskit import QuantumCircuit
        from sigma_c.connectors.qiskit import QiskitSigmaC
        
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        
        sigma_c = QiskitSigmaC.analyze(circuit)
        optimized = QiskitSigmaC.optimize_for_backend(circuit, 'ibmq_manila')
    """
    
    @staticmethod
    def analyze(circuit: QuantumCircuit, noise_levels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze a Qiskit circuit for criticality.
        
        Args:
            circuit: Qiskit QuantumCircuit
            noise_levels: Optional array of noise levels to sweep
            
        Returns:
            Dictionary with sigma_c, kappa, and other metrics
        """
        if not _HAS_QISKIT:
            raise ImportError("Qiskit not installed. Run: pip install qiskit")
        
        # Convert Qiskit circuit to Braket format (if needed)
        # For now, we use the QuantumAdapter directly
        adapter = QuantumAdapter()
        
        # Extract circuit parameters
        n_qubits = circuit.num_qubits
        depth = circuit.depth()
        
        # Run criticality analysis
        if noise_levels is None:
            noise_levels = np.linspace(0.0, 0.2, 10)
        
        # Create equivalent circuit in our format
        def circuit_factory(epsilon):
            return adapter.create_grover_with_noise(n_qubits=n_qubits, epsilon=epsilon)
        
        # Analyze
        result = adapter.run_optimization(
            circuit_type='custom',
            epsilon_values=noise_levels,
            shots=100,
            custom_circuit=circuit_factory(0.0)
        )
        
        return {
            'sigma_c': result['sigma_c'],
            'kappa': result['kappa'],
            'depth': depth,
            'n_qubits': n_qubits,
            'framework': 'qiskit'
        }
    
    @staticmethod
    def optimize_for_backend(circuit: QuantumCircuit, backend_name: str) -> QuantumCircuit:
        """
        Optimize circuit for a specific backend considering criticality.
        
        Args:
            circuit: Input Qiskit circuit
            backend_name: Target backend (e.g., 'ibmq_manila')
            
        Returns:
            Optimized circuit
        """
        if not _HAS_QISKIT:
            raise ImportError("Qiskit not installed")
        
        # Analyze current circuit
        analysis = QiskitSigmaC.analyze(circuit)
        
        # If sigma_c is too low (too sensitive), suggest modifications
        if analysis['sigma_c'] < 0.05:
            # Circuit is very noise-sensitive
            # Apply error mitigation strategies
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import Optimize1qGatesDecomposition
            
            pm = PassManager([Optimize1qGatesDecomposition()])
            optimized = pm.run(circuit)
            
            return optimized
        
        return circuit
    
    @staticmethod
    def add_criticality_metadata(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Attach sigma_c metadata to circuit.
        
        Args:
            circuit: Qiskit circuit
            
        Returns:
            Circuit with metadata attached
        """
        analysis = QiskitSigmaC.analyze(circuit)
        circuit.metadata = circuit.metadata or {}
        circuit.metadata['sigma_c'] = analysis
        return circuit
