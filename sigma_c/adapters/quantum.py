#!/usr/bin/env python3
"""
Sigma-C Quantum Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Quantum Processing Units (QPUs) and simulators.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
import numpy as np
from typing import Any, Dict, List, Optional
import warnings

try:
    from braket.circuits import Circuit
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator
    _HAS_BRAKET = True
except ImportError:
    _HAS_BRAKET = False
    # Dummy classes for type hinting/runtime safety if not present
    class Circuit: 
        def __init__(self): self.instructions = []
        def h(self, q): pass
        def x(self, q): pass
        def cnot(self, c, t): pass
        def rx(self, q, a): pass
        def ry(self, q, a): pass
    class LocalSimulator: pass
    class AwsDevice: pass

class QuantumAdapter(SigmaCAdapter):
    """
    Adapter for Quantum Processing Units (QPUs).
    Ported from vali_rigg3.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not _HAS_BRAKET:
            warnings.warn("Amazon Braket SDK not found. QuantumAdapter will not function correctly.")
        
        self.device_name = self.config.get('device', 'simulator')
        if _HAS_BRAKET:
            self._connect_device()
        
    def _connect_device(self):
        if self.device_name == 'rigetti':
            try:
                self.device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
            except Exception:
                self.device = LocalSimulator("braket_dm")
        else:
            self.device = LocalSimulator("braket_dm")
            
    def get_observable(self, data: Dict[str, int], **kwargs) -> float:
        """
        Compute observable from measurement counts.
        """
        total = sum(data.values())
        if total == 0:
            return 0.0
            
        circuit_type = kwargs.get('circuit_type', 'grover')
        
        if circuit_type == 'grover':
            # Success probability of '11'
            return data.get('11', 0) / total
        elif circuit_type == 'qaoa':
            # MaxCut value
            edges = [(0, 1), (1, 2), (0, 2)]
            cut_value = 0
            for bitstring, cnt in data.items():
                bits = [int(b) for b in bitstring]
                cut = sum(1 for u, v in edges if bits[u] != bits[v])
                cut_value += cut * cnt
            return cut_value / (total * len(edges))
        return 0.0

    # ========== Noise Injection ==========
    def add_physical_noise_layer(self, circuit: Circuit, epsilon: float, layer_idx: int = 0) -> Circuit:
        if not _HAS_BRAKET: return circuit
        
        if epsilon <= 0:
            return circuit
        seed = (layer_idx * 6364136223846793005 + 1442695040888963407) % (2**63 - 1)
        rng = np.random.default_rng(seed)
        qubits_used = set()
        for instr in circuit.instructions:
            tgt = instr.target
            if isinstance(tgt, (list, tuple)):
                qubits_used.update(tgt)
            elif tgt is not None:
                qubits_used.add(tgt)
        if not qubits_used:
            qubits_used = {0}
        for q in qubits_used:
            if rng.random() < epsilon:
                angle = (0.03 + 0.02 * rng.random()) * np.pi * (1 if rng.random() < 0.5 else -1)
                if rng.random() < 0.5:
                    circuit.rx(q, angle)
                else:
                    circuit.ry(q, angle)
        if len(qubits_used) >= 2 and rng.random() < epsilon * 0.5:
            ql = sorted(list(qubits_used))
            circuit.cnot(ql[0], ql[1])
            circuit.cnot(ql[0], ql[1])
        return circuit

    def add_idle_dephasing(self, circuit: Circuit, n_qubits: int, idle_level: float, seed: Optional[int] = None) -> Circuit:
        if not _HAS_BRAKET: return circuit
        
        if idle_level <= 0:
            return circuit
        rng = np.random.default_rng((seed or 0) + int(1e6 * idle_level) + 4242)
        p_dephase = min(0.95, idle_level * 0.6)
        for q in range(n_qubits):
            if rng.random() < p_dephase:
                angle = rng.uniform(-0.06, 0.06) * np.pi
                circuit.ry(q, angle)
        return circuit

    def _rz_physical(self, circuit: Circuit, q: int, theta: float):
        if not _HAS_BRAKET: return
        circuit.rx(q, np.pi/2)
        circuit.ry(q, theta)
        circuit.rx(q, -np.pi/2)

    # ========== Circuit Builders ==========
    def create_grover_with_noise(self, n_qubits: int = 2, epsilon: float = 0.0,
                                 idle_frac: float = 0.0, batch_seed: int = 0, **kwargs) -> Circuit:
        circuit = Circuit()
        if not _HAS_BRAKET: return circuit
        
        layer_idx = 0
        alpha = 0.50
        
        def eff(eps: float) -> float:
            return float(min(0.25, max(0.0, eps + alpha * idle_frac)))
        
        idle_amp = idle_frac
        
        # Initial Hadamards
        for i in range(n_qubits):
            circuit.h(i)
        circuit = self.add_physical_noise_layer(circuit, eff(epsilon), layer_idx)
        layer_idx += 1
        
        # Idle layers
        extra_idle_layers = max(1 if idle_frac > 0 else 0, int(round(12 * idle_frac)))
        for k in range(extra_idle_layers):
            circuit = self.add_idle_dephasing(circuit, n_qubits, idle_amp, seed=batch_seed + k)
            circuit = self.add_physical_noise_layer(circuit, eff(epsilon), layer_idx)
            layer_idx += 1
        
        # Oracle
        circuit.cnot(0, 1)
        self._rz_physical(circuit, 1, np.pi)
        circuit.cnot(0, 1)
        circuit = self.add_physical_noise_layer(circuit, eff(epsilon), layer_idx)
        layer_idx += 1
        
        # More idle
        for k in range(extra_idle_layers):
            circuit = self.add_idle_dephasing(circuit, n_qubits, idle_amp, seed=batch_seed + 100 + k)
            circuit = self.add_physical_noise_layer(circuit, eff(epsilon), layer_idx)
            layer_idx += 1
        
        # Diffusion
        for i in range(n_qubits):
            circuit.h(i)
        for i in range(n_qubits):
            circuit.x(i)
        circuit.cnot(0, 1)
        self._rz_physical(circuit, 1, np.pi)
        circuit.cnot(0, 1)
        for i in range(n_qubits):
            circuit.x(i)
        for i in range(n_qubits):
            circuit.h(i)
        
        circuit = self.add_physical_noise_layer(circuit, eff(epsilon), layer_idx + batch_seed)
        return circuit

    def create_qaoa_with_noise(self, n_qubits: int = 3, depth: int = 1,
                               epsilon: float = 0.0, batch_seed: int = 0, **kwargs) -> Circuit:
        circuit = Circuit()
        if not _HAS_BRAKET: return circuit
        
        layer_idx = 0
        
        for i in range(n_qubits):
            circuit.h(i)
        
        edges = [(0, 1), (1, 2), (0, 2)]
        gamma, beta = 0.25, 1.25
        
        for d in range(depth):
            # Cost layer
            for u, v in edges:
                circuit.cnot(u, v)
                self._rz_physical(circuit, v, 2*gamma)
                circuit.cnot(u, v)
            circuit = self.add_physical_noise_layer(circuit, epsilon, layer_idx + 100 * d + batch_seed)
            layer_idx += 1
            
            # Mixer layer
            for i in range(n_qubits):
                circuit.rx(i, 2*beta)
            circuit = self.add_physical_noise_layer(circuit, epsilon, layer_idx + 100 * d + batch_seed)
            layer_idx += 1
        
        return circuit

    def run_optimization(self, circuit_type='grover', epsilon_values=None, **kwargs):
        """
        Run the optimization loop (formerly run_batched_circuits).
        """
        if epsilon_values is None:
            epsilon_values = np.linspace(0.02, 0.22, 10)
            
        results = {}
        observables = []
        
        for eps in epsilon_values:
            batch_seed = int(eps * 10000)
            if circuit_type == 'grover':
                circuit = self.create_grover_with_noise(epsilon=eps, batch_seed=batch_seed, **kwargs)
            else:
                circuit = self.create_qaoa_with_noise(epsilon=eps, batch_seed=batch_seed, **kwargs)
            
            # Run on device
            if _HAS_BRAKET:
                task = self.device.run(circuit, shots=kwargs.get('shots', 100))
                counts = task.result().measurement_counts
            else:
                # Simulated result for testing without Braket
                counts = {'11': int(kwargs.get('shots', 100) * (0.9 - eps)), '00': int(kwargs.get('shots', 100) * (0.1 + eps))}
            
            obs = self.get_observable(counts, circuit_type=circuit_type)
            observables.append(obs)
            results[eps] = counts
            
        # Compute Susceptibility using Core Engine
        analysis = self.compute_susceptibility(np.array(epsilon_values), np.array(observables))
        
        return {
            'sigma_c': analysis['sigma_c'],
            'kappa': analysis['kappa'],
            'epsilon': epsilon_values,
            'observable': observables,
            'raw_counts': results
        }
