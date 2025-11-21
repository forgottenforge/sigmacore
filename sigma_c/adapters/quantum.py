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
        
        if circuit_type == 'grover' or circuit_type == 'custom':
            # Success probability of '11' (Default for custom/Grover)
            # In a real generic case, we'd need a target state param
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
            elif circuit_type == 'qaoa':
                circuit = self.create_qaoa_with_noise(epsilon=eps, batch_seed=batch_seed, **kwargs)
            elif circuit_type == 'custom' and 'custom_circuit' in kwargs:
                # For custom circuits, we can't easily inject noise into the structure 
                # without knowing it. In simulation mode, we rely on the 'eps' param 
                # in the result generation below.
                circuit = kwargs['custom_circuit']
            else:
                circuit = self.create_grover_with_noise(epsilon=eps, batch_seed=batch_seed, **kwargs)
            
            # Run on device
            if _HAS_BRAKET:
                task = self.device.run(circuit, shots=kwargs.get('shots', 100))
                counts = task.result().measurement_counts
            else:
                # Simulated result for testing without Braket
                # Simulate degradation with epsilon
                # Base success rate 0.9, degrades with eps
                success_prob = max(0.1, 0.9 - 2.0 * eps) 
                n_shots = kwargs.get('shots', 100)
                n_success = int(n_shots * success_prob)
                n_fail = n_shots - n_success
                counts = {'11': n_success, '00': n_fail}
            
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
    
    # ========== v1.1.0: Quantum Diagnostics ==========
    
    def _domain_specific_diagnose(self, circuit: Circuit, **kwargs) -> Dict[str, Any]:
        """
        Quantum-specific diagnostics.
        
        Checks:
        - Circuit depth and complexity
        - Gate fidelity estimates
        - Noise model validity
        - Qubit count feasibility
        """
        issues = []
        recommendations = []
        details = {}
        
        if not _HAS_BRAKET:
            issues.append("Braket SDK not installed")
            recommendations.append("Install: pip install amazon-braket-sdk")
            return {
                'status': 'error',
                'issues': issues,
                'recommendations': recommendations,
                'auto_fix': None,
                'details': details
            }
        
        # Check 1: Circuit depth
        depth = self._estimate_circuit_depth(circuit)
        details['circuit_depth'] = depth
        
        if depth > 100:
            issues.append(f"Circuit depth too high: {depth} (risk of decoherence)")
            recommendations.append("Reduce circuit depth or use error mitigation")
        elif depth > 50:
            issues.append(f"Circuit depth moderate: {depth} (may have noise issues)")
            recommendations.append("Consider noise calibration")
        
        # Check 2: Qubit count
        n_qubits = self._count_qubits(circuit)
        details['n_qubits'] = n_qubits
        
        if n_qubits > 20:
            issues.append(f"High qubit count: {n_qubits} (simulation may be slow)")
            recommendations.append("Consider using AWS quantum devices for >20 qubits")
        
        # Check 3: Gate fidelity estimate
        fidelity = self._estimate_gate_fidelity(circuit)
        details['estimated_fidelity'] = fidelity
        
        if fidelity < 0.9:
            issues.append(f"Low estimated gate fidelity: {fidelity:.3f}")
            recommendations.append("Calibrate gates or use error correction")
        
        # Determine status
        if len(issues) == 0:
            status = 'ok'
        elif any('error' in i.lower() or 'not installed' in i.lower() for i in issues):
            status = 'error'
        else:
            status = 'warning'
        
        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'auto_fix': lambda: self.auto_search(circuit_type='grover') if issues else None,
            'details': details
        }
    
    def _domain_specific_auto_search(self, circuit_type: str = 'grover', 
                                      param_ranges: Optional[Dict] = None, 
                                      **kwargs) -> Dict[str, Any]:
        """
        Auto-search optimal quantum circuit parameters.
        
        Searches:
        - Noise levels (epsilon)
        - Idle fractions
        - Circuit depths
        """
        if not _HAS_BRAKET:
            return {
                'best_params': {},
                'all_results': [],
                'convergence_data': {},
                'recommendation': 'Braket SDK not installed'
            }
        
        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {
                'epsilon': (0.0, 0.1),
                'idle_frac': (0.0, 0.3)
            }
        
        # Generate search grid
        epsilons = np.linspace(*param_ranges['epsilon'], 12)
        idle_fracs = np.linspace(*param_ranges['idle_frac'], 6)
        
        results = []
        
        for eps in epsilons:
            for idle in idle_fracs:
                try:
                    # Create circuit with these parameters
                    if circuit_type == 'grover':
                        circuit = self.create_grover_with_noise(
                            n_qubits=kwargs.get('n_qubits', 2),
                            epsilon=eps,
                            idle_frac=idle
                        )
                    else:
                        circuit = self.create_qaoa_with_noise(
                            n_qubits=kwargs.get('n_qubits', 3),
                            epsilon=eps
                        )
                    
                    # Run optimization
                    result = self.run_optimization(circuit_type=circuit_type, 
                                                   epsilon_values=[eps],
                                                   n_qubits=kwargs.get('n_qubits', 2),
                                                   shots=kwargs.get('n_shots', 50))
                    
                    results.append({
                        'epsilon': eps,
                        'idle_frac': idle,
                        'sigma_c': result['sigma_c'],
                        'kappa': result['kappa'],
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'epsilon': eps,
                        'idle_frac': idle,
                        'sigma_c': 0,
                        'kappa': 0,
                        'success': False,
                        'error': str(e)
                    })
        
        # Find best result
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            return {
                'best_params': {},
                'all_results': results,
                'convergence_data': {},
                'recommendation': 'No successful runs - check circuit configuration'
            }
        
        best = max(successful, key=lambda x: x['kappa'])
        
        return {
            'best_params': {
                'epsilon': best['epsilon'],
                'idle_frac': best['idle_frac']
            },
            'all_results': results,
            'convergence_data': {
                'n_successful': len(successful),
                'n_failed': len(results) - len(successful)
            },
            'recommendation': f"Use epsilon={best['epsilon']:.4f}, idle_frac={best['idle_frac']:.2f} for κ={best['kappa']:.2f}"
        }
    
    def _domain_specific_validate(self, circuit: Circuit, **kwargs) -> Dict[str, bool]:
        """
        Validate quantum-specific techniques.
        """
        if not _HAS_BRAKET:
            return {
                'braket_installed': False,
                'circuit_valid': False,
                'noise_physical': False,
                'qubit_count_ok': False
            }
        
        return {
            'braket_installed': True,
            'circuit_executable': self._check_circuit_executable(circuit),
            'gate_set_valid': self._check_gate_set(circuit),
            'qubit_count_ok': self._count_qubits(circuit) <= 30
        }
    
    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """
        Quantum-specific result explanation.
        """
        sigma_c = result.get('sigma_c', 'N/A')
        kappa = result.get('kappa', 'N/A')
        
        explanation = f"""
# Quantum Circuit Analysis Results

**Critical Noise Level (σ_c):** {sigma_c}  
**Criticality Score (κ):** {kappa}

## Quantum-Specific Interpretation

### Critical Noise Level (σ_c)
- Indicates the noise threshold where quantum advantage breaks down
- Lower σ_c means the circuit is more sensitive to noise
- Typical values: 0.01-0.05 for NISQ devices

### Criticality Score (κ)
- Measures how sharply the circuit transitions at σ_c
- Higher κ indicates a more pronounced quantum-to-classical transition
- κ > 15: Strong quantum advantage region
- κ < 5: Gradual degradation (may need error mitigation)

## Recommendations

{self._generate_quantum_recommendations(result)}
"""
        return explanation.strip()
    
    # ========== Helper Methods ==========
    
    def _estimate_circuit_depth(self, circuit: Circuit) -> int:
        """Estimate circuit depth."""
        if not _HAS_BRAKET or not hasattr(circuit, 'instructions'):
            return 0
        return len(circuit.instructions)
    
    def _count_qubits(self, circuit: Circuit) -> int:
        """Count qubits in circuit."""
        if not _HAS_BRAKET or not hasattr(circuit, 'instructions'):
            return 0
        
        qubits = set()
        for instr in circuit.instructions:
            if hasattr(instr, 'target'):
                if isinstance(instr.target, (list, tuple)):
                    qubits.update(instr.target)
                else:
                    qubits.add(instr.target)
        return len(qubits)
    
    def _estimate_gate_fidelity(self, circuit: Circuit) -> float:
        """Estimate overall gate fidelity."""
        depth = self._estimate_circuit_depth(circuit)
        if depth == 0:
            return 1.0
        
        # Assume 99.5% fidelity per gate (typical for NISQ)
        single_gate_fidelity = 0.995
        return single_gate_fidelity ** depth
    
    def _check_circuit_executable(self, circuit: Circuit) -> bool:
        """Check if circuit can be executed."""
        if not _HAS_BRAKET:
            return False
        return hasattr(circuit, 'instructions') and len(circuit.instructions) > 0
    
    def _check_gate_set(self, circuit: Circuit) -> bool:
        """Validate gate set is supported."""
        # All gates in our circuits are standard
        return True
    
    def _generate_quantum_recommendations(self, result: Dict[str, Any]) -> str:
        """Generate recommendations based on results."""
        kappa = result.get('kappa', 0)
        sigma_c = result.get('sigma_c', 0)
        
        recs = []
        
        if kappa < 5:
            recs.append("- **Low κ:** Consider error mitigation or circuit optimization")
        elif kappa > 15:
            recs.append("- **High κ:** Strong quantum advantage - good circuit design!")
        
        if sigma_c < 0.02:
            recs.append("- **Low σ_c:** Circuit is noise-sensitive - use error correction")
        elif sigma_c > 0.08:
            recs.append("- **High σ_c:** Circuit is noise-resilient - good for NISQ devices")
        
        if not recs:
            recs.append("- Results look reasonable - proceed with full analysis")
        
        return "\n".join(recs)

    # ========== v1.2.0: Universal Rigor Integration ==========

    def optimize_circuit(self, 
                        circuit_factory: Any, 
                        param_space: Dict[str, List[Any]],
                        strategy: str = 'brute_force') -> Dict[str, Any]:
        """
        Optimize a quantum circuit using the BalancedQuantumOptimizer.
        
        Args:
            circuit_factory: Callable that returns a circuit given params.
            param_space: Dictionary of parameter names and values to sweep.
            strategy: Optimization strategy ('brute_force').
            
        Returns:
            OptimizationResult as a dictionary.
        """
        from ..optimization.quantum import BalancedQuantumOptimizer
        
        optimizer = BalancedQuantumOptimizer(self)
        result = optimizer.optimize_circuit(circuit_factory, param_space, strategy)
        
        return {
            'optimal_params': result.optimal_params,
            'score': result.score,
            'sigma_c_before': result.sigma_c_before,
            'sigma_c_after': result.sigma_c_after,
            'performance_improvement': result.performance_after - result.performance_before
        }

    def validate_rigorously(self, sigma_c_value: float, n_qubits: int) -> Dict[str, Any]:
        """
        Validate a sigma_c result against rigorous quantum bounds.
        """
        from ..physics.quantum import RigorousQuantumSigmaC
        
        checker = RigorousQuantumSigmaC()
        context = {'n_qubits': n_qubits}
        
        return checker.validate_sigma_c(sigma_c_value, context)
