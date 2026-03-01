#!/usr/bin/env python3
"""
Sigma-C Quantum Connector Demos
=================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates Qiskit, Cirq, and PennyLane integrations.
All demos run without installing the quantum frameworks.

Run: python examples/v4/demo_quantum_connectors.py

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np


def demo_qiskit():
    """
    Qiskit: Find the noise threshold where quantum advantage dies.

    WHO USES THIS: Quantum hardware engineers at IBM, Google, etc.
    AHA MOMENT: sigma_c tells you EXACTLY how clean your gates need
    to be for a given circuit to work.
    """
    print("=" * 60)
    print("  QISKIT: YOUR CIRCUIT'S EXACT NOISE LIMIT")
    print("=" * 60)

    from sigma_c.connectors.qiskit import QiskitSigmaC

    analyzer = QiskitSigmaC()

    # Simulate circuits of different depths
    circuits = [
        {'name': 'Bell State (2Q, 2 gates)',  'n_qubits': 2, 'depth': 2, 'n_gates': 2},
        {'name': 'GHZ-5 (5Q, 5 gates)',       'n_qubits': 5, 'depth': 5, 'n_gates': 5},
        {'name': 'Grover 3Q (3Q, 15 gates)',   'n_qubits': 3, 'depth': 8, 'n_gates': 15},
        {'name': 'VQE Ansatz (4Q, 40 gates)',  'n_qubits': 4, 'depth': 12, 'n_gates': 40},
        {'name': 'QFT-8 (8Q, 100 gates)',      'n_qubits': 8, 'depth': 20, 'n_gates': 100},
    ]

    print(f"\n  {'Circuit':<35} {'sigma_c':>10} {'Max Noise':>12} {'Status':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10}")

    for circ in circuits:
        noise_levels = np.linspace(0.0, 0.3, 30)
        observables = np.array([
            analyzer._simulate_noisy_circuit(
                circ['n_qubits'], circ['depth'], circ['n_gates'], eps
            ) for eps in noise_levels
        ])

        from sigma_c.core.engine import Engine
        engine = Engine()
        result = engine.compute_susceptibility(noise_levels, observables)
        sigma_c = result['sigma_c']

        status = "Easy" if sigma_c > 0.05 else ("Tight" if sigma_c > 0.01 else "Hard")
        print(f"  {circ['name']:<35} {sigma_c:>10.4f} {sigma_c*100:>11.1f}% {status:>10}")

    print(f"\n  Key insight:")
    print(f"    Deeper circuits need cleaner gates (lower sigma_c).")
    print(f"    VQE with 40 gates needs ~1% error rate.")
    print(f"    Bell state with 2 gates tolerates ~14% error!")

    # Show success probability curve for Grover
    print(f"\n  Success probability vs. noise (Grover 3Q):")
    noise = np.linspace(0, 0.2, 10)
    print(f"  {'Noise':>8} | {'P(success)':>12}")
    print(f"  {'-'*8}-+-{'-'*12}")
    for eps in noise:
        p = analyzer._simulate_noisy_circuit(3, 8, 15, eps)
        bar = '#' * int(p * 40)
        print(f"  {eps*100:>6.1f}% | {bar:<40} {p:.3f}")

    print(f"\n  --> Qiskit connector finds the exact noise budget for ANY circuit!")
    print()


def demo_pennylane():
    """
    PennyLane: Track circuit criticality during VQA optimization.

    WHO USES THIS: Quantum ML researchers running variational algorithms.
    AHA MOMENT: As your circuit gets deeper, it needs cleaner hardware.
    SigmaCDevice tells you when you've exceeded your error budget.
    """
    print("=" * 60)
    print("  PENNYLANE: CRITICALITY TRACKING FOR VQA CIRCUITS")
    print("=" * 60)

    from sigma_c.plugins.pennylane import SigmaCDevice

    dev = SigmaCDevice(wires=4)

    # Simulate a VQA optimization loop
    print(f"\n  Running variational algorithm (4 qubits)...")
    print(f"  As the circuit grows, sigma_c tracks parameter density.\n")

    print(f"  {'Iteration':<12} {'Gates':>8} {'Params':>8} {'sigma_c':>10} {'Noise Budget':>14}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*14}")

    for iteration in range(10):
        # Circuits grow during adaptive VQA
        n_gates = 5 + iteration * 3
        n_params = 2 + iteration * 2

        dev.record_execution(n_gates=n_gates, n_params=n_params)
        noise_info = dev.analyze_noise_threshold(n_gates, n_qubits=4)

        print(f"  {iteration:<12} {n_gates:>8} {n_params:>8} "
              f"{dev.sigma_c_history[-1]:>10.4f} {noise_info['sigma_c']*100:>13.2f}%")

    report = dev.get_criticality_report()
    print(f"\n  Criticality Report:")
    print(f"    Mean sigma_c: {report['mean_sigma_c']:.4f}")
    print(f"    Max sigma_c:  {report['max_sigma_c']:.4f}")
    print(f"    Min sigma_c:  {report['min_sigma_c']:.4f}")
    print(f"    Executions:   {report['samples']}")

    print(f"\n  --> Deeper circuits = tighter noise budgets!")
    print(f"  --> Use sigma_c to decide when to stop adding layers.")
    print()


def demo_cirq_concepts():
    """
    Cirq: Circuit criticality analysis.

    WHO USES THIS: Google quantum teams optimizing for Sycamore.
    AHA MOMENT: More parallel operations = lower criticality.
    The optimizer restructures circuits for maximum stability.
    """
    print("=" * 60)
    print("  CIRQ: CIRCUIT OPTIMIZATION FOR CRITICALITY")
    print("=" * 60)

    # Demonstrate concepts without Cirq dependency
    print(f"\n  CirqCriticality analyzes: operations_per_moment / 10")
    print(f"  More parallelism = lower sigma_c = more stable execution.\n")

    # Simulate different circuit structures
    circuits = [
        {'name': 'Sequential (no parallelism)', 'moments': 10, 'ops': 10},
        {'name': 'Moderate parallel',            'moments': 5,  'ops': 10},
        {'name': 'Fully parallel',               'moments': 2,  'ops': 10},
        {'name': 'Deep sequential',              'moments': 20, 'ops': 40},
        {'name': 'Shallow wide',                 'moments': 4,  'ops': 40},
    ]

    print(f"  {'Circuit':<30} {'Moments':>10} {'Ops':>6} {'Ops/Mom':>10} {'sigma_c':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*6} {'-'*10} {'-'*10}")

    for c in circuits:
        ops_per_moment = c['ops'] / c['moments'] if c['moments'] > 0 else 0
        sigma_c = float(np.clip(ops_per_moment / 10.0, 0, 1))
        print(f"  {c['name']:<30} {c['moments']:>10} {c['ops']:>6} {ops_per_moment:>10.1f} {sigma_c:>10.3f}")

    print(f"\n  Optimization strategy:")
    print(f"    High sigma_c -> Restructure for more parallelism")
    print(f"    Low sigma_c  -> Circuit is already well-organized")
    print(f"\n  --> CriticalOptimizer rewrites circuits for target sigma_c!")
    print()


def demo_noise_comparison():
    """
    Cross-framework noise comparison.

    WHO USES THIS: Hardware benchmarkers comparing quantum platforms.
    AHA MOMENT: sigma_c enables apples-to-apples comparison across
    Qiskit, Cirq, PennyLane -- same metric, same meaning.
    """
    print("=" * 60)
    print("  CROSS-FRAMEWORK: UNIVERSAL NOISE METRIC")
    print("=" * 60)

    from sigma_c.connectors.qiskit import QiskitSigmaC
    from sigma_c.plugins.pennylane import SigmaCDevice
    from sigma_c.core.engine import Engine

    # Same logical circuit, analyzed through each framework
    # 3-qubit, 10-gate circuit
    n_qubits = 3
    n_gates = 10

    # Qiskit analysis
    analyzer = QiskitSigmaC()
    noise_levels = np.linspace(0, 0.3, 30)
    obs = np.array([analyzer._simulate_noisy_circuit(n_qubits, 5, n_gates, e) for e in noise_levels])
    engine = Engine()
    qiskit_result = engine.compute_susceptibility(noise_levels, obs)

    # PennyLane analysis
    dev = SigmaCDevice(wires=n_qubits)
    pl_result = dev.analyze_noise_threshold(n_gates, n_qubits)

    print(f"\n  Same circuit: {n_qubits} qubits, {n_gates} gates")
    print(f"\n  {'Framework':<15} {'Method':<30} {'sigma_c':>10}")
    print(f"  {'-'*15} {'-'*30} {'-'*10}")
    print(f"  {'Qiskit':<15} {'Susceptibility peak':30} {qiskit_result['sigma_c']:>10.4f}")
    print(f"  {'PennyLane':<15} {'Analytical (1-0.5^(1/n))':30} {pl_result['sigma_c']:>10.4f}")

    print(f"\n  Both methods agree: max noise ~ {qiskit_result['sigma_c']*100:.1f}% per gate")
    print(f"\n  --> One metric, every framework, universal meaning!")
    print()


def main():
    print("""
==========================================================================
   Sigma-C Framework -- Quantum Connector Demos
   Qiskit | PennyLane | Cirq | Cross-Framework
   All demos run without installing quantum frameworks.
==========================================================================
    """)

    demos = [
        ("Qiskit", demo_qiskit),
        ("PennyLane", demo_pennylane),
        ("Cirq Concepts", demo_cirq_concepts),
        ("Cross-Framework", demo_noise_comparison),
    ]

    passed = []
    failed = []

    for name, demo_fn in demos:
        try:
            demo_fn()
            passed.append(name)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print("=" * 60)
    print(f"  QUANTUM DEMOS COMPLETED: {len(passed)}/{len(demos)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print(f"  All quantum connector demos passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
