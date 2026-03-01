#!/usr/bin/env python3
"""
Sigma-C Edge & LLM Cost Demos
================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates EdgeAdapter, MLAdapter, and LLMCostAdapter.
All demos run locally without external dependencies.

Run: python examples/v4/demo_edge_llm.py

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np


def demo_edge():
    """
    Edge Computing: Find the optimal frequency for IoT devices.

    WHO USES THIS: Embedded engineers designing battery-powered systems.
    AHA MOMENT: There's a cliff where doubling frequency gives you
    almost no performance gain but kills your battery.
    sigma_c finds that cliff.
    """
    print("=" * 60)
    print("  EDGE: FIND YOUR IOT DEVICE'S SWEET SPOT")
    print("=" * 60)

    from sigma_c.adapters.edge import EdgeAdapter

    adapter = EdgeAdapter(config={'hardware': 'cortex_m4'})

    # Simulate MCU frequency vs power vs performance
    frequencies = np.array([16, 32, 48, 64, 80, 96, 112, 128, 144, 160,
                            176, 192, 208, 224, 240, 256], dtype=float)  # MHz

    # Power grows quadratically (P ~ f^2 for CMOS)
    power = 0.001 * frequencies ** 2  # mW

    # Performance is linear up to a point, then memory-bound
    performance = np.minimum(frequencies, 180.0)  # MIPS, saturates at 180 MHz

    result = adapter.analyze_power_efficiency(frequencies, power, performance)

    print(f"\n  Simulating Cortex-M4 frequency scaling (16-256 MHz):")
    print(f"\n  {'Freq (MHz)':<12} {'Power (mW)':>12} {'Perf (MIPS)':>12} {'Perf/Watt':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    critical_freq = result['critical_frequency']
    for i in range(len(frequencies)):
        eff = performance[i] / (power[i] + 1e-9)
        marker = " <-- sigma_c" if abs(frequencies[i] - critical_freq) < 8 else ""
        print(f"  {frequencies[i]:>10.0f}   {power[i]:>10.1f}   {performance[i]:>10.1f}   {eff:>10.1f}{marker}")

    print(f"\n  Critical frequency: {result['critical_frequency']:.0f} MHz")
    print(f"  Max efficiency:     {result['max_efficiency']:.1f} MIPS/W")

    print(f"\n  --> Above {result['critical_frequency']:.0f} MHz, each MHz costs")
    print(f"      exponentially more power for diminishing performance!")
    print(f"  --> Set your MCU to {result['critical_frequency']:.0f} MHz for optimal battery life.")
    print()


def demo_ml_adapter():
    """
    ML Adapter: Find optimal hyperparameters by balancing accuracy vs. robustness.

    WHO USES THIS: ML engineers tuning dropout, learning rate, weight decay.
    AHA MOMENT: Sigma_c tells you whether your model is overfitting (too low)
    or underfitting (too high). The sweet spot is the critical point.
    """
    print("=" * 60)
    print("  ML ADAPTER: HYPERPARAMETER SWEET SPOT FINDER")
    print("=" * 60)

    from sigma_c.adapters.ml import MLAdapter

    adapter = MLAdapter()

    # Sweep learning rates
    lrs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    print(f"\n  Sweeping learning rates (batch_size=64, dropout=0.1):")
    print(f"\n  {'LR':>10} {'Accuracy':>10} {'Robustness':>12} {'Combined':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    best_combined = 0
    best_lr = None
    for lr in lrs:
        params = {'learning_rate': lr, 'batch_size': 64, 'dropout': 0.1, 'weight_decay': 0.001}
        acc = adapter.train_and_evaluate(None, params)['accuracy']
        rob = adapter.measure_robustness(None, params)['sigma_c']
        combined = 0.7 * acc + 0.3 * rob

        marker = ""
        if combined > best_combined:
            best_combined = combined
            best_lr = lr
            marker = " <-- best"

        print(f"  {lr:>10.4f} {acc:>10.3f} {rob:>12.3f} {combined:>10.3f}{marker}")

    print(f"\n  Optimal LR: {best_lr}")
    print(f"  --> Balances accuracy (70%) with robustness (30%)!")

    # Now sweep dropout
    print(f"\n  Sweeping dropout (LR={best_lr}, batch_size=64):")
    dropouts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    print(f"\n  {'Dropout':>10} {'Accuracy':>10} {'Robustness':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*12}")
    for d in dropouts:
        params = {'learning_rate': best_lr, 'batch_size': 64, 'dropout': d, 'weight_decay': 0.001}
        acc = adapter.train_and_evaluate(None, params)['accuracy']
        rob = adapter.measure_robustness(None, params)['sigma_c']
        print(f"  {d:>10.2f} {acc:>10.3f} {rob:>12.3f}")

    print(f"\n  --> Higher dropout = more robust but less accurate.")
    print(f"  --> sigma_c finds the balance point!")
    print()


def demo_llm_cost():
    """
    LLM Cost: Choose the cheapest model that's still safe.

    WHO USES THIS: Product engineers choosing between GPT-4, Claude, Llama.
    AHA MOMENT: The cheapest model isn't always the best value.
    sigma_c finds the Pareto-optimal choice: best safety per dollar.
    """
    print("=" * 60)
    print("  LLM COST: CHEAPEST SAFE MODEL SELECTOR")
    print("=" * 60)

    from sigma_c.adapters.llm_cost import LLMCostAdapter

    adapter = LLMCostAdapter(config={'budget_per_1k': 0.05})

    models = [
        {'name': 'GPT-4o',         'cost': 0.030, 'hallucination_rate': 0.02},
        {'name': 'Claude Opus',    'cost': 0.075, 'hallucination_rate': 0.01},
        {'name': 'Claude Sonnet',  'cost': 0.015, 'hallucination_rate': 0.03},
        {'name': 'GPT-4o-mini',    'cost': 0.005, 'hallucination_rate': 0.08},
        {'name': 'Llama 70B',      'cost': 0.001, 'hallucination_rate': 0.12},
        {'name': 'Llama 8B',       'cost': 0.0002,'hallucination_rate': 0.25},
        {'name': 'Mistral Large',  'cost': 0.012, 'hallucination_rate': 0.05},
    ]

    result = adapter.analyze_cost_safety(models)

    print(f"\n  Model Comparison (max hallucination rate: 15%):")
    print(f"\n  {'Model':<18} {'Cost/1K':>10} {'Halluc.':>10} {'Safe?':>8} {'Value':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for m in sorted(models, key=lambda x: x['cost']):
        safe = m['hallucination_rate'] < 0.15
        value = 1.0 / (m['cost'] * m['hallucination_rate'] + 1e-9) if safe else 0
        marker = " *" if m['name'] == result['best_model'] else ""
        print(f"  {m['name']:<18} ${m['cost']:>8.4f} {m['hallucination_rate']*100:>9.1f}% "
              f"{'Yes' if safe else 'NO':>8} {value:>10.0f}{marker}")

    print(f"\n  Winner: {result['best_model']}")
    print(f"  Cost:   ${result['optimal_cost']:.4f} per 1K tokens")
    print(f"  Safety: {result['safety_score']*100:.0f}% (sigma_c = {result['sigma_c']:.3f})")

    print(f"\n  --> Llama 8B is cheapest but fails safety threshold!")
    print(f"  --> {result['best_model']} has the best value (quality/cost ratio)!")
    print()


def demo_edge_observable():
    """
    Edge Adapter: Energy efficiency as observable.

    WHO USES THIS: IoT fleet managers monitoring device health.
    AHA MOMENT: When efficiency drops, the device is overloaded.
    sigma_c tells you exactly where the overload transition happens.
    """
    print("=" * 60)
    print("  EDGE OBSERVABLE: FLEET EFFICIENCY MONITORING")
    print("=" * 60)

    from sigma_c.adapters.edge import EdgeAdapter

    adapter = EdgeAdapter(config={'hardware': 'esp32'})

    # Simulate a fleet of 20 devices with increasing load
    np.random.seed(42)
    n_devices = 20
    loads = np.linspace(0.1, 1.0, n_devices)

    print(f"\n  Simulating {n_devices} ESP32 devices at different loads:")
    print(f"\n  {'Load':>8} {'Perf':>8} {'Power':>8} {'Efficiency':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    efficiencies = []
    for load in loads:
        perf = 100 * (1 - np.exp(-3 * load))  # Saturating performance
        power = 50 + 200 * load ** 2            # Quadratic power
        eff = perf / power

        # Create data row for get_observable
        data_row = np.array([[perf, power]])
        obs = adapter.get_observable(data_row)
        efficiencies.append(obs)

        print(f"  {load:>7.1f}x {perf:>7.1f} {power:>7.1f}W {obs:>11.3f}")

    # Find the critical load point
    from sigma_c.core.engine import Engine
    engine = Engine()
    result = engine.compute_susceptibility(loads, np.array(efficiencies))

    print(f"\n  Critical load (sigma_c): {result['sigma_c']:.2f}x")
    print(f"  Kappa:                   {result['kappa']:.1f}")
    print(f"\n  --> Devices above {result['sigma_c']:.1f}x load are wasting energy!")
    print(f"  --> Scale out your fleet before hitting {result['sigma_c']:.1f}x load.")
    print()


def main():
    print("""
==========================================================================
   Sigma-C Framework -- Edge, ML & LLM Demos
   Edge Computing | ML Hyperparameters | LLM Cost Optimization
   All demos run locally without external dependencies.
==========================================================================
    """)

    demos = [
        ("Edge Computing", demo_edge),
        ("ML Adapter", demo_ml_adapter),
        ("LLM Cost", demo_llm_cost),
        ("Edge Observable", demo_edge_observable),
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
    print(f"  EDGE/ML DEMOS COMPLETED: {len(passed)}/{len(demos)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print(f"  All Edge/ML demos passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
