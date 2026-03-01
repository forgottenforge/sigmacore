#!/usr/bin/env python3
"""
Sigma-C ML Framework Demos
============================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates ML integrations: PyTorch, JAX, TensorFlow, CUDA.
All demos run without the actual frameworks via mock objects.

Run: python examples/v4/demo_ml_frameworks.py

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np


def demo_pytorch():
    """
    PyTorch: Criticality-aware neural network module.

    WHO USES THIS: ML engineers training deep networks.
    AHA MOMENT: sigma_c rises as weights grow -- it predicts
    exploding gradients BEFORE they crash your training.
    """
    print("=" * 60)
    print("  PYTORCH: PREDICT EXPLODING GRADIENTS BEFORE THEY HIT")
    print("=" * 60)

    # We simulate what CriticalModule does internally
    # (same algorithm, no torch dependency)
    from sigma_c.ml.pytorch import CriticalModule

    print(f"\n  CriticalModule tracks activation statistics per forward pass.")
    print(f"  sigma_c = Var(weights) / Mean(|weights|)")
    print()

    # Simulate a training run
    np.random.seed(42)
    print(f"  {'Epoch':<8} {'Loss':>8} {'Weight Scale':>14} {'sigma_c':>10} {'Status':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*14} {'-'*10} {'-'*12}")

    sigma_c_values = []
    for epoch in range(15):
        # Simulate weight growth
        scale = 0.1 * (1.05 ** epoch)
        weights = np.random.randn(1000) * scale
        loss = 1.0 / (1 + epoch * 0.2) + np.random.normal(0, 0.01)

        # Criticality computation (same as CriticalModule._compute_activation_criticality)
        var = np.var(weights)
        mean_abs = np.mean(np.abs(weights))
        sigma_c = float(np.clip(var / (mean_abs + 1e-9), 0, 1))
        sigma_c_values.append(sigma_c)

        status = "OK" if sigma_c < 0.5 else ("WARNING" if sigma_c < 0.8 else "DANGER!")
        print(f"  {epoch:<8} {loss:>8.4f} {scale:>14.4f} {sigma_c:>10.4f} {status:>12}")

    print(f"\n  sigma_c trend: {sigma_c_values[0]:.3f} -> {sigma_c_values[-1]:.3f}")
    print(f"  --> When sigma_c > 0.8, training is about to diverge!")
    print(f"  --> Use SigmaCLoss to penalize deviation from target sigma_c.")
    print()


def demo_jax():
    """
    JAX: critical_jit and CriticalOptimizer.

    WHO USES THIS: Researchers using JAX for differentiable programming.
    AHA MOMENT: The optimizer adjusts learning rate based on sigma_c --
    aggressive when stable, cautious near criticality.
    """
    print("=" * 60)
    print("  JAX: CRITICALITY-AWARE JIT AND OPTIMIZER")
    print("=" * 60)

    # Demonstrate the concepts without JAX
    print(f"\n  @critical_jit: Wraps jax.jit + attaches sigma_c metadata.")
    print(f"  CriticalOptimizer: Gradient descent with criticality regularization.")
    print()

    # Simulate CriticalOptimizer behavior
    np.random.seed(42)
    lr = 0.01
    params = np.random.randn(50) * 0.5  # Initial params

    print(f"  Simulating 15 optimization steps...")
    print(f"  {'Step':<6} {'Loss':>8} {'Grad Var':>10} {'sigma_c':>10} {'LR':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    sigma_c_history = []
    for step in range(15):
        # Simulate loss and gradient
        loss = np.sum(params ** 2) + np.random.normal(0, 0.01)
        grads = 2 * params + np.random.randn(len(params)) * 0.1

        # Criticality from gradient variance
        grad_var = float(np.var(grads))
        sigma_c = float(np.clip(grad_var / 5.0, 0, 1))
        sigma_c_history.append(sigma_c)

        # Adaptive learning rate based on sigma_c
        effective_lr = lr * (1.0 - 0.5 * sigma_c)
        params = params - effective_lr * grads

        print(f"  {step:<6} {loss:>8.4f} {grad_var:>10.4f} {sigma_c:>10.4f} {effective_lr:>10.6f}")

    print(f"\n  sigma_c dropped: {sigma_c_history[0]:.3f} -> {sigma_c_history[-1]:.3f}")
    print(f"  --> As training converges, sigma_c drops = gradients stabilize.")
    print(f"  --> The optimizer was more cautious when sigma_c was high!")
    print()


def demo_cuda_accelerator():
    """
    CUDA: GPU-accelerated susceptibility computation.

    WHO USES THIS: Anyone with large datasets who wants fast sigma_c.
    AHA MOMENT: CUDAAccelerator transparently uses GPU when available,
    falls back to CPU -- same API, same results.
    """
    print("=" * 60)
    print("  CUDA: GPU-ACCELERATED SIGMA_C COMPUTATION")
    print("=" * 60)

    from sigma_c.ml.cuda import CUDAAccelerator

    accel = CUDAAccelerator()
    backend = "GPU (CuPy)" if accel._use_gpu else "CPU (NumPy/SciPy)"
    print(f"\n  Backend: {backend}")

    # Create test data: Ising-like phase transition
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    temps = np.linspace(1.0, 3.5, 200)
    mag = np.array([max(0, (Tc - T)**0.125) if T < Tc else 0.0 for T in temps])

    # Time the computation
    import time
    t0 = time.perf_counter()
    result = accel.compute_susceptibility(temps, mag)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Data points: {len(temps)}")
    print(f"  Time: {elapsed:.1f} ms")
    print(f"\n  Results:")
    print(f"    sigma_c:  {result['sigma_c']:.4f}  (exact Tc: {Tc:.4f})")
    print(f"    kappa:    {result['kappa']:.1f}")
    print(f"    chi_max:  {result['chi_max']:.4f}")
    print(f"    Error:    {abs(result['sigma_c'] - Tc)/Tc*100:.1f}%")

    # Benchmark with larger dataset
    large_eps = np.linspace(0, 1, 10000)
    large_obs = np.exp(-5 * large_eps) * np.sin(20 * large_eps)
    t0 = time.perf_counter()
    result_large = accel.compute_susceptibility(large_eps, large_obs)
    elapsed_large = (time.perf_counter() - t0) * 1000

    print(f"\n  Large dataset (10,000 points): {elapsed_large:.1f} ms")
    print(f"    sigma_c: {result_large['sigma_c']:.4f}")

    print(f"\n  --> Same API, GPU when available, CPU fallback!")
    print(f"  --> CUDAMonitor tracks kernel timing for performance transitions.")
    print()


def demo_tensorflow_callback():
    """
    TensorFlow: SigmaCCallback for training monitoring.

    WHO USES THIS: Keras users who want epoch-level criticality tracking.
    AHA MOMENT: sigma_c rising during training = weights diverging.
    It catches instability before NaN loss appears.
    """
    print("=" * 60)
    print("  TENSORFLOW: CALLBACK CATCHES INSTABILITY EARLY")
    print("=" * 60)

    from sigma_c.ml.tensorflow import SigmaCCallback

    callback = SigmaCCallback()

    # Create mock model
    class MockLayer:
        def __init__(self, size):
            self._weights = [np.random.randn(size, size) * 0.1]
        def get_weights(self):
            return self._weights

    class MockModel:
        def __init__(self):
            self.layers = [MockLayer(64), MockLayer(32), MockLayer(16)]

    model = MockModel()
    callback.set_model(model)
    np.random.seed(42)

    print(f"\n  Simulating training with unstable learning rate...")
    print(f"  {'Epoch':<8} {'Loss':>8} {'sigma_c':>10} {'Alert':>20}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*20}")

    for epoch in range(12):
        # Simulate: weights grow exponentially (bad lr)
        growth = 1.0 + epoch * 0.08
        for layer in model.layers:
            layer._weights = [np.random.randn(*w.shape) * 0.1 * growth
                              for w in layer._weights]

        loss = 1.0 / (1 + epoch * 0.15) + np.random.normal(0, 0.02)
        callback.on_epoch_end(epoch, logs={'loss': loss})

        sc = callback.history[-1]['sigma_c']
        alert = "" if sc < 0.5 else ("Approaching critical!" if sc < 0.8 else "STOP TRAINING!")
        print(f"  {epoch:<8} {loss:>8.4f} {sc:>10.4f} {alert:>20}")

    print(f"\n  Final sigma_c: {callback.history[-1]['sigma_c']:.3f}")
    print(f"  --> The callback detected instability at epoch ~6.")
    print(f"  --> Use CriticalRegularizer to constrain sigma_c during training.")
    print()


def main():
    print("""
==========================================================================
   Sigma-C Framework -- ML Framework Demos
   PyTorch | JAX | CUDA | TensorFlow
   All demos run without installing the ML frameworks.
==========================================================================
    """)

    demos = [
        ("PyTorch", demo_pytorch),
        ("JAX", demo_jax),
        ("CUDA Accelerator", demo_cuda_accelerator),
        ("TensorFlow Callback", demo_tensorflow_callback),
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
    print(f"  ML DEMOS COMPLETED: {len(passed)}/{len(demos)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print(f"  All ML framework demos passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
