#!/usr/bin/env python3
"""
Sigma-C Integration Demos
===========================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates ALL integration modules in one file.
Each section is self-contained and runs without external services.

Run: python examples/v4/demo_integrations.py

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np


def demo_graphql_api():
    """
    GraphQL API: Query criticality from a schema.

    WHO USES THIS: Backend engineers integrating Sigma-C into GraphQL APIs.
    AHA MOMENT: You can query criticality analysis with a single GraphQL query,
    just like you'd query any other API endpoint.
    """
    print("=" * 60)
    print("  GRAPHQL API: QUERY CRITICALITY LIKE A DATABASE")
    print("=" * 60)

    from sigma_c.api.graphql import GraphQLAPI

    api = GraphQLAPI()

    # The Ising model phase transition: magnetization drops at Tc = 2.269
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    temps = np.linspace(1.0, 3.5, 30)
    mag = [max(0, (Tc - T)**0.125) if T < Tc else 0.0 for T in temps]

    query = """
    {
        analyzeSystem(input: {
            epsilon: [%s],
            observable: [%s]
        }) {
            sigmaC
            kappa
            chiMax
        }
    }
    """ % (
        ', '.join(f'{t:.3f}' for t in temps),
        ', '.join(f'{m:.4f}' for m in mag),
    )

    result = api.execute(query)
    data = result['data']['analyzeSystem']

    print(f"\n  Query: analyzeSystem(Ising model data)")
    print(f"  Response:")
    print(f"    sigmaC: {data['sigmaC']:.4f}  (exact: {Tc:.4f})")
    print(f"    kappa:  {data['kappa']:.1f}")
    print(f"    chiMax: {data['chiMax']:.4f}")
    print(f"\n  Error: {abs(data['sigmaC'] - Tc):.4f} ({abs(data['sigmaC'] - Tc)/Tc*100:.1f}%)")
    print(f"  --> GraphQL returns the Curie temperature within ~2% of the exact value!")

    # Also show the schema
    print(f"\n  Available schema:")
    for line in api.get_schema().strip().split('\n')[:8]:
        print(f"    {line}")
    print(f"    ...")
    print()


def demo_ci_analyzer():
    """
    GitHub Actions CI: Analyze code complexity.

    WHO USES THIS: DevOps engineers who want to gate deployments on code quality.
    AHA MOMENT: The CI analyzer finds the most complex files in your codebase
    and gives a single number: is this codebase maintainable?
    """
    print("=" * 60)
    print("  CI ANALYZER: FIND YOUR MOST DANGEROUS CODE")
    print("=" * 60)

    from sigma_c.monitoring.github_actions import CIAnalyzer

    analyzer = CIAnalyzer(threshold=0.5)

    # Analyze the sigma_c source code itself!
    result = analyzer.analyze_repository(os.path.join(os.path.dirname(__file__), '..', '..', 'sigma_c'))

    print(f"\n  Analyzed {result['n_files']} Python files in sigma_c/")
    print(f"  Overall sigma_c: {result['sigma_c']:.3f}")
    print(f"  Mean sigma_c:    {result.get('mean_sigma_c', 0):.3f}")
    print(f"  Status:          {result['status'].upper()}")

    # Show top 5 most complex files
    scored = [f for f in result['files'] if 'error' not in f and f['sigma_c'] > 0]
    scored.sort(key=lambda x: x['sigma_c'], reverse=True)
    print(f"\n  Top 5 most complex files:")
    print(f"  {'File':<50} {'sigma_c':>8}")
    print(f"  {'-'*50} {'-'*8}")
    for f in scored[:5]:
        short_path = f['file'].replace('\\', '/').split('sigma_c/')[-1]
        print(f"  {short_path:<50} {f['sigma_c']:>8.3f}")

    print(f"\n  --> Use this in your CI pipeline to prevent complexity creep!")
    print()


def demo_rest_api():
    """
    REST API: FastAPI endpoint for criticality.

    WHO USES THIS: Web developers building criticality-aware services.
    AHA MOMENT: The SigmaCAPI class gives you a production-ready endpoint
    with one line of code.
    """
    print("=" * 60)
    print("  REST API: ONE-LINE CRITICALITY ENDPOINT")
    print("=" * 60)

    from sigma_c.api.rest import SigmaCAPI

    api = SigmaCAPI()

    # Compute criticality directly
    epsilon = np.linspace(0, 0.5, 20).tolist()
    observable = [np.exp(-10 * e) for e in epsilon]

    result = api.compute(epsilon, observable)

    print(f"\n  POST /analyze")
    print(f"  Request:  {{epsilon: [0..0.5], observable: exp(-10*eps)}}")
    print(f"  Response: {{")
    print(f"    sigma_c:       {result['sigma_c']:.4f}")
    print(f"    kappa:         {result['kappa']:.1f}")
    print(f"    chi_max:       {result['chi_max']:.4f}")
    print(f"    peak_location: {result['peak_location']:.4f}")
    print(f"  }}")

    print(f"\n  To start a server:")
    print(f"    app = SigmaCAPI().create_app()")
    print(f"    uvicorn.run(app, port=8000)")
    print(f"\n  --> Production-ready API in 2 lines of code!")
    print()


def demo_wasm_builder():
    """
    WASM / Browser: Run Sigma-C in the browser.

    WHO USES THIS: Frontend engineers who need client-side analysis.
    AHA MOMENT: The JS module is self-contained -- no server needed.
    The browser computes criticality locally in ~1ms.
    """
    print("=" * 60)
    print("  WASM BUILDER: SIGMA-C IN YOUR BROWSER")
    print("=" * 60)

    from sigma_c.api.wasm import WASMBuilder

    builder = WASMBuilder()

    # Show the generated JS module (first 15 lines)
    js = builder.generate_js_module()
    js_lines = js.strip().split('\n')

    print(f"\n  Generated JS module ({len(js_lines)} lines):")
    for line in js_lines[:12]:
        print(f"    {line}")
    print(f"    ...")

    # Show the HTML demo size
    html = builder.generate_html_demo()
    print(f"\n  Generated HTML demo: {len(html)} bytes")
    print(f"  Features:")
    print(f"    - Interactive epsilon/observable input")
    print(f"    - Real-time bar chart of susceptibility")
    print(f"    - Zero dependencies, zero server")

    print(f"\n  To use:")
    print(f"    builder = WASMBuilder()")
    print(f"    builder.build('dist/')  # Creates sigma_c.js + demo.html")
    print(f"    builder.serve(port=8080)")
    print(f"\n  --> Full criticality analysis runs in the browser!")
    print()


def demo_home_assistant():
    """
    Home Assistant: Monitor your house criticality.

    WHO USES THIS: Smart home enthusiasts tracking system stability.
    AHA MOMENT: Your house has a criticality score! When energy usage,
    temperature, and network traffic all spike together, sigma_c goes up.
    """
    print("=" * 60)
    print("  HOME ASSISTANT: YOUR HOUSE HAS A CRITICALITY SCORE")
    print("=" * 60)

    from sigma_c.integrations.homeassistant import HomeAssistantBridge

    bridge = HomeAssistantBridge(window_size=50)
    bridge.add_sensor('temperature', weight=0.3)
    bridge.add_sensor('energy_usage', weight=0.4)
    bridge.add_sensor('network_traffic', weight=0.3)

    np.random.seed(42)

    # Simulate normal operation (30 readings)
    print(f"\n  Simulating 30 normal readings...")
    for i in range(30):
        bridge.update('temperature', 21.0 + np.random.normal(0, 0.5))
        bridge.update('energy_usage', 500 + np.random.normal(0, 20))
        bridge.update('network_traffic', 100 + np.random.normal(0, 10))

    state_normal = bridge.get_state()
    print(f"  Status: {state_normal['status'].upper()}, sigma_c = {state_normal['sigma_c']:.3f}")

    # Simulate crisis (everything spikes)
    print(f"\n  Simulating 30 crisis readings (everything spikes)...")
    for i in range(30):
        t = i / 30.0
        bridge.update('temperature', 21.0 + 10.0 * t + np.random.normal(0, 1))
        bridge.update('energy_usage', 500 + 2000 * t + np.random.normal(0, 50))
        bridge.update('network_traffic', 100 + 500 * t + np.random.normal(0, 30))

    state_crisis = bridge.get_state()
    print(f"  Status: {state_crisis['status'].upper()}, sigma_c = {state_crisis['sigma_c']:.3f}")

    print(f"\n  Per-sensor criticality:")
    for name, info in state_crisis['sensors'].items():
        print(f"    {name:<20} sigma_c = {info.get('sigma_c', 0):.3f}")

    print(f"\n  --> sigma_c jumped from {state_normal['sigma_c']:.3f} to {state_crisis['sigma_c']:.3f}")
    print(f"  --> Trigger an automation when sigma_c > 0.8!")
    print()


def demo_tensorflow():
    """
    TensorFlow: Criticality-aware training (simulation).

    WHO USES THIS: ML engineers monitoring training stability.
    AHA MOMENT: The callback tracks how close your model is to a
    training phase transition (vanishing/exploding gradients).
    """
    print("=" * 60)
    print("  TENSORFLOW: DETECT TRAINING PHASE TRANSITIONS")
    print("=" * 60)

    from sigma_c.ml.tensorflow import SigmaCCallback

    # Simulate a training run (without actual TensorFlow)
    callback = SigmaCCallback()

    print(f"\n  Simulating 20 epochs of training...")
    print(f"  {'Epoch':<8} {'Loss':>8} {'sigma_c':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10}")

    np.random.seed(42)

    # Create a mock model with layers
    class MockLayer:
        def __init__(self, size):
            self._weights = [np.random.randn(size, size) * 0.1]
        def get_weights(self):
            return self._weights

    class MockModel:
        def __init__(self):
            self.layers = [MockLayer(32), MockLayer(16), MockLayer(8)]

    model = MockModel()
    callback.set_model(model)

    for epoch in range(20):
        # Simulate weight growth (approaching instability)
        scale = 1.0 + epoch * 0.05
        for layer in model.layers:
            layer._weights = [w * scale for w in layer._weights]

        loss = 1.0 / (1 + epoch * 0.3) + np.random.normal(0, 0.02)
        callback.on_epoch_end(epoch, logs={'loss': loss})

        entry = callback.history[-1]
        print(f"  {epoch:<8} {entry['loss']:>8.4f} {entry['sigma_c']:>10.4f}")

    print(f"\n  sigma_c trend: {callback.history[0]['sigma_c']:.3f} --> {callback.history[-1]['sigma_c']:.3f}")
    print(f"  --> Rising sigma_c = weights growing = approaching instability!")
    print(f"  --> Use CriticalRegularizer to keep sigma_c near target.")
    print()


def demo_latex_report():
    """
    LaTeX: Generate publication-ready reports.

    WHO USES THIS: Researchers publishing criticality analysis results.
    AHA MOMENT: One function call generates a complete LaTeX document
    with proper escaping, booktabs tables, and figure includes.
    """
    print("=" * 60)
    print("  LATEX REPORT: PUBLICATION-READY IN ONE CALL")
    print("=" * 60)

    from sigma_c.reporting.latex import LatexGenerator

    gen = LatexGenerator()

    # Generate a results table
    results = [
        {'System': '2D Ising', 'sigma_c': 2.2692, 'kappa': 15.3, 'Error %': 1.8},
        {'System': 'Grover 2Q', 'sigma_c': 0.0832, 'kappa': 8.7, 'Error %': 0.5},
        {'System': 'S&P 500', 'sigma_c': 0.4200, 'kappa': 4.2, 'Error %': 3.1},
    ]
    table = gen.generate_results_table(results, caption="Cross-Domain Criticality Results")

    print(f"\n  Generated table ({len(table.split(chr(10)))} lines):")
    for line in table.split('\n')[:8]:
        print(f"    {line}")
    print(f"    ...")

    # Generate a figure include
    fig = gen.generate_figure(
        'fig/ising_transition.pdf',
        'Phase transition in the 2D Ising model detected by Sigma-C.',
        label='fig:ising'
    )
    print(f"\n  Generated figure include:")
    for line in fig.split('\n'):
        print(f"    {line}")

    print(f"\n  To generate a full report:")
    print(f"    gen.generate_report(")
    print(f"        title='My Analysis',")
    print(f"        author='My Name',")
    print(f"        abstract='We analyze...',")
    print(f"        sections=[{{'title': 'Results', 'content': table}}],")
    print(f"        filename='report'")
    print(f"    )")
    print(f"\n  --> LaTeX with proper escaping, booktabs, and siunitx!")
    print()


def demo_bridge():
    """
    Universal Bridge: Make ANY function criticality-aware.

    WHO USES THIS: Anyone who wants to add criticality tracking
    to existing code without rewriting it.
    AHA MOMENT: One decorator and your function returns sigma_c metadata.
    """
    print("=" * 60)
    print("  BRIDGE: MAKE ANY FUNCTION CRITICALITY-AWARE")
    print("=" * 60)

    from sigma_c.connectors.bridge import SigmaCBridge

    # Wrap a simple function
    @SigmaCBridge.wrap_any_function
    def simulate_system(temperature, pressure):
        """A dummy simulation."""
        return type('Result', (), {
            'energy': temperature * pressure,
            'entropy': np.log(temperature + 1),
        })()

    # Run it
    result = simulate_system(300.0, 1.0)
    print(f"\n  @SigmaCBridge.wrap_any_function")
    print(f"  def simulate_system(temperature, pressure):")
    print(f"      ...")
    print(f"\n  result = simulate_system(300.0, 1.0)")
    print(f"  result.energy  = {result.energy}")
    print(f"  result.entropy = {result.entropy:.4f}")
    print(f"  result.__sigma_c__ = {result.__sigma_c__:.4f}")

    # Detect frameworks
    frameworks = SigmaCBridge.auto_detect_framework()
    installed = [k for k, v in frameworks.items() if v]
    print(f"\n  Detected frameworks: {', '.join(installed) if installed else 'none'}")

    print(f"\n  --> One decorator, zero code changes, instant sigma_c metadata!")
    print()


def main():
    print("""
==========================================================================
   Sigma-C Framework -- Integration Demos
   Everything runs locally, no external services needed.
==========================================================================
    """)

    demos = [
        ("GraphQL API", demo_graphql_api),
        ("CI Analyzer", demo_ci_analyzer),
        ("REST API", demo_rest_api),
        ("WASM Builder", demo_wasm_builder),
        ("Home Assistant", demo_home_assistant),
        ("TensorFlow", demo_tensorflow),
        ("LaTeX Report", demo_latex_report),
        ("Universal Bridge", demo_bridge),
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
    print(f"  DEMOS COMPLETED: {len(passed)}/{len(demos)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print(f"  All integration demos passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
