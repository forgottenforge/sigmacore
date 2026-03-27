# Sigma-C Framework v1.2.3 User Guide

Welcome to the **Sigma-C Framework**, the universal standard for critical susceptibility optimization.

This guide is designed to take you from installation to mastering advanced optimization techniques across Quantum, GPU, Financial, and Machine Learning domains.

## New in v3.0

Version 3.0 expands Sigma-C with three new optimization domains and a powerful analytical layer:

- **Number Theory Domain** -- Analyze iterative maps (Collatz-type) with sigma_c susceptibility analysis.
- **Protein Stability Domain** -- Optimize protein mutation stability using thermodynamic sigma_c metrics.
- **Computational Linguistics Domain** -- Measure language model robustness via distributional stability.
- **Contraction Geometry** -- A new theoretical framework (D, gamma, sigma product) that explains *why* systems converge, complementing sigma_c which shows *where* they are critical. See [Core Concepts](docs/02_core_concepts.md) for details.

## Number Theory Domain

Analyze iterative arithmetic maps using contraction geometry and sigma_c:

```python
from sigma_c_framework.sigma_c.adapters.number_theory import NumberTheoryAdapter

adapter = NumberTheoryAdapter(q=3, c=1, modulus=2)
result = adapter.analyze_map(start_values=range(1, 1000))
print(f"sigma_c={result.sigma_c:.4f}, D={result.D:.3f}, gamma={result.gamma:.3f}")
print(f"Classification: {result.map_type}")
```

See detailed documentation: [docs/domains/number_theory.md](docs/domains/number_theory.md)

## Protein Stability Domain

Evaluate protein mutation effects on thermodynamic stability:

```python
from sigma_c_framework.sigma_c.adapters.protein import ProteinStabilityAdapter

adapter = ProteinStabilityAdapter(pdb_id="1UBQ")
result = adapter.analyze_mutations(mutations=["A28G", "L43V", "I61A"])
print(f"sigma_c={result.sigma_c:.4f}, ddG_mean={result.ddG_mean:.2f} kcal/mol")
print(f"Stability verdict: {result.verdict}")
```

See detailed documentation: [docs/domains/protein.md](docs/domains/protein.md)

## Computational Linguistics Domain

Measure language model robustness under distributional perturbations:

```python
from sigma_c_framework.sigma_c.adapters.linguistics import LinguisticsAdapter

adapter = LinguisticsAdapter(model="gpt2", corpus="wikitext-103")
result = adapter.analyze_robustness(perturbation_types=["typo", "synonym", "deletion"])
print(f"sigma_c={result.sigma_c:.4f}, perplexity_shift={result.perplexity_shift:.2f}")
print(f"Most vulnerable to: {result.weakest_perturbation}")
```

See detailed documentation: [docs/domains/linguistics.md](docs/domains/linguistics.md)

## 📚 Table of Contents

### 1. [Getting Started](docs/01_getting_started.md)
- **Installation**: Setup via pip or source.
- **Quickstart**: Your first optimization in 5 minutes.
- **CLI Basics**: Using the `sigma-c` command line tool.

### 2. [Core Concepts](docs/02_core_concepts.md)
- **The Philosophy**: Balancing Performance vs. Stability.
- **Understanding $\sigma_c$**: The math behind the metric.
- **Architecture**: How the Universal Optimizer works.

### 3. [Configuration](docs/03_configuration.md)
- **Global Config**: `config.yaml` and environment variables.
- **Hardware Setup**: Configuring AWS Braket, NVIDIA drivers, and API keys.

### 4. Domain Guides
Deep dives into specific optimization domains:
- **[Quantum Optimization](docs/domains/quantum.md)**: Hardware-aware compilation, noise models, Grover/QAOA.
- **[GPU Optimization](docs/domains/gpu.md)**: Kernel tuning, Roofline analysis, thermal management.
- **[Financial Optimization](docs/domains/financial.md)**: Trading strategies, risk models, market data.
- **[Machine Learning](docs/domains/ml.md)**: Hyperparameter tuning, robustness, architecture search.

### 5. Advanced Topics
- **[Callbacks System](docs/advanced/callbacks.md)**: Customizing the optimization lifecycle.
- **[Visualization](docs/advanced/visualization.md)**: Interpreting landscapes and Pareto frontiers.
- **[Extending Sigma-C](docs/advanced/extending.md)**: Creating custom adapters and optimizers.

### 6. [API Reference](docs/06_api_reference.md)
- Detailed class and function documentation.

---

## 🆘 Support
- **Issues**: [GitHub Issues](https://github.com/forgottenforge/sigmacore/issues)
- **Contact**: nfo@forgottenforge.xyz

Copyright (c) 2025 ForgottenForge.xyz
