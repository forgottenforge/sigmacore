# Protein Stability Domain

## Overview

The Protein Stability domain uses the contraction index $\sigma = D \cdot \gamma$ to quantify protein thermodynamic stability and predict disease onset for amyloid-forming mutations. The framework models how destabilizing mutations shift the folding equilibrium toward aggregation-prone intermediates, with sigma serving as a scale-free stability metric.

The central relation is:

$$\sigma = \exp\left(\frac{\Delta\Delta G}{N \cdot R \cdot T}\right)$$

where $\Delta\Delta G$ is the change in free energy of unfolding due to mutation, $N$ is the number of residues, $R$ is the gas constant, and $T$ is temperature. When $\sigma \geq 1$, the native state is no longer thermodynamically dominant and aggregation becomes favorable.

## Key Concepts

### Thermodynamic Sigma

The baseline thermodynamic contraction index for a protein at temperature $T$:

$$\sigma_{\text{thermo}} = \exp\left(\frac{-\Delta G}{N \cdot R \cdot T}\right)$$

This captures the inherent stability of the folded state. For a stable protein at physiological temperature (310 K), $\sigma_{\text{thermo}} < 1$.

### Mutation Effect

A destabilizing mutation ($\Delta\Delta G > 0$) increases $\sigma$ toward the critical threshold of 1.0. The mutational contraction index is:

$$\sigma_{\text{mut}} = \exp\left(\frac{\Delta\Delta G}{N \cdot R \cdot T}\right)$$

Higher $\sigma_{\text{mut}}$ values indicate greater destabilization and earlier expected disease onset.

### Drift Model

The sigma drift model captures the age-dependent accumulation of proteostatic stress:

$$\sigma(t) = \sigma_{\text{baseline}} + r \cdot \frac{(t - 30)}{10}$$

where $r$ is the drift rate per decade (default 0.003) and $t$ is age in years. Disease onset occurs when $\sigma(t) = 1$.

### Negative Controls

The framework explicitly delineates its scope of applicability:

- **In-scope**: Stability-driven amyloid diseases (e.g., TTR, LYZ, GSN), where destabilizing mutations cause aggregation via partial unfolding.
- **Out-of-scope**: Gain-of-function mechanisms (e.g., SOD1/ALS) and templated conversion (e.g., PRNP/prion diseases), where aggregation is not primarily driven by thermodynamic instability.

The `classify_mechanism` method provides explicit scope validation before analysis.

## API Reference

### `ProteinAdapter`

```python
from sigma_c.adapters.protein import ProteinAdapter
```

**Constructor**:
```python
ProteinAdapter(
    protein_name: Optional[str] = None,  # 'TTR', 'LYZ', 'GSN', 'SOD1', 'PRNP'
    N: Optional[int] = None,              # Number of residues
    T: float = 310.0,                     # Temperature in Kelvin
    delta_G_wt: Optional[float] = None,   # Wild-type stability (kcal/mol)
    config: Optional[Dict] = None
)
```

**Core Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | Compute sigma from mutation or delta_G data |
| `sigma_product` | `(D: float, gamma: float) -> float` | Compute sigma = D * gamma |
| `sigma_thermodynamic` | `(delta_G: float, N: int, T: float) -> float` | Thermodynamic contraction index |
| `sigma_mutation` | `(delta_delta_G: float, N: int, T: float) -> float` | Mutational contraction index |
| `sigma_drift` | `(sigma_baseline: float, age: float, rate: float) -> float` | Age-dependent sigma |
| `predict_onset` | `(sigma_baseline: float, rate: float) -> float` | Predict disease onset age |
| `onset_envelope` | `(sigma_baseline: float, rate_range: Tuple) -> Tuple` | Onset range for drift rate range |
| `classify_mechanism` | `(protein_data: dict) -> dict` | Classify disease mechanism |
| `validate_scope` | `(protein_data: dict) -> dict` | Check if sigma analysis applies |
| `analyze_protein` | `(mutations: List[Dict]) -> dict` | Full mutation set analysis with correlations |

**Built-in Datasets** (class attributes):

| Attribute | Description | Size |
|-----------|-------------|------|
| `TTR_MUTATIONS` | Transthyretin mutations with sigma and onset | 25 mutations |
| `LYZ_MUTATIONS` | Lysozyme mutations | 6 mutations |
| `GSN_MUTATIONS` | Gelsolin mutations | 5 mutations |
| `SOD1_MUTATIONS` | SOD1 mutations (negative control) | 10 mutations |
| `PRNP_MUTATIONS` | Prion protein mutations (negative control) | 7 mutations |
| `REFERENCE_PROTEINS` | Small globular protein reference data | 4 proteins |

**Diagnostics Methods** (inherited from `SigmaCAdapter`):

| Method | Signature | Description |
|--------|-----------|-------------|
| `diagnose` | `(data=None, **kwargs) -> Dict` | Check N, delta_G_wt, and data validity |
| `validate` | `(data=None, **kwargs) -> Dict[str, bool]` | Validate technique requirements |
| `explain` | `(result: Dict, **kwargs) -> str` | Generate human-readable explanation |

## Quick Example

```python
from sigma_c.adapters.protein import ProteinAdapter

adapter = ProteinAdapter(protein_name='TTR')
result = adapter.analyze_protein(adapter.TTR_MUTATIONS)
print(f"sigma vs onset rho = {result['correlation']['sigma_vs_onset']['spearman_rho']:.3f}")
```

## Usage Patterns

### Computing Sigma for a Single Mutation

```python
adapter = ProteinAdapter(protein_name='TTR')
sigma = adapter.sigma_mutation(delta_delta_G=1.2, N=127, T=310.0)
print(f"sigma = {sigma:.4f}")  # > 1 indicates destabilized
```

### Predicting Disease Onset

The drift model predicts when sigma reaches the critical threshold of 1.0:

```python
adapter = ProteinAdapter(protein_name='TTR')
sigma_base = adapter.sigma_mutation(1.2, 127)
onset = adapter.predict_onset(sigma_base)
print(f"Predicted onset: {onset:.0f} years")

# Onset envelope for different drift rates
earliest, latest = adapter.onset_envelope(sigma_base, rate_range=(0.02, 0.05))
print(f"Onset window: {earliest:.0f} -- {latest:.0f} years")
```

### Scope Validation Before Analysis

Always check whether the sigma framework applies to your protein system:

```python
adapter = ProteinAdapter(protein_name='SOD1')
mechanism = adapter.classify_mechanism({
    'has_stable_fold': True,
    'gain_of_function': True,
    'mutations_destabilizing': False,
})
print(f"Mechanism: {mechanism['mechanism']}")         # 'gain_of_function'
print(f"Sigma applicable: {mechanism['sigma_applicable']}")  # False
```

### Integration with the Universe Factory

```python
from sigma_c import Universe

adapter = Universe.protein(protein_name='TTR')
result = adapter.analyze_protein(adapter.TTR_MUTATIONS)
```

## Dual-Basin Model

The `DualBasinModel` implements a simplified Monte Carlo simulation with two competing
attractors (native fold and amyloid state). The energy landscape interpolates between them:

E(alpha) = (1 - alpha) * E_nat + alpha * E_amy

At low alpha, the native basin dominates. At high alpha, the amyloid basin takes over.
The contraction index sigma = D * gamma crosses 1.0 before native contacts are lost,
providing an early warning signal.

```python
from sigma_c.adapters.protein import DualBasinModel

model = DualBasinModel(N=30, S=8, contacts=12)
result = model.simulate(alpha=0.3, n_steps=3000, n_trials=10)
print(f"sigma = {result['sigma']:.3f}, Q_nat = {result['Q_nat_mean']:.3f}")

# Find critical alpha
alpha_c = model.find_critical_alpha(n_steps=1000, n_trials=5)
print(f"Critical alpha: {alpha_c:.3f}")
```

## Validation Datasets

### TTR (Transthyretin) -- Primary Validation

25 mutations spanning protective (T119M, $\Delta\Delta G = -0.8$ kcal/mol) to aggressive (L55P, $\Delta\Delta G = 2.5$ kcal/mol) phenotypes.

**Key result**: Spearman correlation between $\sigma$ and observed onset age is $\rho = -0.984$, confirming that higher sigma predicts earlier disease onset.

| Mutation | $\Delta\Delta G$ (kcal/mol) | $\sigma$ | Onset (years) | Phenotype |
|----------|---------------------------|----------|---------------|-----------|
| T119M | -0.8 | 0.917 | -- | Protective |
| V30M | 1.2 | 0.941 | 33 | FAP-I |
| L55P | 2.5 | 0.955 | 20 | FAP-aggressive |
| V122I | 0.5 | 0.932 | 65 | FAC |

### LYZ (Lysozyme) and GSN (Gelsolin) -- Cross-Validation

Six lysozyme mutations and five gelsolin mutations confirm the same $\sigma$--onset relationship in independent amyloid-forming proteins.

### SOD1 and PRNP -- Negative Controls

- **SOD1** (10 mutations): ALS is primarily a gain-of-function disease. The $\sigma$ framework is not expected to predict onset accurately, and the `classify_mechanism` method correctly identifies this as out-of-scope.
- **PRNP** (7 mutations): Prion diseases involve templated conversion. The framework correctly flags this mechanism as outside the scope of thermodynamic stability analysis.

These negative controls demonstrate that the framework does not overfit to any protein system indiscriminately.

### Verification Code

```python
adapter = ProteinAdapter(protein_name='TTR')

# Primary validation
result = adapter.analyze_protein(adapter.TTR_MUTATIONS)
rho = result['correlation']['sigma_vs_onset']['spearman_rho']
assert rho < -0.9, "Strong negative correlation expected"

# Negative control
mechanism = adapter.classify_mechanism({
    'has_stable_fold': True,
    'gain_of_function': True
})
assert mechanism['sigma_applicable'] is False
```
