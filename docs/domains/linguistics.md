# Computational Linguistics Domain

## Overview

The Computational Linguistics domain investigates the relationship between **etymological depth** (ED) and **semantic change** over time. The central finding is that words with deeper etymological layering undergo more semantic change, even after controlling for word frequency. This mirrors the contraction geometry principle: higher morphological complexity (analogous to higher $D$) correlates with greater semantic drift (analogous to higher $\gamma$).

The framework provides embedded word lists for English and German, with statistical tools for correlation analysis, mediation analysis, and cross-linguistic comparison.

## Key Concepts

### Etymological Depth (ED)

ED measures the number of morphological derivation layers a word has undergone from its etymological root:

| ED | Description | Examples |
|----|-------------|----------|
| 1 | Proto-roots, core vocabulary | *I, you, eye, sun, water, fire* |
| 2 | Single derivation or borrowing | *husband, kingdom, freedom, army* |
| 3 | Double derivation | *beautiful, wonderful, discover* |
| 4 | Triple derivation or complex compounds | *unfortunately, communication* |
| 5 | Extreme layering | *trivial, disaster, preposterous* |

### Procrustes Alignment

Semantic change is measured by comparing word embedding vectors across historical time periods. The Procrustes alignment method rotates embedding spaces to maximize overlap, then computes the cosine distance for each word between aligned spaces. Higher distance indicates greater semantic shift.

### Mediation Analysis

The Baron-Kenny mediation framework tests whether word frequency mediates the ED--change relationship:

$$\text{ED} \xrightarrow{c} \text{Change}$$

$$\text{ED} \xrightarrow{a} \text{Frequency} \xrightarrow{b} \text{Change}$$

The indirect effect ($a \times b$) captures the portion of the ED--change relationship that operates through frequency, while the direct effect ($c'$) captures the portion that is independent of frequency.

### Transparency Paradox

Among derived words (ED $\geq$ 2), morphologically **transparent** words (whose internal structure is visible, e.g., *teacher* = *teach* + *-er*) show **less** semantic change than **opaque** words (whose etymology has been obscured, e.g., *salary* from Latin *salarium*). This is called the transparency paradox because transparency anchors meaning to the compositional parts.

## API Reference

### `LinguisticsAdapter`

```python
from sigma_c.adapters.linguistics import LinguisticsAdapter
```

**Constructor**:
```python
LinguisticsAdapter(
    language: str = 'english',  # 'english', 'german', or 'french'
    config: Optional[Dict] = None
)
```

**Word-Level Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_observable` | `(data, **kwargs) -> float` | Semantic change for a word or data dict |
| `etymological_depth` | `(word: str) -> Optional[int]` | Look up ED of a word (1--5) |
| `semantic_change` | `(word: str) -> Optional[float]` | Look up semantic change magnitude |

**Statistical Analysis Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `correlation_analysis` | `(ed_values, change_values) -> Dict` | Pearson and Spearman correlations |
| `fixed_point_test` | `(ed_values, change_values) -> Dict` | Welch t-test: ED=1 vs ED>1 |
| `mediation_analysis` | `(ed_values, freq_values, change_values) -> Dict` | Baron-Kenny mediation (ED -> Freq -> Change) |
| `transparency_effect` | `(change_values, transparency_labels) -> Dict` | Compare transparent vs opaque words (ED >= 2) |
| `german_anchor_test` | `() -> Dict` | ANOVA across German P/T/O categories |
| `run_full_analysis` | `() -> Dict` | Run all analyses and return combined results |

**Return Dictionaries**:

`correlation_analysis` returns:
- `pearson_r`, `pearson_p`: Pearson correlation and p-value
- `spearman_rho`, `spearman_p`: Spearman rank correlation and p-value
- `n`: Sample size

`fixed_point_test` returns:
- `t_statistic`, `p_value`: Welch t-test results
- `cohens_d`: Effect size
- `mean_ed1`, `mean_ed_gt1`: Group means

`mediation_analysis` returns:
- `total_effect` ($c$), `direct_effect` ($c'$), `indirect_effect` ($c - c'$)
- `indirect_pct`: Percentage mediated
- `sobel_z`, `sobel_p`: Sobel test for mediation significance

`transparency_effect` returns:
- `cohens_d`, `p_value`: Effect size and significance
- `mean_transparent`, `mean_opaque`: Group means
- `direction`: Comparison result string

`german_anchor_test` returns:
- `F_statistic`, `p_value`: One-way ANOVA results
- `pairwise`: Dict of pairwise Welch t-tests (P vs T, P vs O, T vs O)
- `mirror_effect`: Boolean, whether mean(P) < mean(T) < mean(O)

**Diagnostics Methods** (inherited from `SigmaCAdapter`):

| Method | Signature | Description |
|--------|-----------|-------------|
| `diagnose` | `(data=None, **kwargs) -> Dict` | Check dataset balance and outliers |
| `validate` | `(data=None, **kwargs) -> Dict[str, bool]` | Validate sample size and ED coverage |
| `explain` | `(result: Dict, **kwargs) -> str` | Generate human-readable explanation |

## Advanced Methods (v3.0)

### Procrustes Alignment

For users who provide their own historical word embeddings, the adapter includes
orthogonal Procrustes alignment:

```python
R = adapter.orthogonal_procrustes(W_1850, W_2000, n_anchors=5000)
change = adapter.aligned_cosine_distance(v_old, v_new, R)
```

### Three-Regime Model

Extends ED analysis by classifying words into three stability regimes:

```python
result = adapter.three_regime_model()
# result['regimes']: {'prime': {...}, 'opaque': {...}, 'transparent': {...}}
# result['anova_F']: F-statistic for regime differences
```

### Cross-Linguistic Validation

```python
french = adapter.french_replication()  # n=140, r ~ 0.37
comparison = adapter.cross_linguistic_comparison()  # EN + DE + FR summary
```

## Quick Example

```python
from sigma_c.adapters.linguistics import LinguisticsAdapter

adapter = LinguisticsAdapter(language='english')
results = adapter.run_full_analysis()
print(f"ED-change correlation: r = {results['correlation']['pearson_r']:.3f}")
```

## Usage Patterns

### Looking Up Individual Words

```python
adapter = LinguisticsAdapter(language='english')

# ED=1: core vocabulary, minimal change
print(adapter.etymological_depth('water'))     # 1
print(adapter.semantic_change('water'))         # low value (~0.3)

# ED=4: highly derived, more change
print(adapter.etymological_depth('communication'))  # 4
print(adapter.semantic_change('communication'))     # higher value (~0.5)
```

### Running Individual Analyses

Each analysis can be run independently with custom data or with embedded defaults:

```python
adapter = LinguisticsAdapter(language='english')

# Correlation only
corr = adapter.correlation_analysis()
print(f"Spearman rho = {corr['spearman_rho']:.3f}, p = {corr['spearman_p']:.2e}")

# Mediation analysis
med = adapter.mediation_analysis()
print(f"Indirect effect: {med['indirect_pct']:.1f}% mediated through frequency")
```

### German Cross-Linguistic Test

The German data uses a different categorization scheme (P/T/O) that provides an independent test of the same underlying phenomenon:

```python
adapter = LinguisticsAdapter(language='english')  # German data is always available
ga = adapter.german_anchor_test()
print(f"ANOVA F = {ga['F_statistic']:.2f}, p = {ga['p_value']:.2e}")
print(f"Mirror effect (P < T < O): {ga['mirror_effect']}")
```

### Integration with the Universe Factory

```python
from sigma_c import Universe

adapter = Universe.linguistics(language='english')
results = adapter.run_full_analysis()
```

## Cross-Linguistic Validation

### English

The embedded English dataset contains 225 words across ED levels 1--5:

- **ED 1**: 113 words (proto-roots)
- **ED 2**: 56 words (single derivation)
- **ED 3**: 29 words (double derivation)
- **ED 4**: 23 words (triple derivation)
- **ED 5**: 4 words (extreme layering)

Mean semantic change increases monotonically with ED (0.333, 0.423, 0.497, 0.524, 0.565).

### German Mirror Effect

The German analysis uses three etymological categories that mirror the English ED levels:

| Category | Description | Mean Change |
|----------|-------------|-------------|
| P (Primes) | Native Germanic roots | 0.420 |
| T (Transparent) | Transparently derived compounds | 0.486 |
| O (Opaque) | Borrowings and opaque derivations | 0.530 |

The **mirror effect** ($P < T < O$) holds: words with less visible internal structure undergo more semantic change. This is verified via one-way ANOVA across the three categories.

### Verification Code

```python
adapter = LinguisticsAdapter(language='english')
results = adapter.run_full_analysis()

# ED-change correlation is positive and significant
assert results['correlation']['pearson_r'] > 0
assert results['correlation']['pearson_p'] < 0.05

# ED=1 words change less than ED>1 words
assert results['fixed_point']['cohens_d'] > 0

# German mirror effect holds
assert results['german_anchor']['mirror_effect'] is True
```
