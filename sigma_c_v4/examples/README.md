# sigma_c v4 — Hero Examples

Five scenarios that every engineer / scientist / curious reader understands,
covering all three regimes of the foundation paper. Each produces a Card.

| # | Scenario | Regime | Aha effect |
|---|----------|--------|------------|
| 1 | `01_coffee_cooling.py` | I (single mode) | "When does coffee cool by half?" → σ_c equals the cooling time |
| 2 | `02_zipf_wealth.py` | III (no peak) | Wealth distribution is scale-invariant → σ_c = ⊥ is the answer |
| 3 | `03_two_windows.py` | I (probe demo) | Same system, two windows: different σ_c, same τ |
| 4 | `04_bimodal_relaxation.py` | II (multi-mode) | Two relaxation channels → σ_c is vector-valued |
| 5 | `05_ising_chain.py` | I (paper anchor) | 1D Ising correlation length recovered analytically |

## Run

```
cd <repository-root>
python -m sigma_c_v4.examples.01_coffee_cooling
python -m sigma_c_v4.examples.02_zipf_wealth
python -m sigma_c_v4.examples.03_two_windows
python -m sigma_c_v4.examples.04_bimodal_relaxation
python -m sigma_c_v4.examples.05_ising_chain
```

Each example writes a PNG card next to itself.

## Design

The cards follow a FUSE-inspired light layout, with sigma_c-specific
semantics: regime colour-coding (blue = clean single mode, amber = multi-mode,
grey = scale-invariant), provenance badge (analytic vs fitted ρ_⋆), and
a theorem-backing footer that cites the paper labels via `THEOREM_MAP.md`.

Every output either cites a theorem or admits it cannot.
