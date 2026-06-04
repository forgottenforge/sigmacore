# Contributing to Sigma-C Framework

Thank you for your interest in contributing. The project is dual-licensed
(AGPL-3.0-or-later OR Commercial) and is developed at ForgottenForge.

## Before you start

- Read the **[Code of Conduct](CODE_OF_CONDUCT.md)** — it applies to all
  interactions on this repository.
- Skim the **[README](README.md)** to understand the two layers
  (`sigma_c_v4/` disciplined kernel + `sigma_c/` v3 multi-adapter stack)
  and pick the one your change belongs to.
- Familiarise yourself with the foundation paper:
  Wurm, M. C. (2026). *Operational scale selection: axioms, spectral
  concentration, and a regime trichotomy.* Zenodo,
  [doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818).
  v4 surfaces must map to a labelled theorem/proposition/definition in
  the paper (see [`THEOREM_MAP.md`](THEOREM_MAP.md)).

## How to contribute

### Bug reports

Use the **Bug report** issue template. Include:

- Sigma-C version (`python -c "import sigma_c_v4; print(sigma_c_v4.__version__)"`)
- Operating system and Python version
- Minimal reproducible example (≤ 30 lines)
- Expected vs observed behaviour

### Feature requests

Use the **Feature request** issue template. For v4 kernel additions,
specify which theorem/proposition the new surface would cite. If there
is no paper anchor for a proposed v4 feature, the discipline is to
**not ship it in v4**; consider a v3-stack adapter instead, or open a
discussion for a foundation-paper extension.

### Pull requests

1. Fork the repo and create a feature branch from `main`:
   `git checkout -b feat/short-description`
2. Make your change. Keep it focused — one concern per PR.
3. Run the test suite locally:
   ```bash
   python -m pytest sigma_c_v4/tests/ -v
   python -m pytest tests/ -v          # v3 tests
   ```
4. For v4 kernel changes: if you add a new public surface, add a
   `cite("label:name")` reference in its docstring pointing at the
   paper label that backs it. Add or update `THEOREM_MAP.md` if the
   label is new.
5. For v3 adapter changes: keep public API backwards-compatible unless
   bumping a major version.
6. Commit with a clear message (see *Commit style* below).
7. Open the PR using the pull-request template.

### Commit style

- One concern per commit.
- First line: short imperative ≤ 72 chars, e.g.
  `core/faithfulness: add explicit C_R bound for F3 condition`.
- Body (optional): why, not what.
- Sign-off if you can: `git commit -s` adds `Signed-off-by:`.
- **Do not add `Co-Authored-By:` trailers for AI assistants.** AI
  assistance is declared once in the README acknowledgements and in
  the paper's AI-assistance statement, not as a per-commit attribution.

### Tests

- New v4 surfaces require a test in `sigma_c_v4/tests/` that either
  reproduces a paper-printed numerical value (golden test) or asserts a
  discipline invariant (provenance, regime, citation).
- New v3 adapters require at least one functional test in `tests/`.
- Do not commit golden tests that rely on private datasets.

### Code style

- Python: PEP 8, line length 100. We do not enforce a formatter, but
  consistency with the surrounding file is appreciated.
- Type hints encouraged on public surfaces.
- Docstrings: short, paper-anchored where applicable. Cite via
  `cite("thm:foo")` from `sigma_c_v4.theorem_map`.

## Development setup

```bash
git clone https://github.com/forgottenforge/sigmacore
cd sigmacore
python -m venv .venv
source .venv/bin/activate         # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
python -m pytest
```

## Reporting security issues

Do not open a public issue for security problems. See
[`SECURITY.md`](SECURITY.md) for responsible-disclosure instructions.

## License of contributions

By submitting a contribution, you agree to license it under the
dual-license model described in [`LICENSE.txt`](LICENSE.txt). For
substantial contributions we may ask you to sign a contributor license
agreement (CLA) before merging.
