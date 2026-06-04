<!--
Thank you for your contribution! Please fill out the sections below.
Delete sections that do not apply.
-->

## What does this PR do?

A clear description of the change. One concern per PR is preferred.

## Layer

- [ ] `sigma_c_v4` disciplined-reader kernel
- [ ] `sigma_c` v3 multi-adapter applications stack
- [ ] Tooling / docs / CI

## Theorem anchor (v4 kernel changes only)

Which paper label does the new / changed surface cite?

- Paper label: `thm:...` / `prop:...` / `def:...`
- Updated `THEOREM_MAP.md`: yes / no / not needed

## Tests

- [ ] Added or updated tests for this change
- [ ] All tests pass locally (`python -m pytest`)
- [ ] If this is a v4 golden test, the numerical anchor matches the
      foundation paper's printed value (or `verify_examples.py` output)

## Backwards compatibility

- [ ] Public API unchanged
- [ ] Public API change is documented in the PR description and called
      out in the next release notes
- [ ] Breaking change requires a major-version bump

## Checklist

- [ ] I have read the [Contributing guidelines](../CONTRIBUTING.md)
- [ ] I have read the [Code of Conduct](../CODE_OF_CONDUCT.md)
- [ ] I have not added `Co-Authored-By:` trailers for AI assistants
- [ ] My contribution is licensed under the project dual-license
