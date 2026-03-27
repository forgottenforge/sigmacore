#!/usr/bin/env python3
"""
Tests for sigma_c.core.validation -- formal validation tools.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.core.validation import (
    check_boundary_conditions,
    permutation_test,
    peak_clarity_test,
    observable_quality_score,
)


# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------

class TestBoundaryConditionsSigmoid:
    def test_exists_true(self):
        eps = np.linspace(0, 1, 100)
        obs = 1.0 / (1.0 + np.exp(20 * (eps - 0.5)))  # sigmoid decay
        result = check_boundary_conditions(obs, eps)
        assert result['exists'] is True


class TestBoundaryConditionsFlat:
    def test_exists_false(self):
        eps = np.linspace(0, 1, 100)
        obs = np.ones_like(eps) * 0.5  # constant -- no transition
        result = check_boundary_conditions(obs, eps)
        assert result['exists'] is False


# ------------------------------------------------------------------
# Permutation test
# ------------------------------------------------------------------

class TestPermutationSignificant:
    def test_signal_has_low_p(self):
        # Strong monotonic signal with steep gradient -- highly unlikely under permutation
        np.random.seed(42)
        eps = np.linspace(0, 1, 100)
        obs = 1.0 / (1.0 + np.exp(20 * (eps - 0.5)))  # sharp sigmoid
        result = permutation_test(eps, obs, n_permutations=999)
        assert result['p_value'] < 0.05, (
            f"p_value = {result['p_value']}, expected < 0.05 for clear signal"
        )


class TestPermutationNoise:
    def test_noise_has_high_p(self):
        np.random.seed(123)
        eps = np.linspace(0, 1, 50)
        obs = np.random.randn(50)  # pure noise
        result = permutation_test(eps, obs, n_permutations=500)
        assert result['p_value'] > 0.05, (
            f"p_value = {result['p_value']}, expected > 0.05 for pure noise"
        )


# ------------------------------------------------------------------
# Peak clarity
# ------------------------------------------------------------------

class TestPeakClarityPassFail:
    def test_pass(self):
        result = peak_clarity_test(kappa=5.0, kappa_min=3.0)
        assert result['passes'] is True
        assert result['margin'] == pytest.approx(2.0)

    def test_fail(self):
        result = peak_clarity_test(kappa=2.0, kappa_min=3.0)
        assert result['passes'] is False
        assert result['margin'] == pytest.approx(-1.0)


# ------------------------------------------------------------------
# Observable quality
# ------------------------------------------------------------------

class TestObservableQualityGood:
    def test_all_criteria_pass(self):
        eps = np.linspace(0, 1, 100)
        data = np.exp(-5 * eps)  # large range, high SNR, many points
        result = observable_quality_score(data, eps)
        assert result['passes'] is True
        assert result['score'] >= 0.75
        # Verify all individual criteria pass
        for name, criterion in result['criteria'].items():
            assert criterion['passes'], (
                f"Criterion '{name}' failed unexpectedly"
            )
