#!/usr/bin/env python3
"""
Tests for sigma_c.core.derivatives -- derivative estimation module.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.core.derivatives import (
    savitzky_golay_derivative,
    spline_derivative,
    gp_regression_derivative,
    select_best_method,
    compute_derivative,
)


# ------------------------------------------------------------------
# Helper: generate sin/cos test data
# ------------------------------------------------------------------

def _sin_data(n=200, noise_std=0.0):
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + noise_std * np.random.RandomState(0).randn(n)
    true_deriv = np.cos(x)
    return x, y, true_deriv


# ------------------------------------------------------------------
# Individual methods on sin(x) -> cos(x)
# ------------------------------------------------------------------

class TestSavitzkyGolaySin:
    def test_derivative_of_sin_matches_cos(self):
        x, y, true_deriv = _sin_data(n=200)
        dy = savitzky_golay_derivative(x, y, window_length=11, polyorder=3)
        # Check that the middle region agrees within 0.15
        mid = slice(20, 180)
        max_err = np.max(np.abs(dy[mid] - true_deriv[mid]))
        assert max_err < 0.15, f"Max error {max_err} too large for savitzky_golay"


class TestSplineSin:
    def test_derivative_of_sin_matches_cos(self):
        x, y, true_deriv = _sin_data(n=200)
        dy = spline_derivative(x, y)
        mid = slice(20, 180)
        max_err = np.max(np.abs(dy[mid] - true_deriv[mid]))
        assert max_err < 0.15, f"Max error {max_err} too large for spline"


class TestGPSin:
    def test_derivative_of_sin_matches_cos(self):
        # GP is more expensive; use fewer points  # timeout: < 5s with n=60
        x, y, true_deriv = _sin_data(n=60)
        dy_mean, dy_std = gp_regression_derivative(x, y, noise_level=0.01)
        mid = slice(10, 50)
        max_err = np.max(np.abs(dy_mean[mid] - true_deriv[mid]))
        assert max_err < 0.3, f"Max error {max_err} too large for GP"


# ------------------------------------------------------------------
# Automatic method selection
# ------------------------------------------------------------------

class TestAutoSelectionClean:
    def test_selects_savitzky_golay_for_clean_data(self):
        x, y, _ = _sin_data(n=200, noise_std=0.0)
        method = select_best_method(x, y)
        assert method == 'savitzky_golay'


# ------------------------------------------------------------------
# Unified compute_derivative interface
# ------------------------------------------------------------------

class TestComputeDerivativeGaussian:
    def test_backward_compatible(self):
        x, y, _ = _sin_data(n=100)
        result = compute_derivative(x, y, method='gaussian')
        assert 'derivative' in result
        assert result['method_used'] == 'gaussian'
        assert len(result['derivative']) == len(x)


class TestComputeDerivativeAuto:
    def test_works(self):
        x, y, _ = _sin_data(n=100)
        result = compute_derivative(x, y, method='auto')
        assert 'derivative' in result
        assert result['method_used'] in ('savitzky_golay', 'spline', 'gp')
        assert len(result['derivative']) == len(x)
