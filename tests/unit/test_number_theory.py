#!/usr/bin/env python3
"""
Tests for sigma_c.adapters.number_theory -- NumberTheoryAdapter.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.adapters.number_theory import NumberTheoryAdapter
from sigma_c.core.contraction import embedding_depth, single_step_map


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInit:
    def test_creates_adapter(self):
        adapter = NumberTheoryAdapter()
        assert adapter is not None
        assert adapter.q == 3
        assert adapter.c == 1


# ------------------------------------------------------------------
# Contraction defect
# ------------------------------------------------------------------

class TestDM:
    def test_D_8_above_threshold(self):
        adapter = NumberTheoryAdapter()
        D = adapter.compute_D_M(8)
        assert D >= 1.5, f"D_8 = {D}, expected >= 1.5"


# ------------------------------------------------------------------
# Drift
# ------------------------------------------------------------------

class TestGamma:
    def test_gamma_12_near_expected(self):
        # For the Collatz cycle map, gamma at M=12 should be near 9/16 = 0.5625
        adapter = NumberTheoryAdapter(map_type='collatz')
        gamma = adapter.compute_gamma_M(12)
        assert abs(gamma - 0.5625) < 0.15, (
            f"gamma_12 = {gamma}, expected near 0.5625"
        )


# ------------------------------------------------------------------
# Behavior prediction
# ------------------------------------------------------------------

class TestPredictConvergent:
    def test_collatz_convergent(self):
        adapter = NumberTheoryAdapter(map_type='collatz')
        result = adapter.predict_behavior()
        assert result['prediction'] == 'convergent', (
            f"Expected 'convergent', got '{result['prediction']}'"
        )


# ------------------------------------------------------------------
# Twelve map predictions
# ------------------------------------------------------------------

class TestTwelvePredictions:
    def test_all_correct(self):
        # timeout: M=12 sweep over 12 maps, each fast
        adapter = NumberTheoryAdapter()
        result = adapter.verify_twelve_predictions(M=12)
        assert result['success_rate'] == 1.0, (
            f"success_rate = {result['success_rate']}, expected 1.0. "
            f"Failed: {[r for r in result['results'] if not r['match']]}"
        )


# ------------------------------------------------------------------
# Countdown analysis
# ------------------------------------------------------------------

class TestCountdown:
    def test_ed_decreases_properly(self):
        """Countdown decomposition should show ed decreasing in countdown phases."""
        adapter = NumberTheoryAdapter(map_type='collatz_single')
        result = adapter.analyze_countdown(7)
        phases = result['phases']
        for phase in phases:
            if phase['phase'] == 'countdown':
                assert phase['ed_start'] >= 2
                assert phase['ed_end'] <= phase['ed_start']


# ------------------------------------------------------------------
# Reset distribution (Geo(1/2))
# ------------------------------------------------------------------

class TestResetGeo:
    def test_chi_squared_passes(self):
        # Stochastic test: run with larger M for better convergence
        adapter = NumberTheoryAdapter(map_type='collatz_single')
        result = adapter.verify_reset_distribution(M=12, n_samples=5000)
        # Use lenient threshold: p > 0.001 (very conservative)
        assert result['p_value'] > 0.001, (
            f"Chi-squared test rejected Geo(1/2): p={result['p_value']}"
        )


# ------------------------------------------------------------------
# Diagnostics and validation
# ------------------------------------------------------------------

class TestDiagnose:
    def test_returns_status(self):
        adapter = NumberTheoryAdapter()
        diag = adapter.diagnose()
        assert 'status' in diag
        assert diag['status'] in ('ok', 'warning', 'error')


class TestValidate:
    def test_all_passed(self):
        adapter = NumberTheoryAdapter()
        val = adapter.validate_techniques()
        assert val['all_passed'] is True, (
            f"Failed checks: {val['failed_checks']}"
        )
