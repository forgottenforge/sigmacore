#!/usr/bin/env python3
"""
Tests for sigma_c.adapters.linguistics -- LinguisticsAdapter.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.adapters.linguistics import LinguisticsAdapter


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInitEnglish:
    def test_language_english(self):
        adapter = LinguisticsAdapter(language='english')
        assert adapter.language == 'english'


# ------------------------------------------------------------------
# Etymological depth lookup
# ------------------------------------------------------------------

class TestEDLookup:
    def test_water_ed1(self):
        adapter = LinguisticsAdapter(language='english')
        assert adapter.etymological_depth('water') == 1

    def test_beautiful_ed3(self):
        adapter = LinguisticsAdapter(language='english')
        assert adapter.etymological_depth('beautiful') == 3

    def test_algorithm_ed4(self):
        adapter = LinguisticsAdapter(language='english')
        assert adapter.etymological_depth('algorithm') == 4


# ------------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------------

class TestCorrelation:
    def test_positive_correlation(self):
        adapter = LinguisticsAdapter(language='english')
        result = adapter.correlation_analysis()
        assert result['pearson_r'] > 0.3, (
            f"Pearson r = {result['pearson_r']}, expected > 0.3"
        )


# ------------------------------------------------------------------
# Fixed-point test (ED=1 vs ED>1)
# ------------------------------------------------------------------

class TestFixedPoint:
    def test_large_cohens_d(self):
        adapter = LinguisticsAdapter(language='english')
        result = adapter.fixed_point_test()
        assert result['cohens_d'] > 0.8, (
            f"Cohen's d = {result['cohens_d']}, expected > 0.8"
        )


# ------------------------------------------------------------------
# German anchor test
# ------------------------------------------------------------------

class TestGermanAnchorF:
    def test_significant_anova(self):
        adapter = LinguisticsAdapter(language='german')
        result = adapter.german_anchor_test()
        assert result['F_statistic'] > 5, (
            f"F = {result['F_statistic']}, expected > 5"
        )
        assert result['mirror_effect'] is True


# ------------------------------------------------------------------
# Diagnostics and validation
# ------------------------------------------------------------------

class TestDiagnose:
    def test_returns_status(self):
        adapter = LinguisticsAdapter(language='english')
        diag = adapter.diagnose()
        assert 'status' in diag
        assert diag['status'] in ('ok', 'warning', 'error')


class TestValidate:
    def test_all_passed(self):
        adapter = LinguisticsAdapter(language='english')
        val = adapter.validate_techniques()
        assert val['all_passed'] is True, (
            f"Failed checks: {val['failed_checks']}"
        )
