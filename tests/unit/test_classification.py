#!/usr/bin/env python3
"""
Tests for sigma_c.core.classification -- four-type classification module.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.core.classification import (
    MapType,
    classify_operation,
    analyze_type_d,
    analyze_type_o,
    analyze_type_s,
    analyze_type_r,
)


# ------------------------------------------------------------------
# MapType enum
# ------------------------------------------------------------------

class TestMapTypeEnum:
    def test_all_four_values_exist(self):
        assert MapType.DISSIPATIVE.value == "D"
        assert MapType.OVERSATURATED.value == "O"
        assert MapType.SYMMETRIC.value == "S"
        assert MapType.REVERSIBLE.value == "R"


# ------------------------------------------------------------------
# classify_operation
# ------------------------------------------------------------------

class TestClassifyDissipative:
    def test_D_greater_than_1(self):
        result = classify_operation(D=1.5, gamma=0.75)
        assert result == MapType.DISSIPATIVE


class TestClassifyOversaturated:
    def test_growing_preimage(self):
        result = classify_operation(D=1.5, gamma=0.75, has_growing_preimage=True)
        assert result == MapType.OVERSATURATED


class TestClassifySymmetric:
    def test_bijective_with_symmetry(self):
        result = classify_operation(D=1.0, is_bijective=True, has_symmetry=True)
        assert result == MapType.SYMMETRIC


class TestClassifyReversible:
    def test_bijective_without_symmetry(self):
        result = classify_operation(D=1.0, is_bijective=True, has_symmetry=False)
        assert result == MapType.REVERSIBLE


# ------------------------------------------------------------------
# Type D analysis
# ------------------------------------------------------------------

class TestTypeDConvergent:
    def test_type_d_convergent(self):
        result = analyze_type_d(D=1.7, gamma=0.75, has_cycles=False)
        assert result.prediction == 'convergent'
        d = result.to_dict()
        assert d['type'] == 'D'
        assert d['gamma'] < 1.0


class TestTypeDDivergent:
    def test_type_d_divergent(self):
        result = analyze_type_d(D=1.4, gamma=1.25, has_cycles=False)
        assert result.prediction == 'divergent'


class TestTypeDCycles:
    def test_type_d_cycles(self):
        result = analyze_type_d(D=1.33, gamma=0.75, has_cycles=True)
        assert result.prediction == 'convergent_to_cycles'
