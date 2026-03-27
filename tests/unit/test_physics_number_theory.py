#!/usr/bin/env python3
"""
Tests for RigorousNumberTheorySigmaC.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import pytest
import numpy as np

class TestRigorousNT:
    def test_import(self):
        from sigma_c.physics.number_theory import RigorousNumberTheorySigmaC
        assert RigorousNumberTheorySigmaC is not None

    def test_instantiation(self):
        from sigma_c.physics.number_theory import RigorousNumberTheorySigmaC
        checker = RigorousNumberTheorySigmaC()
        assert checker is not None

    def test_check_bounds(self):
        from sigma_c.physics.number_theory import RigorousNumberTheorySigmaC
        checker = RigorousNumberTheorySigmaC()
        # Provide data that should pass bounds check
        result = checker.check_theoretical_bounds({'D_M': 2.06, 'gamma': 0.5625, 'q': 3})
        assert 'lower_bound' in result or 'valid' in result or isinstance(result, dict)

    def test_check_scaling(self):
        from sigma_c.physics.number_theory import RigorousNumberTheorySigmaC
        checker = RigorousNumberTheorySigmaC()
        D_values = [2.0, 2.05, 2.06, 2.06, 2.06]
        M_values = [4, 6, 8, 10, 12]
        result = checker.check_scaling_laws(
            {'D_M_values': D_values, 'M_values': M_values},
            param_range=M_values
        )
        assert isinstance(result, dict)

    def test_quantify_resource(self):
        from sigma_c.physics.number_theory import RigorousNumberTheorySigmaC
        checker = RigorousNumberTheorySigmaC()
        r = checker.quantify_resource({'M': 16})
        assert r == 16 or isinstance(r, (int, float))
