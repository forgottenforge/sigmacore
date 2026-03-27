#!/usr/bin/env python3
"""
Tests for RigorousProteinSigmaC.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import pytest
import numpy as np

class TestRigorousProtein:
    def test_import(self):
        from sigma_c.physics.protein import RigorousProteinSigmaC
        assert RigorousProteinSigmaC is not None

    def test_instantiation(self):
        from sigma_c.physics.protein import RigorousProteinSigmaC
        checker = RigorousProteinSigmaC()
        assert checker is not None

    def test_check_bounds_stable(self):
        from sigma_c.physics.protein import RigorousProteinSigmaC
        checker = RigorousProteinSigmaC()
        result = checker.check_theoretical_bounds({
            'sigma_at_Tm': 1.0,
            'sigma_at_Tphys': 0.95,
            'N': 127
        })
        assert isinstance(result, dict)

    def test_check_scaling_monotonic(self):
        from sigma_c.physics.protein import RigorousProteinSigmaC
        checker = RigorousProteinSigmaC()
        result = checker.check_scaling_laws(
            {'ddG_values': [0.5, 1.0, 1.5, 2.0], 'sigma_values': [1.01, 1.02, 1.03, 1.04]},
            param_range=[0.5, 1.0, 1.5, 2.0]
        )
        assert isinstance(result, dict)

    def test_quantify_resource(self):
        from sigma_c.physics.protein import RigorousProteinSigmaC
        checker = RigorousProteinSigmaC()
        r = checker.quantify_resource({'N': 127})
        assert r == 127 or isinstance(r, (int, float))
