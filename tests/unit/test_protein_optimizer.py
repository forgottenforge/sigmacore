#!/usr/bin/env python3
"""
Tests for ProteinInterventionOptimizer.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import pytest
import numpy as np

class TestProteinInterventionOptimizer:
    def test_import(self):
        from sigma_c.optimization.protein import ProteinInterventionOptimizer
        assert ProteinInterventionOptimizer is not None

    def test_instantiation(self):
        from sigma_c.optimization.protein import ProteinInterventionOptimizer
        opt = ProteinInterventionOptimizer()
        assert opt is not None

    def test_evaluate_performance(self):
        from sigma_c.optimization.protein import ProteinInterventionOptimizer
        opt = ProteinInterventionOptimizer()
        score = opt._evaluate_performance({'Q_nat': 0.8}, {})
        assert 0.0 <= score <= 1.0

    def test_evaluate_stability(self):
        from sigma_c.optimization.protein import ProteinInterventionOptimizer
        opt = ProteinInterventionOptimizer()
        score = opt._evaluate_stability({'sigma': 0.5}, {})
        assert score > 0  # sigma < 1 means good stability
