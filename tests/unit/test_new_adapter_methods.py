#!/usr/bin/env python3
"""
Tests for newly added adapter methods (piecewise, DualBasin, Procrustes, etc).
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import pytest
import numpy as np

class TestNumberTheoryPiecewise:
    def test_piecewise_k0_is_collatz(self):
        from sigma_c.adapters.number_theory import NumberTheoryAdapter
        nt = NumberTheoryAdapter()
        # k=0 should give same as single-step Collatz
        from sigma_c.core.contraction import single_step_map
        for n in [3, 5, 7, 11, 13]:
            assert nt.piecewise_map(n, 0) == single_step_map(n, 3, 1)

    def test_piecewise_k8_is_5n1(self):
        from sigma_c.adapters.number_theory import NumberTheoryAdapter
        from sigma_c.core.contraction import odd_part
        nt = NumberTheoryAdapter()
        for n in [3, 5, 7, 11, 13]:
            assert nt.piecewise_map(n, 8) == odd_part(5 * n + 1)

    def test_gamma_interpolation_crosses_one(self):
        from sigma_c.adapters.number_theory import NumberTheoryAdapter
        nt = NumberTheoryAdapter()
        gi = nt.gamma_interpolation(M=10)
        assert gi['critical_k'] is not None
        assert 2.0 < gi['critical_k'] < 7.0

class TestLinguisticsNewMethods:
    def test_three_regime_model(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter(language='english')
        tr = la.three_regime_model()
        assert 'regimes' in tr
        assert 'prime' in tr['regimes']
        assert tr['anova_F'] > 1.0

    def test_french_replication(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter(language='english')
        fr = la.french_replication()
        assert fr['n'] == 140
        assert fr['pearson_r'] > 0  # Positive correlation expected

    def test_cross_linguistic(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter(language='english')
        cl = la.cross_linguistic_comparison()
        assert 'english' in cl
        assert 'german' in cl
        assert 'french' in cl

    def test_procrustes_orthogonal(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter()
        W1 = np.random.randn(50, 20)
        W2 = np.random.randn(50, 20)
        R = la.orthogonal_procrustes(W1, W2)
        assert R.shape == (20, 20)
        # Check orthogonality: R @ R.T should be close to I
        assert np.allclose(R @ R.T, np.eye(20), atol=0.01)

    def test_aligned_cosine_identical(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter()
        v = np.random.randn(20)
        R = np.eye(20)
        d = la.aligned_cosine_distance(v, v, R)
        assert d < 0.01  # Identical vectors should have ~0 distance

class TestProteinNewData:
    def test_cross_protein_data_exists(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter()
        assert len(pa.CROSS_PROTEIN_VALIDATION) >= 15

    def test_app_vus_data_exists(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter()
        assert len(pa.APP_VUS_PREDICTIONS) >= 10

    def test_cross_protein_all_have_ddG(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter()
        for entry in pa.CROSS_PROTEIN_VALIDATION:
            assert 'ddG' in entry
            assert 'N' in entry
