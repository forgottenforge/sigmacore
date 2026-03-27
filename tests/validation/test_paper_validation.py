#!/usr/bin/env python3
"""
Paper validation tests: Reproduce key results from the papers.
These tests verify that the framework produces results consistent
with published findings.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import pytest
import numpy as np

class TestPaper1CBFI:
    """Reproduce key results from the contraction geometry paper."""

    def test_twelve_map_predictions_all_correct(self):
        from sigma_c.adapters.number_theory import NumberTheoryAdapter
        nt = NumberTheoryAdapter()
        result = nt.verify_twelve_predictions(M=10)
        assert result['success_rate'] == 1.0, f"Only {result['correct']}/{result['total']} correct"

    def test_D_M_lower_bound_4_3(self):
        from sigma_c.core.contraction import compute_contraction_defect, cycle_map
        for M in range(3, 13):
            D = compute_contraction_defect(cycle_map, M)
            assert D >= 4/3 - 0.01, f"D_{M}={D:.4f} < 4/3"

    def test_gamma_converges_to_q_over_4(self):
        from sigma_c.core.contraction import compute_drift, single_step_map
        for q in [3, 5, 7]:
            f = lambda n, q=q: single_step_map(n, q, 1)
            gamma = compute_drift(f, 14)
            expected = q / 4
            assert abs(gamma - expected) < 0.05, f"q={q}: gamma={gamma:.4f}, expected {expected}"

class TestPaper5Protein:
    """Reproduce key results from the protein stability paper."""

    def test_ttr_correlation_strongly_negative(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter(protein_name='TTR')
        result = pa.analyze_protein(pa.TTR_MUTATIONS)
        corr = result['correlation']
        if 'sigma_vs_onset' in corr:
            rho = corr['sigma_vs_onset']['spearman_rho']
            assert rho < -0.9, f"TTR rho={rho}, expected < -0.9"

    def test_sod1_negative_control_not_significant(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter(protein_name='SOD1')
        result = pa.analyze_protein(pa.SOD1_MUTATIONS)
        corr = result['correlation']
        if 'sigma_vs_onset' in corr:
            p = corr['sigma_vs_onset']['p_value']
            assert p > 0.05, f"SOD1 p={p}, should not be significant"

    def test_sigma_at_Tm_equals_one(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter()
        sigma = pa.sigma_thermodynamic(0.0, 100)  # dG=0 at T_m
        assert abs(sigma - 1.0) < 1e-10

    def test_destabilizing_mutations_sigma_above_one(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter(protein_name='TTR')
        for mut in pa.TTR_MUTATIONS:
            if mut['ddG'] > 0:
                sigma = pa.sigma_mutation(mut['ddG'], pa.N)
                assert sigma > 1.0, f"{mut['name']} with ddG={mut['ddG']} has sigma={sigma}"

class TestPaper4Linguistics:
    """Reproduce key results from the linguistics paper."""

    def test_ed_change_correlation_significant(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter(language='english')
        corr = la.correlation_analysis()
        assert corr['pearson_r'] > 0.3
        assert corr['pearson_p'] < 0.001

    def test_fixed_point_large_effect(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter(language='english')
        fp = la.fixed_point_test()
        assert fp['cohens_d'] > 0.8

    def test_german_mirror_effect(self):
        from sigma_c.adapters.linguistics import LinguisticsAdapter
        la = LinguisticsAdapter(language='german')
        ga = la.german_anchor_test()
        assert ga['mirror_effect'] is True
        assert ga['F_statistic'] > 5.0

class TestNegativeControls:
    """Verify that framework correctly identifies out-of-scope cases."""

    def test_sod1_wrong_mechanism(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter(protein_name='SOD1')
        mech = pa.classify_mechanism({
            'has_stable_fold': True,
            'mutations_destabilizing': True,
            'gain_of_function': True
        })
        assert mech['mechanism'] == 'gain_of_function'

    def test_prnp_templated(self):
        from sigma_c.adapters.protein import ProteinAdapter
        pa = ProteinAdapter(protein_name='PRNP')
        mech = pa.classify_mechanism({
            'has_stable_fold': True,
            'mutations_destabilizing': True,
            'templated': True
        })
        assert mech['mechanism'] == 'templated_conversion'
