#!/usr/bin/env python3
"""
Tests for sigma_c.adapters.protein -- ProteinAdapter.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np
from scipy import stats

from sigma_c.adapters.protein import ProteinAdapter


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInitTTR:
    def test_N_and_T(self):
        adapter = ProteinAdapter(protein_name='TTR')
        assert adapter.N == 127
        assert adapter.T == 310


# ------------------------------------------------------------------
# Thermodynamic sigma
# ------------------------------------------------------------------

class TestSigmaThermodynamicAtTm:
    def test_sigma_at_zero_dG(self):
        adapter = ProteinAdapter(protein_name='TTR')
        # sigma = exp(-0 / (N*R*T)) = exp(0) = 1.0
        sigma = adapter.sigma_thermodynamic(0.0, adapter.N)
        assert sigma == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------------
# Mutational sigma
# ------------------------------------------------------------------

class TestSigmaMutationDestabilizing:
    def test_sigma_greater_than_1(self):
        adapter = ProteinAdapter(protein_name='TTR')
        # Positive ddG -> sigma > 1
        sigma = adapter.sigma_mutation(1.0, adapter.N)
        assert sigma > 1.0, f"sigma = {sigma}, expected > 1 for ddG > 0"


class TestSigmaMutationStabilizing:
    def test_sigma_less_than_1(self):
        adapter = ProteinAdapter(protein_name='TTR')
        # Negative ddG -> sigma < 1
        sigma = adapter.sigma_mutation(-1.0, adapter.N)
        assert sigma < 1.0, f"sigma = {sigma}, expected < 1 for ddG < 0"


# ------------------------------------------------------------------
# TTR correlation (sigma vs onset)
# ------------------------------------------------------------------

class TestTTRCorrelation:
    def test_rho_negative(self):
        adapter = ProteinAdapter(protein_name='TTR')
        mutations = [m for m in adapter.TTR_MUTATIONS if m.get('onset') is not None]
        sigmas = [m['sigma'] for m in mutations]
        onsets = [m['onset'] for m in mutations]
        rho, _ = stats.spearmanr(sigmas, onsets)
        assert rho < -0.9, f"Spearman rho = {rho}, expected < -0.9"


# ------------------------------------------------------------------
# SOD1 negative control
# ------------------------------------------------------------------

class TestSOD1NegativeControl:
    def test_weak_or_nonsignificant_correlation(self):
        adapter = ProteinAdapter(protein_name='TTR')
        mutations = adapter.SOD1_MUTATIONS
        sigmas = [m['sigma'] for m in mutations]
        onsets = [m['onset'] for m in mutations]
        rho, p = stats.spearmanr(sigmas, onsets)
        # For a negative control, either |rho| < 0.5 or p > 0.05
        assert abs(rho) < 0.5 or p > 0.05, (
            f"SOD1 rho={rho}, p={p}: expected weak or non-significant correlation"
        )


# ------------------------------------------------------------------
# Onset prediction
# ------------------------------------------------------------------

class TestOnsetPrediction:
    def test_predict_onset_returns_reasonable_age(self):
        adapter = ProteinAdapter(protein_name='TTR')
        # For a mutation with sigma ~ 0.94, onset should be > 30
        onset = adapter.predict_onset(0.94)
        assert onset > 30, f"Predicted onset = {onset}, expected > 30"


# ------------------------------------------------------------------
# Mechanism classification
# ------------------------------------------------------------------

class TestClassifyMechanism:
    def test_stability_driven_for_TTR(self):
        adapter = ProteinAdapter(protein_name='TTR')
        result = adapter.classify_mechanism({
            'has_stable_fold': True,
            'delta_G': 6.0,
            'mutations_destabilizing': True,
        })
        assert result['mechanism'] == 'stability_driven'
        assert result['sigma_applicable'] is True


# ------------------------------------------------------------------
# Diagnostics and validation
# ------------------------------------------------------------------

class TestDiagnose:
    def test_returns_status(self):
        adapter = ProteinAdapter(protein_name='TTR')
        diag = adapter.diagnose()
        assert 'status' in diag
        assert diag['status'] in ('ok', 'warning', 'error')


class TestValidate:
    def test_core_checks_pass(self):
        adapter = ProteinAdapter(protein_name='TTR')
        val = adapter.validate_techniques()
        assert val['checks']['N_set'] is True
        assert val['checks']['T_positive'] is True
        assert val['checks']['R_correct'] is True
