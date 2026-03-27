#!/usr/bin/env python3
"""
Tests for sigma_c.beyond.information -- information-theoretic module.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.beyond.information import (
    bits_lost,
    landauer_cost,
    information_summary,
)


# ------------------------------------------------------------------
# bits_lost
# ------------------------------------------------------------------

class TestBitsLost2:
    def test_result_is_1(self):
        # log2(2) = 1.0
        assert bits_lost(2.0) == pytest.approx(1.0)


class TestBitsLost43:
    def test_result_approx_0_415(self):
        # log2(4/3) ~ 0.41504
        result = bits_lost(4.0 / 3.0)
        assert result == pytest.approx(0.41504, abs=0.001)


class TestLandauerCost:
    def test_positive_for_D_gt_1(self):
        cost = landauer_cost(2.0, T=300.0)
        assert cost > 0, f"Landauer cost = {cost}, expected positive"


class TestBitsLostError:
    def test_raises_for_D_lt_1(self):
        with pytest.raises(ValueError):
            bits_lost(0.5)


# ------------------------------------------------------------------
# information_summary
# ------------------------------------------------------------------

class TestInformationSummary:
    def test_all_keys_present(self):
        result = information_summary(D=2.0, gamma=0.75, T=300.0)
        expected_keys = {
            'bits_lost_per_step',
            'landauer_cost_J',
            'landauer_cost_eV',
            'entropy_production_rate',
            'D',
            'gamma',
            'sigma_product',
            'net_contraction',
            'interpretation',
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )
        assert result['net_contraction'] is True
        assert result['bits_lost_per_step'] == pytest.approx(1.0)
