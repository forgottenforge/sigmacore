#!/usr/bin/env python3
"""
Tests for sigma_c.core.contraction -- contraction geometry module.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import pytest
import numpy as np

from sigma_c.core.contraction import (
    v2,
    odd_part,
    embedding_depth,
    single_step_map,
    cycle_map,
    compute_contraction_defect,
    compute_drift,
    compute_drift_v2_sum,
    classify_map,
    countdown_decomposition,
    deterministic_trajectory,
)


# ------------------------------------------------------------------
# Basic number-theoretic helpers
# ------------------------------------------------------------------

class TestV2:
    def test_v2_8(self):
        assert v2(8) == 3

    def test_v2_12(self):
        assert v2(12) == 2

    def test_v2_1(self):
        assert v2(1) == 0


class TestOddPart:
    def test_odd_part_12(self):
        assert odd_part(12) == 3

    def test_odd_part_7(self):
        assert odd_part(7) == 7


class TestEmbeddingDepth:
    def test_ed_7(self):
        # 7 = 0b111, ed = v2(8) = 3
        assert embedding_depth(7) == 3

    def test_ed_3(self):
        # 3 = 0b11, ed = v2(4) = 2
        assert embedding_depth(3) == 2

    def test_ed_1(self):
        # 1 = 0b1, ed = v2(2) = 1
        assert embedding_depth(1) == 1


# ------------------------------------------------------------------
# Map functions
# ------------------------------------------------------------------

class TestSingleStepMap:
    def test_f_3(self):
        # f(3) = odd(3*3+1) = odd(10) = 5
        assert single_step_map(3) == 5

    def test_f_5(self):
        # f(5) = odd(3*5+1) = odd(16) = 1
        assert single_step_map(5) == 1

    def test_f_7(self):
        # f(7) = odd(3*7+1) = odd(22) = 11
        assert single_step_map(7) == 11


class TestCycleMap:
    def test_F_3(self):
        # ed(3)=2, L=2; F(3)=odd(3^2*(3+1)/2^2 - 1) = odd(9*1 -1) = odd(8) = 1
        assert cycle_map(3) == 1

    def test_F_7(self):
        # ed(7)=3, L=3; F(7) = odd(3^3*(7+1)/2^3 -1) = odd(27*1 -1) = odd(26) = 13
        assert cycle_map(7) == 13


# ------------------------------------------------------------------
# Contraction defect and drift
# ------------------------------------------------------------------

class TestDMLowerBound:
    """D_M should be >= 4/3 for the Collatz single-step map at M=4..12."""

    @pytest.mark.parametrize("M", range(4, 13))
    def test_D_M_lower_bound(self, M):
        # timeout: each call for M<=12 finishes well under 5s
        f = lambda n: single_step_map(n, 3, 1)
        D = compute_contraction_defect(f, M)
        assert D >= 4 / 3, f"D_{M} = {D} is below 4/3"


class TestDriftConverges:
    """gamma for q=3 single-step map approaches 0.75 as M grows."""

    def test_drift_converges(self):
        # Check at M=10 and M=12 -- should be close to 3/4
        for M in (10, 12):
            gamma = compute_drift_v2_sum(3, M)
            assert abs(gamma - 0.75) < 0.1, (
                f"gamma at M={M} is {gamma}, expected near 0.75"
            )


# ------------------------------------------------------------------
# Map classification
# ------------------------------------------------------------------

class TestClassifyCollatz:
    def test_classify_collatz(self):
        # Collatz: D > 1, gamma < 1, no known non-trivial cycles -> convergent
        result = classify_map(D=1.7, gamma=0.75, has_cycles=False)
        assert result['prediction'] == 'convergent'


class TestClassify5n1:
    def test_classify_5n1(self):
        # 5n+1: D > 1, gamma > 1 -> divergent
        result = classify_map(D=1.4, gamma=1.25, has_cycles=False)
        assert result['prediction'] == 'divergent'


class TestClassify3nMinus1:
    def test_classify_3n_minus_1(self):
        # 3n-1: D > 1, gamma < 1, has cycles -> convergent_to_cycles
        result = classify_map(D=1.33, gamma=0.75, has_cycles=True)
        assert result['prediction'] == 'convergent_to_cycles'


# ------------------------------------------------------------------
# Countdown theorem
# ------------------------------------------------------------------

class TestCountdownTheorem:
    def test_countdown_ed_decreases(self):
        """During a countdown phase, ed should decrease by 1 at each step
        for starting values with ed >= 2."""
        n = 7  # ed(7) = 3
        ed_prev = embedding_depth(n)
        current = n
        for _ in range(ed_prev - 1):
            current = single_step_map(current, 3, 1)
            ed_cur = embedding_depth(current)
            assert ed_cur == ed_prev - 1, (
                f"Expected ed to drop from {ed_prev} to {ed_prev - 1}, "
                f"got {ed_cur} at value {current}"
            )
            ed_prev = ed_cur


# ------------------------------------------------------------------
# Deterministic trajectory
# ------------------------------------------------------------------

class TestDeterministicTrajectory:
    def test_closed_form_matches_iterative(self):
        """deterministic_trajectory should match iterated single_step_map."""
        for n in (7, 15, 31, 63):
            traj_closed = deterministic_trajectory(n)
            ed = embedding_depth(n)
            if ed < 2:
                continue
            # Build iterative trajectory over ed-1 steps
            traj_iter = [n]
            current = n
            for _ in range(ed - 1):
                current = single_step_map(current, 3, 1)
                traj_iter.append(current)
            assert traj_closed == traj_iter, (
                f"Mismatch for n={n}: closed={traj_closed}, iter={traj_iter}"
            )
